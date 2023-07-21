"""
Copy the dicom files from the raw directory to the destination directory.

Original dicoms are stored in
/neodata/oxr/innocare/raw/NTUH-HC/{PatientID}/I_{StudyNo}_{SeriesNo}_{InstanceNo}.dcm

Some dicoms have problems, so we have to use the fixed dicoms.
Fixed dicoms are stored in
/neodata/oxr/innocare/raw/fix/{PatientID}/I_{StudyNo}_{SeriesNo}_{InstanceNo}.dcm
/neodata/oxr/innocare/raw/fix0428/{PatientID}/I_{StudyNo}_{SeriesNo}_{InstanceNo}.dcm
/neodata/oxr/innocare/raw/fix0505/{PatientID}/I_{StudyNo}_{SeriesNo}_{InstanceNo}.dcm

Rules:
    1. Replace the SeriesNo for each view with rule: S4 > S3 > S2 (LAT) > S1 (PA)
    2. If multiple StudyNo exists for this SeriesNo, use the largest one

Destination directory structure:
/neodata/oxr/innocare/dicom/{PatientID}/image_{view}_{channel}.dcm
"""

import glob
import os
import re
from multiprocessing import Pool
from shutil import copyfile

import pandas as pd
import pydicom
from tqdm import tqdm

DST_DIR = "/neodata/oxr/innocare/dicom"
SRC_DIRS = [
    "/neodata/oxr/innocare/raw/fix0505",
    "/neodata/oxr/innocare/raw/fix0428",
    "/neodata/oxr/innocare/raw/fix",
    "/neodata/oxr/innocare/raw/NTUH-HC",
]

# Get the CAC_scores.csv from 0_match_cac.py
uids = pd.read_csv("/neodata/oxr/innocare/CAC_scores.csv", usecols=["UID"])["UID"]


def prepare_dicom(uid):
    """"""
    try:
        # Find the src dir, with priority fix0505 > fix0428 > fix > NTUH-HC
        src_dir = None
        for src_dir in SRC_DIRS:
            if os.path.exists(os.path.join(src_dir, uid)):
                break

        # Replace the SeriesNo for each view with rule: S4 > S3 > S2 (LAT) > S1 (PA)
        for s in range(1, 5):
            s = f"S{s}"

            # Take the largest StudyNo with >= 3 dcms
            dcm_paths = glob.glob(f"{src_dir}/{uid}/I_*_{s}_*.dcm")
            study_paths_mapping = {}
            for dcm_path in dcm_paths:
                study_no = re.search(r"I_([0-9]+)_", dcm_path).group(1)
                if study_no in study_paths_mapping:
                    study_paths_mapping[study_no].append(dcm_path)
                else:
                    study_paths_mapping[study_no] = [dcm_path]
            study_paths_mapping = {
                study_no: dcm_paths
                for study_no, dcm_paths in study_paths_mapping.items()
                if len(dcm_paths) >= 3
            }
            if len(study_paths_mapping) == 0:
                continue
            study_no = max(study_paths_mapping)
            dcm_paths = study_paths_mapping[study_no]

            # Check which view is this SeriesNo
            dcm_paths = [p for p in dcm_paths if re.search(r"_001\.dcm", p)]
            if len(dcm_paths) != 1:
                raise ValueError(
                    f"{len(dcm_paths)} of instance 001 found for study {study_no}"
                )
            dcm_path = dcm_paths[0]
            if s == "S1":
                front = dcm_path
            elif s == "S2":
                lateral = dcm_path
            else:
                dcm = pydicom.dcmread(dcm_path)
                if dcm.SeriesDescription == "Chest PA":
                    front = dcm_path
                elif dcm.SeriesDescription == "Chest LAT":
                    lateral = dcm_path
                else:
                    raise ValueError(
                        "Unknown series description: " + dcm.SeriesDescription
                    )

        # Get the src files
        src_files = {
            "image_front_combined": front,
            "image_front_soft": front.replace("_001.dcm", "_002.dcm"),
            "image_front_hard": front.replace("_001.dcm", "_003.dcm"),
            "image_lateral_combined": lateral,
            "image_lateral_soft": lateral.replace("_001.dcm", "_002.dcm"),
            "image_lateral_hard": lateral.replace("_001.dcm", "_003.dcm"),
        }

        # Copy the dicom files from the src to the dst
        os.makedirs(f"{DST_DIR}/{uid}", exist_ok=True)
        for dst_filename, src_file in src_files.items():
            copyfile(src_file, f"{DST_DIR}/{uid}/{dst_filename}.dcm")

    except Exception as e:
        print(uid, e)


# create a pool of processes
with Pool(8) as pool:
    # apply the function to the list using the pool and tqdm progress bar
    list(tqdm(pool.imap_unordered(prepare_dicom, uids), total=len(uids)))

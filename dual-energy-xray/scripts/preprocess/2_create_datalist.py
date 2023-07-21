import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

DICOM_DIR = "/neodata/oxr/innocare/dicom"

df = pd.read_csv(
    "/neodata/oxr/innocare/CAC_scores.csv", usecols=["UID", "鈣化指數"], index_col="UID"
)
uids = df.index

datalist = []
for uid in uids:
    # Add all images paths to data
    data = {
        channel: uid + "/" + channel + ".dcm"
        for channel in [
            "image_front_combined",
            "image_front_soft",
            "image_front_hard",
            "image_lateral_combined",
            "image_lateral_soft",
            "image_lateral_hard",
        ]
    }
    # Check if image path exists
    for path in data.values():
        path = os.path.join(DICOM_DIR, path)
        if not os.path.exists(path):
            raise ValueError(f"{path} not found")

    data["cac_score"] = df.loc[uid, "鈣化指數"]
    datalist.append(data)


train, valid_test = train_test_split(
    datalist,
    test_size=0.2,
    random_state=42,
    stratify=[data["cac_score"] > 400 for data in datalist],
)
valid, test = train_test_split(
    valid_test,
    test_size=0.5,
    random_state=42,
    stratify=[data["cac_score"] > 400 for data in valid_test],
)

datalist = {"train": train, "valid": valid, "test": test}

with open("/neodata/oxr/innocare/datalist.json", "w") as fp:
    json.dump(datalist, fp, indent=4)

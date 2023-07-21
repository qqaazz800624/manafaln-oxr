import sys
import tempfile
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import SimpleITK as sitk
from monai import transforms as T
from monai.data.decathlon_datalist import load_decathlon_datalist
from monai.utils import PostFix
from monai.utils.misc import ImageMetaKey
from radiomics.featureextractor import RadiomicsFeatureExtractor
from tqdm import tqdm

from manafaln.utils import load_yaml

sys.path.append(".")
from custom.transforms.fix_horizontal_flip import FixHorizontalFlipd
from custom.transforms.heart_seg import HeartSegmentationd


def load_datalist(datalist_path: str, base_dir: str) -> List[Dict[str, Any]]:
    datalist = []
    for key in load_yaml(datalist_path).keys():
        subset = load_decathlon_datalist(
            data_list_file_path=datalist_path,
            data_list_key=key,
            base_dir=base_dir,
        )
        datalist = datalist + subset
    return datalist


def prepare_transforms():
    transforms = T.Compose(
        [
            T.LoadImaged(
                keys=["image_front_combined", "image_front_soft", "image_front_hard"],
                ensure_channel_first=True,
                reader="PydicomReader",
            ),
            T.Resized(
                keys=["image_front_combined", "image_front_soft", "image_front_hard"],
                spatial_size=[512, 512],
            ),
            T.ScaleIntensityd(
                keys=["image_front_combined", "image_front_soft", "image_front_hard"]
            ),
            FixHorizontalFlipd(
                key="image_front_combined",
                spatial_axis=0,
                model_weight="custom/transforms/flip_combined.ckpt",
                model_config={
                    "name": "DenseNet121",
                    "args": {
                        "spatial_dims": 2,
                        "in_channels": 1,
                        "out_channels": 1,
                        "pretrained": False,
                    },
                },
            ),
            FixHorizontalFlipd(
                key="image_front_soft",
                spatial_axis=0,
                model_weight="custom/transforms/flip_soft.ckpt",
                model_config={
                    "name": "DenseNet121",
                    "args": {
                        "spatial_dims": 2,
                        "in_channels": 1,
                        "out_channels": 1,
                        "pretrained": False,
                    },
                },
            ),
            FixHorizontalFlipd(
                key="image_front_hard",
                spatial_axis=0,
                model_weight="custom/transforms/flip_hard.ckpt",
                model_config={
                    "name": "DenseNet121",
                    "args": {
                        "spatial_dims": 2,
                        "in_channels": 1,
                        "out_channels": 1,
                        "pretrained": False,
                    },
                },
            ),
            T.ConcatItemsd(
                keys=["image_front_combined", "image_front_soft", "image_front_hard"],
                name="heart",
            ),
            HeartSegmentationd(
                key="heart",
                heart_only=True,
                model_config={
                    "name": "DeepLabV3Plus",
                    "path": "segmentation_models_pytorch",
                    "args": {
                        "in_channels": 3,
                        "classes": 6,
                        "encoder_name": "tu-resnest50d",
                        "encoder_weights": None,
                    },
                },
                model_weight="lightning_logs/seg/checkpoints/best_model.ckpt",
            ),
            T.SqueezeDimd(
                keys=["image_front_combined", "image_front_soft", "image_front_hard"],
                dim=0,
            ),
            T.Transposed(
                keys=[
                    "image_front_combined",
                    "image_front_soft",
                    "image_front_hard",
                    "heart",
                ],
                indices=[1, 0],
            ),
            T.EnsureTyped(keys="heart", data_type="numpy", dtype="int"),
            T.EnsureTyped(
                keys=["image_front_combined", "image_front_soft", "image_front_hard"],
                data_type="numpy",
            ),
        ]
    )
    return transforms


def extract_radiomics(
    image: np.ndarray, mask: np.ndarray, extractor: RadiomicsFeatureExtractor
):
    """
    Args:
        image (np.ndarray): image with shape [W, H] with values 0 to 1.
        mask (np.ndarray): mask with shape [W, H] with values 0 and 1.
    """
    # Convert the NumPy array to a SimpleITK image
    itk_image = sitk.GetImageFromArray(image)
    itk_mask = sitk.GetImageFromArray(mask)

    # Create a temporary file with a .nii.gz extension
    file_itk_image = tempfile.NamedTemporaryFile(suffix=".nii.gz").name
    file_itk_mask = tempfile.NamedTemporaryFile(suffix=".nii.gz").name

    # Save the image as a nifti file
    sitk.WriteImage(itk_image, file_itk_image)
    sitk.WriteImage(itk_mask, file_itk_mask)

    features = extractor.execute(file_itk_image, file_itk_mask)

    features = {
        k: (v.item() if isinstance(v, np.ndarray) else v) for k, v in features.items()
    }

    return features


def save_to_csv(features_list, csv_path):
    # Convert the features into a pandas DataFrame
    df = pd.DataFrame(features_list)

    # Save the DataFrame as CSV
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    datalist_path = "data/neodata/datalist.json"
    base_dir = "data/neodata/dicom"

    transforms = prepare_transforms()

    extractor = RadiomicsFeatureExtractor(
        binWidth=0.1, resampledPixelSpacing=None, interpolator=sitk.sitkBSpline
    )

    datalist = load_datalist(datalist_path, base_dir)
    features_list = []
    for data in tqdm(datalist):
        data = transforms(data)

        # extract uid from data/neodata/dicom/{uid}/image_front_combined.dcm
        uid = data[PostFix.meta("image_front_combined")][
            ImageMetaKey.FILENAME_OR_OBJ
        ].split("/")[-2]
        mask = data["heart"]

        # Extract Radiomic features for each channel
        features = {"uid": uid}
        for channel in ["image_front_combined", "image_front_soft", "image_front_hard"]:
            image = data[channel]
            channel_features = extract_radiomics(image, mask, extractor)
            channel_features = {
                channel + "_" + k: v for k, v in channel_features.items()
            }
            features.update(channel_features)
        features_list.append(features)

    save_to_csv(features_list, "scripts/analytics/radiomics.csv")

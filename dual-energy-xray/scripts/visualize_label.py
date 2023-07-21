import sys

from monai.apps.detection.transforms.array import BoxToMask
from monai.transforms import LoadImage, ScaleIntensity, ToNumpy, MapLabelValue

sys.path.append(".")
from custom.transforms.parse_labelme import ParseLabelMeDetectionLabel
from manafaln.transforms import AnyDim, LoadJSON, OverlayMask, SaveImage


def main(image_file, label_file, filename):
    label = LoadJSON(json_only=True)(label_file)
    image = LoadImage(image_only=True, ensure_channel_first=True)(image_file)
    image = ScaleIntensity()(image)

    bboxes, labels = ParseLabelMeDetectionLabel()(label)
    bboxes = ToNumpy()(bboxes)
    labels = ToNumpy()(labels)
    labels = MapLabelValue(["LAD", "LCX", "RCA"], [1, 1, 1])(labels)

    masks = BoxToMask(bg_label=0)(bboxes, labels, image.shape[1:])
    masks = AnyDim(dim=0, keep_dim=True)(masks)
    visual = OverlayMask(alpha=0.2)(image, masks)

    SaveImage(
        output_postfix=filename,
        separate_folder=False,
        output_ext="png",
        output_dtype="uint8",
        scale=255,
    )(visual)


if __name__ == "__main__":
    files = {
        "I_00000018_S1_001": (
            "/neodata/oxr/innocare/raw/label-dev/018_20221206/I_00000018_S1_001/I_00000018_S1_001_combined.json",
            "/neodata/oxr/innocare/raw/label-dev/018_20221206/I_00000018_S1_001/I_00000018_S1_001_combined.png",
        ),
        "I_00000018_S2_001": (
            "/neodata/oxr/innocare/raw/label-dev/018_20221206/I_00000018_S2_001/I_00000018_S2_001_combined.json",
            "/neodata/oxr/innocare/raw/label-dev/018_20221206/I_00000018_S2_001/I_00000018_S2_001_combined.png",
        ),
        "I_00000021_S1_001": (
            "/neodata/oxr/innocare/raw/label-dev/021_20221209_葉雲道/I_00000021_S1_001/I_00000021_S1_001_combined.json",
            "/neodata/oxr/innocare/raw/label-dev/021_20221209_葉雲道/I_00000021_S1_001/I_00000021_S1_001_combined.png",
        ),
        "I_00000021_S2_001": (
            "/neodata/oxr/innocare/raw/label-dev/021_20221209_葉雲道/I_00000021_S2_001/I_00000021_S2_001_combined.json",
            "/neodata/oxr/innocare/raw/label-dev/021_20221209_葉雲道/I_00000021_S2_001/I_00000021_S2_001_combined.png",
        ),
    }

    for filename, (label_file, image_file) in files.items():
        main(image_file, label_file, filename)

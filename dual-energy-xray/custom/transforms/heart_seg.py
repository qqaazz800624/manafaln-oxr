import torch
from monai.transforms import MapTransform, MaskIntensity

from manafaln.core.builders import ModelBuilder


class HeartSegmentationd(MapTransform):
    def __init__(
        self, key: str, model_config: dict, model_weight: str, heart_only: bool = False
    ):
        self.key = key
        self.heart_only = heart_only

        self.model: torch.nn.Module = ModelBuilder()(model_config)

        model_weight = torch.load(model_weight, map_location="cpu")["state_dict"]
        for k in list(model_weight.keys()):
            k_new = k.replace(
                "model.", "", 1
            )  # e.g. "model.conv.weight" => conv.weight"
            model_weight[k_new] = model_weight.pop(k)

        self.model.load_state_dict(model_weight)
        self.model.eval()

    def __call__(self, data):
        d = dict(data)
        # img shape (C, W, H) => (B, C, W, H)
        img = d[self.key].unsqueeze(0)
        logit = self.model(img)
        mask_heart = torch.sigmoid(logit)[0, 2] > 0.5  # take segmentation mask of heart
        if self.heart_only:
            d[self.key] = mask_heart
        else:
            mask_heart = mask_heart.unsqueeze(0)
            maskintensity = MaskIntensity(mask_data=mask_heart)
            d[self.key] = maskintensity(img.squeeze(0))
        return d

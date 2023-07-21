from typing import Union

import torch
from monai.transforms import Flip, MapTransform

from manafaln.core.builders import ModelBuilder


class FixHorizontalFlipd(MapTransform):
    def __init__(
        self,
        key: str,
        model_config: dict,
        model_weight: str,
        spatial_axis: int = -1,
    ):
        self.key = key

        self.flipper = Flip(spatial_axis=spatial_axis)

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
        # img shape (C, W, H) => (B, C, H, W)
        img = d[self.key].unsqueeze(0)
        flip = self.model(img)
        if torch.sigmoid(flip) > 0.5:
            d[self.key] = self.flipper(d[self.key])
        return d

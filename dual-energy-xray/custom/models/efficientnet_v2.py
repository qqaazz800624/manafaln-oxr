
import torch
import torch.nn as nn
from manafaln.core.builders import ModelBuilder
from torchvision.models import efficientnet_v2_s


class torchvisionmodels(nn.Module):
    def __init__(
        self, 
        model_config: dict,
        num_classes: int,
    ):
        super().__init__()
        self.backbone: torch.nn.Module = ModelBuilder()(model_config)
        self.linear = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.linear(x)
        return x
    
# class efficientnetv2(nn.Module):
#     def __init__(
#         self, 
#         num_classes: int,
#     ):
#         super().__init__()
#         self.backbone = efficientnet_v2_s(weights='IMAGENET1K_V1')
#         self.linear = nn.Linear(1000, num_classes)

#     def forward(self, x):
#         x = self.backbone(x)
#         x = self.linear(x)
#         return x


from typing import Dict

import torch
from monai.transforms import Resize

from manafaln.apps.utils import build_workflow
from manafaln.core.transforms import build_transforms
from manafaln.utils import get_items, load_yaml, update_items


class Model:
    def __init__(self, ckpt_path, config, gpu_id=None):
        self.set_device(gpu_id)
        self.set_model(ckpt_path, config)

    def set_device(self, gpu_id):
        """
        Set device from gpu_id.
        """
        if gpu_id is not None and torch.cuda.is_available():
            self.device = "cuda:" + str(gpu_id)
        else:
            self.device = "cpu"

    def set_model(self, ckpt_path, config):
        """
        Initializes model from config.
        """
        self.model = build_workflow(
            {
                "name": "SupervisedLearningV2",
                "settings": {},
                "components": {"model": config},
            },
            ckpt=ckpt_path,
        )
        self.model.eval()
        self.model.to(device=self.device)

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        inputs = get_items(data, self.model.model_input_keys)
        inputs = self.model.transfer_batch_to_device(inputs, self.device, 0)
        prediction = self.model(*inputs)
        data = update_items(data, self.model.model_output_keys, prediction)
        return data


class InferenceHandler:
    def __init__(self, seg_ckpt, cac_ckpt, config, gpu_id=None):
        config = load_yaml(config)
        self.seg_model = Model(seg_ckpt, config["seg_model"], gpu_id=gpu_id)
        self.cac_model = Model(cac_ckpt, config["cac_model"], gpu_id=gpu_id)
        self.preprocess = build_transforms(config["preprocess"])
        self.midprocess = build_transforms(config["midprocess"])
        self.postprocess = build_transforms(config["postprocess"])

    def __call__(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Make a prediction on input data.
        Args:
            data (Dict[str, torch.Tensor]): input data. Containing
                "image_front_combined": torch.Tensor, dtype float, shape (W, H)
                "image_front_soft": torch.Tensor, dtype float, shape (W, H)
                "image_front_hard": torch.Tensor, dtype float, shape (W, H)
        Returns:
            Dict[str, torch.Tensor]: predictions. Containing
                "pred_cac": torch.Tensor, dtype float, shape ()
                "pred_seg_heart": torch.Tensor, dtype bool, shape (W, H)
        """
        input_shape = data["image_front_combined"].shape

        data = self.preprocess(data)
        data = self.seg_model(data)
        data = self.midprocess(data)
        data = self.cac_model(data)
        data = self.postprocess(data)

        # Resize heart segmentation back to input size
        data["pred_seg_heart"] = Resize(spatial_size=input_shape, mode="nearest")(
            data["pred_seg_heart"].unsqueeze(0)
        ).squeeze(0)

        return data

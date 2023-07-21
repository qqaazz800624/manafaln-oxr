import os

import numpy as np
from handler import InferenceHandler
from monai.transforms import LoadImaged


def inference(inputs, seg_ckpt, cac_ckpt, config, gpu_id):
    handler = InferenceHandler(seg_ckpt, cac_ckpt, config, gpu_id)

    keys = ["image_front_combined", "image_front_soft", "image_front_hard"]
    inputs = {k: os.path.join(inputs, f"{k}.dcm") for k in keys}
    inputs = LoadImaged(keys=keys, reader="PydicomReader")(inputs)

    preds = handler(inputs)

    pred_heart = preds["pred_seg_heart"]
    pred_heart = pred_heart.cpu().numpy()
    pred_heart = pred_heart.T  # (W, H) -> (H, W)
    pred_heart = (pred_heart * 255).astype(np.uint8)  # [0, 1] -> [0, 255]

    pred_cac = preds["pred_cac"]
    pred_cac = pred_cac.item()

    return pred_heart, pred_cac


if __name__ == "__main__":
    import argparse

    from skimage.io import imsave

    args = argparse.ArgumentParser()
    args.add_argument("--inputs", type=str)
    args.add_argument("--output", type=str)
    args.add_argument("--seg_ckpt", type=str, default="checkpoints/heart_seg.ckpt")
    args.add_argument("--cac_ckpt", type=str, default="checkpoints/cac_clf.ckpt")
    args.add_argument("--config", type=str, default="config.yaml")
    args.add_argument("--gpu_id", type=int, default=0)
    args = args.parse_args()

    pred_heart, pred_cac = inference(
        args.inputs,
        args.seg_ckpt,
        args.cac_ckpt,
        args.config,
        args.gpu_id,
    )

    imsave(args.output, pred_heart)
    print(pred_cac)

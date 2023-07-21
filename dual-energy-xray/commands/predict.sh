#!/bin/bash

python -m manafaln.apps.predict \
    -c ${2:-"configs/1_heart_segmentation/predict.yaml"} \
    -f lightning_logs/$1/checkpoints/best_model.ckpt

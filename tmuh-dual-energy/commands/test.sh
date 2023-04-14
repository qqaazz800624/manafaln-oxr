#!/bin/bash

python -m manafaln.apps.validate \
    -c ${2:-"config/train.yaml"} \
    -f lightning_logs/version_$1/checkpoints/best_model.ckpt

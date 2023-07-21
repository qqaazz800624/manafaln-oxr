#!/bin/bash

python -m manafaln.apps.validate \
    -c ${2:-"configs/2_cac_classification/test.yaml"} \
    -f lightning_logs/$1/checkpoints/best_model.ckpt

#!/bin/bash

python -m manafaln.apps.train \
    -s 42 \
    -c ${1:-"configs/2_cac_classification/train.yaml"}

#!/bin/bash

python -m manafaln.utils.cross_validation \
    -c ${2:-"configs/2_cac_classification/train_b.yaml"}\
    -k ${3:-10}\
    -e \
    -v ${1:-7} 


python -m manafaln.utils.cross_validation \
    -c ${2:-"configs/2_cac_classification/train_b.yaml"}\
    -s \
    -e \
    -v ${1:-4}
    
# Dual Energy Cardiovascular Calcification Classification

## About The Project

This is the code to develop deep learning model to predict Coronary Artery Calcium (CAC) Score from Dual Energy Chest X-ray.

## Table of Contents

- [Dual Energy Cardiovascular Calcification Classification](#dual-energy-cardiovascular-calcification-classification)
  - [About The Project](#about-the-project)
  - [Table of Contents](#table-of-contents)
  - [Built with](#built-with)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Setting Up the Workspace](#setting-up-the-workspace)
  - [Setting Up the Environment](#setting-up-the-environment)
- [Data](#data)
  - [Preparing Data](#preparing-data)
  - [Preprocessing Data](#preprocessing-data)
- [Training](#training)
  - [Fixing Horizontal Flip](#fixing-horizontal-flip)
  - [Heart Segmentation](#heart-segmentation)
  - [CAC Score](#cac-score)
- [Inference](#inference)

## Built with

* [MONAI](https://monai.io/) - A PyTorch-based, open-source framework for deep learning in medical imaging.
* [Lightning](https://www.lightning.ai/) - A PyTorch-based, open-source framework for deep learning research that aims to standardize the implementation of common research tasks.
* [Manafaln](https://gitlab.com/nanaha1003/manafaln) - A package developed by our lab that provides a highly flexible configuration-based training tool for model training with MONAI and Lightning frameworks in medical imaging.

# Installation

## Requirements

* [Miniconda](https://docs.conda.io/en/latest/miniconda.html) version 22.11.1 or later.

## Setting Up the Workspace

1. Clone this repository by running the following command in your terminal:
    ```sh
    git clone https://github.com/HenleyHwang/cardiovascular-calcification.git
    ```

2. Change directory into the folder
    ```sh
    cd cardiovascular-calcification
    ```

3. After cloning the repository, your project workspace should have the following folder structure:
    ```
    cardiovascular-calcification
    ├── commands
    ├── configs
    ├── custom
    ├── inference
    ├── scripts
    ├── README.md
    └── requirements.txt
    ```

This folder structure is a standard layout for a `Manafaln` project, with each folder serving a specific purpose:

- `commands`: This folder contains shell scripts for executing specific tasks, such as model training or testing.
- `configs`: This folder contains YAML configuration files for the training, testing, and inference processes.
- `custom`: This folder contains custom modules that can be easily imported by specifying the `path` in the configuration file.
- `data`: This folder contains the project's data.
- `inference`: This folder contains sample code for inference.
- `lightning_logs`: This folder stores training logs and model checkpoints.
- `manafaln`: This folder contains the source code for the `Manafaln` package.
- `scripts`: This folder contains miscellaneous Python scripts for data processing, evaluation, or visualization.
- `README.md`: A file that explains the purpose of the project, provides installation instructions, and outlines usage guidelines.
- `requirements.txt`: A file containing a list of required Python packages and their versions. Package managers like `pip` use this file to install the necessary dependencies.

## Setting Up the Environment

1. Create a new `conda` environment by running the following commands in your terminal:
    ```sh
    conda create -n cardiovascular-calcification python==3.9.12
    conda activate cardiovascular-calcification
    ```
    This creates a new environment called cardiovascular-calcification with Python version 3.9.12 and activates it.

2. Install required packages by running the following command in your terminal:
    ```sh
    pip install -r requirements.txt
    ```
    This installs the required packages into your current environment.

# Data

## Preparing Data

1. Create an empty folder to store data
    ```sh
    mkdir data
    ```

2. Create a soft link to the data on cluster
    ```sh
    ln -s /neodata/oxr/innocare data/neodata
    ```

3. Once done, you should see the following folder structure
    ```
    data
    └── neodata
        └── raw
            ├── fix
            ├── fix0428
            ├── fix0505
            ├── NTUH-HC
            ├── 111-060-F臨床試驗(睿生光電).xlsx
            └── 收案表單.csv
    ```

Each folder serving a specific purpose:

- `fix`, `fix0428`, `fix0505` : These folders contains fixed DICOMs.
- `NTUH-HC`: This folder contains raw DICOM for each patient.
- `111-060-F臨床試驗(睿生光電).xlsx`: This Excel contains CAC scores for each patient in AN numbers.
- `收案表單.csv`: This is a csv to match AN numbers with DICOM uids.

## Preprocessing Data

1. Match the CAC scores for each DICOM uids to `data/neodata/CAC_scores.csv`
    ```sh
    python scripts/preprocess/0_match_cac.py
    ```

2. Prepare and fixing DICOMS to `data/neodata/dicom`
    ```sh
    python scripts/preprocess/1_prepare_dicom.py
    ```

3. Create data list to `data/neodata/datalist.json`
    ```sh
    python scripts/create_datalist.py
    ```
    This creates a `datalist.json` file in the data folder, which contains the file paths for the each DICOM channels and CAC score for each patient.

    The sample layout is as follows:

    <details>
    <summary>Example</summary>

    ```json
    {
        "train": [
            {
            "image_front_combined": "xxx_xxxxxxxx/image_front_combined.dcm",
            "image_front_soft": "xxx_xxxxxxxx/image_front_soft.dcm",
            "image_front_hard": "xxx_xxxxxxxx/image_front_hard.dcm",
            "image_lateral_combined": "xxx_xxxxxxxx/image_lateral_combined.dcm",
            "image_lateral_soft": "xxx_xxxxxxxx/image_lateral_soft.dcm",
            "image_lateral_hard": "xxx_xxxxxxxx/image_lateral_hard.dcm",
            "cac_score": xxx.x
        },
            ...
        ],
        "valid": [
            ...
        ],
        "test":[
            ...
        ]
    }
    ```

    </details>
    <br>

# Training

## Fixing Horizontal Flip
Some PA view of DE images are horizontally flipped, we need to train a classification model to determined if the image is horizontally flipped to fix it. A total of 3 models are trained to fix each channel (combined, soft, and hard). This model is trained using TMUH dataset, at `/neodata/oxr/tmuh`.

1. Train model to determined whether to flip for combined channel
    ```sh
    bash commands/train.sh configs/0_fix_horizontal_flip/combined.yaml
    ```
2.  Copy the trained model weight to `custom/transforms/`, note to change <version_no> to version number of this model
    ```sh
    cp lightning_logs/version_<version_no>/checkpoints/best_model.ckpt custom/transforms/ flip_combined.ckpt
    ```

3. Repeat step 1 and step 2 for soft channel.
    ```sh
    bash commands/train.sh configs/0_fix_horizontal_flip/soft.yaml
    cp lightning_logs/version_<version_no>/checkpoints/best_model.ckpt custom/transforms/ flip_soft.ckpt
    ```

4. Repeat step 1 and step 2 for hard channel.
    ```sh
    bash commands/train.sh configs/0_fix_horizontal_flip/hard.yaml
    cp lightning_logs/version_<version_no>/checkpoints/best_model.ckpt custom/transforms/ flip_hard.ckpt
    ```

## Heart Segmentation
NTUH-HC dataset does not contain segmentation label for heart, so the heart segmentation model is trained on TMUH dataset. Since only PA view is available for TMUH dataset, we only train heart segmentation model for PA view.

1. Train heart segmentation model using TMUH dataset.
    ```sh
    bash commands/train.sh configs/1_heart_segmentation/train.yaml
    ```

2. Inference heart segmentation for NTUH-HC dataset.
    ```sh
    bash commands/predict.sh version_<version_no> configs/1_heart_segmentation/predict.yaml
    ```
    The prediction results can be found at `data/neodata/pred`

## CAC Score
Train a classification or linear regression model to predict CAC score.

1. Train CAC classification model.
    ```sh
    bash commands/train.sh
    ```

2. Test the performance of the model.
    ```sh
    bash commands/test.sh version_<version_no>
    ```

# Inference
InnoCare will develop the UI for CAC prediction. We need to provide a sample code to use our trained model for them. Currently, only heart segmentation have sample code, at `inference/`. The folder structure is as follows:
```
inference
├── checkpoints
│   ├── flip_combined.ckpt
│   ├── flip_hard.ckpt
│   ├── flip_soft.ckpt
│   └── heart_seg.ckpt
├── configs
│   └── heart_seg.yaml
├── samples
├── custom
├── handler.py
├── inference.py
├── README.md
└── requirements.txt
```

Each folder serving a specific purpose:

- `checkpoints` : This folders contains checkpoints for models.
- `configs`: This folder contains config to build workflow, preprocess transforms, and postprocess transforms.
- `custom`: Same custom folder as project root, but without model weights.
- `samples`: This folder contains sample DICOMs for the sample code.
- `handler.py`: The python script that build workflow, preprocess data, inference and postprocess data.
- `inference.py`: A sample code to use the handler.
- `README.md`: A tutorial to run the sample code.
- `requirements.txt`: Same requirement folder as project root.

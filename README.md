<div align="center">
    <h2>
       DynamicVis: An Efficient and General Visual Foundation Model for Remote Sensing Image Understanding
    </h2>
</div>
<br>

<div align="center">
  <img src="resources/DynamicVis.png" width="800"/>
</div>
<br>
<div align="center">
  <a href="https://github.com/KyanChen/DynamicVis">
    <span style="font-size: 20px; ">Homepage</span>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2503.16426">
    <span style="font-size: 20px; ">arXiv</span>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="resources/DynamicVis.pdf">
    <span style="font-size: 20px; ">PDF</span>
  </a>
</div>
<br>
<br>

[![GitHub stars](https://badgen.net/github/stars/KyanChen/DynamicVis)](https://github.com/KyanChen/DynamicVis)
[![license](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2503.16426-b31b1b.svg)](https://arxiv.org/abs/2503.16426)

<br>
<br>

<div align="center">

English | [简体中文](README_zh-CN.md)

</div>


## Introduction

This repository contains the official implementation of the paper [DynamicVis: An Efficient and General Visual Foundation Model for Remote Sensing Image Understanding](https://arxiv.org/abs/2503.16426), developed based on the [OpenMMLab](https://openmmlab.com/codebase) framework.

DynamicVis is a dynamic visual perception foundation model for remote sensing, achieving efficient low-resource parsing of ultra-large images (2048x2048 pixels processing requires only 800MB GPU RAM) through a selective region-aware architecture and multi-instance meta-embedding learning. The model demonstrates exceptional performance across nine remote sensing downstream tasks, with ~20x computational efficiency and ~97% memory reduction compared to ViT, enabling cross-task understanding of high-resolution remote sensing imagery.

The current branch has been tested on Linux systems with PyTorch 2.x and CUDA 12.1, supporting Python 3.10+ and compatible with most CUDA versions.

If you find this project helpful, please give us a star ⭐️. Your support is our greatest motivation.



<details open>
<summary>Main Features</summary>

- API interfaces and usage methods highly consistent with OpenMMLab
- Open-sourced DynamicVis models and weights of different scales as described in the paper
- Supports fine-tuning and testing for nine remote sensing downstream tasks mentioned in the paper

</details>

## Changelog

🌟 **2025.03.20** Released DynamicVis project.

🌟 **2025.03.21** Updated DynamicVis pretraining code.


## TODO

- [X] Organize DynamicVis pretraining code
- [ ] Organize fine-tuning and testing code for nine tasks in the paper
- [ ] Upload DynamicVis model weights
- [ ] Upload DynamicVis model weights based on Mamba2


## Contents

- [Introduction](#introduction)
- [Changelog](#changelog)
- [TODO](#todo)
- [Table of Contents](#Contents)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Model Pretraining](#model-pretraining)
- [Model Fine-tuning](#model-fine-tuning)
- [Testing](#testing)
- [FAQ](#faq)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)
- [Contact](#contact)

## Installation

### Prerequisites

- Linux OS (Windows not supported for Mamba)
- Python 3.10+ (3.11 recommended)
- PyTorch 2.0+ (2.4 recommended)
- CUDA 11.7+ (12.1 recommended)
- MMCV 2.0+ (2.2 recommended)
- Mamba 2.2.4

### Environment Setup

We recommend using Miniconda for installation. The following commands will create a virtual environment named `dynamicvis` and install PyTorch and MMCV. The default CUDA version in these instructions is **12.1**. Modify accordingly if using a different CUDA version.

<details>

**Step 0**: Install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install).

**Step 1**: Create and activate a virtual environment:

```shell
conda create -n dynamicvis python=3.11 -y
conda activate dynamicvis
```

**Step 2**: Install [PyTorch 2.4.x](https://pytorch.org/get-started/previous-versions/).

Linux:

```shell
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
# OR
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

**Step 3**: Install [MMCV 2.2.x](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).

```shell
pip install -U openmim
mim install mmcv==2.2.0
# OR
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
```

**Step 4**: Install [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) and [Mamba 2.2.4](https://github.com/state-spaces/mamba):

```shell
# Install causal-conv1d
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip install causal_conv1d-1.5.0.post8+cu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl

# Install mamba
wget https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip install mamba_ssm-2.2.4+cu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```

Note: If you encounter issues, try using `pip install`.

```shell
pip install causal-conv1d==1.5.0.post8
pip install mamba==2.2.4

# OR
# Compile and install first, and keep the whl file for future installations to avoid recompilation
# pip wheel --wheel-dir=../../software/mamba2-2.4.4/ causal-conv1d==1.5.0.post8 -i https://pypi.org/simple
# pip wheel --wheel-dir=../../software/mamba2-2.4.4/ mamba-ssm==2.2.4 -i https://pypi.org/simple
# pip install ../../software/mamba2-2.4.4/causal_conv1d-1.5.0.post8-cp311-cp311-linux_x86_64.whl
# pip install ../../software/mamba2-2.4.4/mamba_ssm-2.2.4-cp311-cp311-linux_x86_64.whl
```



**Step 5**: Install other dependencies:

```shell
pip install transformers==4.49.0 
pip install -U ipdb braceexpand mat4py pycocotools shapely ftfy scipy terminaltables wandb
```

</details>


### Install DynamicVis

Download or clone the DynamicVis repository。

```shell
git clone git@github.com:KyanChen/DynamicVis.git
cd DynamicVis
```

## Dataset Preparation

<details>

### Pretraining Dataset

#### fMoW Dataset

##### Download Data

- Dataset: [fMoW Dataset](https://github.com/fMoW/dataset)
- Download the **fMoW-rgb** subset:

```shell
pip install awscli

# Delete AWS configuration file
rm -rf ~/.aws
# List files
aws s3 ls --no-sign-request s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/

# Download files
aws s3 sync --no-sign-request s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/ ./data/fmow-rgb/
```

##### Organization

We use [WebDataset](https://github.com/webdataset/webdataset) to organize large-scale pretraining data. WebDataset is a data loading library for large-scale datasets that can efficiently handle large datasets.

```shell
python tools_DynamicVis/tools_data/fMoW/get_fmow_train_val_data.py
```

### Scene Classification Datasets

#### UC Merced Dataset

##### Download Data

- [UC Merced Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html)

##### Directory Structure

```
${DATASET_ROOT}  # Dataset root, e.g.: /home/username/data/UC
├── airplane
│   ├── airplane01.tif
│   ├── airplane02.tif
│   └── ...
├── ...
└── ...
```

Note: We provide dataset partitioning files in the `datainfo` folder. You can also use the [Python script](tools_DynamicVis/tools_data/UCMerced/split_trainval.py) to partition the dataset.

#### AID Dataset

##### Download Data

- [AID Dataset](https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets)

##### Directory Structure

```
${DATASET_ROOT}
├── airplane
│   ├── airplane01.jpg
│   ├── airplane02.jpg
│   └── ...
├── ...
└── ...
```

Note: We provide dataset partitioning files in the `datainfo` folder. You can also use the [Python script](tools_DynamicVis/tools_data/AID/split_trainval.py) to partition the dataset.

</details>

## Model Pretraining

### Config File Overview

We provide configuration files for DynamicVis models of different parameter sizes as described in the paper. You can find them in the [configs_DynamicVis/fMoW](configs_DynamicVis/fMoW) folder. The config files maintain consistent API interfaces and usage methods with OpenMMLab. Below are some key parameter explanations. For more information on the parameters, refer to the [OpenMMLab documentation](https://mmsegmentation.readthedocs.io/en/latest/user_guides/1_config.html).


<details>

**Parameter Explanation**:

- `work_dir`: Output path for model training, generally no need to modify.
- `data_root`: Dataset root directory, **modify to the absolute path of the dataset root**.
- `code_root`: Code root directory, **modify to the absolute path of the code root**.
- `batch_size`: Batch size per GPU, **modify according to GPU memory size**.
- `max_epochs`: Maximum number of training epochs, generally no need to modify.
- `val_interval`: Interval of validation set, generally no need to modify.
- `vis_backends/WandbVisBackend`: Configuration of network-side visualization tools, **after uncommenting, you need to register an account on the `wandb` official website to view the visualization results during training in a web browser**.
- `load_from`: Path to the model's pretraining checkpoint, generally no need to modify.
- `resume`: Whether to resume training from a checkpoint, generally no need to modify.
- `default_hooks/CheckpointHook`: Configuration of model checkpoint saving during training, generally no need to modify.
- `model/backbone`: Visual backbone of the DynamicVis model, **modify according to actual situation**.
- `model/backbone/arch`: Configuration of the main network, **modify according to actual situation**.
- `model/backbone/spatial_token_keep_ratios`: Spatial token retention ratio, **modify according to actual situation**.
- `model/pre_neck`: FPN Neck of the DynamicVis model.
- `model/neck`: Region feature extractor of the DynamicVis model, generally no need to modify.
- `model/head`: Classification head of the DynamicVis model, generally no need to modify.
- `optim_wrapper`: Configuration of the optimizer, generally no need to modify.
- `data_preprocessor/mean/std`: Mean and standard deviation of data preprocessing, generally no need to modify.

</details>


### Training Commands

```shell
# Single GPU
python tools_mmpretrain/train.py configs_DynamicVis/fMoW/name_to_config.py

# Multi-GPU
sh tools_mmpretrain/dist_train.sh configs_DynamicVis/fMoW/name_to_config.py ${GPU_NUM}
```

### Testing

```shell
# Single GPU
python tools_mmpretrain/test.py configs_DynamicVis/fMoW/name_to_config.py ${CHECKPOINT_FILE}

# Multi-GPU
sh tools_mmpretrain/dist_test.sh configs_DynamicVis/fMoW/name_to_config.py ${CHECKPOINT_FILE} ${GPU_NUM}
```

## Model Fine-tuning

### Scene Classification

#### Config Files


We provide configuration files for the UC Merced and AID datasets mentioned in the paper. You can find them in the [UC configuration file](configs_DynamicVis/UCMerced) and [AID configuration file](configs_DynamicVis/AID) folders.


The following are some key parameter explanations other than the pretraining part of the Config.

<details>

**Parameter Explanation**:

- xx

</details>

### Fine-tuning

```shell
# Single GPU
python tools_mmpretrain/train.py configs_DynamicVis/UCMerced/name_to_config.py

# Multi-GPU
sh tools_mmpretrain/dist_train.sh configs_DynamicVis/UCMerced/name_to_config.py ${GPU_NUM}
```

### Testing

```shell
# Single GPU
python tools_mmpretrain/test.py configs_DynamicVis/UCMerced/name_to_config.py ${CHECKPOINT_FILE}

# Multi-GPU
sh tools_mmpretrain/dist_test.sh configs_DynamicVis/UCMerced/name_to_config.py ${CHECKPOINT_FILE} ${GPU_NUM}
```

## FAQ

<details>


We list some common problems and their corresponding solutions here. If you find any problems missing, please feel free to submit a PR to enrich this list. If you cannot find help here, please use [issue](https://github.com/KyanChen/DynamicVis/issues) to seek help. Please fill in all the required information in the template, which will help us locate the problem more quickly.

### 1. Do I need to install the MM series packages?

We recommend that you do not install the MM series packages (such as MMDet), as we have included everything you need. If you install the MM series packages, you may encounter errors when running the code. If you encounter an error that the module has not been registered, please check:

- Whether the module is a package that needs to be installed, if so, install it
- Whether the MM series packages are installed, if so, uninstall them
- Whether `@MODELS.register_module()` is added before the class name, if not, add it
- Whether `from .xxx import xxx` is added in `__init__.py`, if not, add it
- Whether `custom_imports = dict(imports=['dynamicvis'], allow_failed_imports=False)` is added in the Config file, if not, add it

### 2. Solution to dist_train.sh: Bad substitution

If you encounter a `Bad substitution` error when running `dist_train.sh`, please use `bash dist_train.sh` to run the script.


</details>

## Acknowledgements

This project is built upon [OpenMMLab](https://openmmlab.com/codebase). We thank the OpenMMLab developers.

## Citation

If you use DynamicVis in your research, please cite:

```bibtex
@article{chen2025dynamicvis,
  title={DynamicVis: An Efficient and General Visual Foundation Model for Remote Sensing Image Understanding},
  author={Chen, Keyan and Liu, Chenyang and Chen, Bowen and Li, Wenyuan and Zou, Zhengxia and Shi, Zhenwei},
  journal={arXiv preprint arXiv:2503.16426},
  year={2025}
}
```

## License

This project is licensed under the [Apache 2.0 License](LICENSE).

## Contact

For further questions❓, feel free to contact us 👬

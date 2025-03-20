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
    <span style="font-size: 20px; ">项目主页</span>
  </a>
  &nbsp;&nbsp;&nbsp;&nbsp;
  <a href="https://arxiv.org/abs/2501.xx">
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
[![arXiv](https://img.shields.io/badge/arXiv-2501.xx-b31b1b.svg)](https://arxiv.org/abs/2501.xxx)

<br>
<br>

<div align="center">

[English](README.md) | 简体中文

</div>


## 简介

本项目仓库是论文 [DynamicVis: An Efficient and General Visual Foundation Model for Remote Sensing Image Understanding](https://arxiv.org/abs/2501.xxx) 的代码实现，基于 [OpenMMLab](https://openmmlab.com/codebase) 代码库进行开发。

遥感动态视觉感知大模型DynamicVis，通过选择性区域感知架构与多实例元嵌入学习，实现超大尺寸图像的高效低资源解析（2048x2048像素处理仅需800MB显存）。该模型在九大遥感下游任务中展现卓越性能，计算效率达ViT的~20倍且内存消耗降低~97%，为高分辨率遥感影像的跨任务理解提供支持。

当前分支在 Linux 系统，PyTorch 2.x 和 CUDA 12.1 下测试通过，支持 Python 3.10+，能兼容绝大多数的 CUDA 版本。

如果你觉得本项目对你有帮助，请给我们一个 star ⭐️，你的支持是我们最大的动力。

<details open>
<summary>主要特性</summary>

- 与 OpenMMLab 高度保持一致的 API 接口及使用方法
- 开源了论文中不同版本大小的 DynamicVis 模型和权重
- 支持论文中九种遥感下游任务的微调和测试


</details>

## 更新日志

🌟 **2025.03.20** 发布了 DynamicVis 项目。


## TODO

- [ ] 整理DynamicVis的预训练代码
- [ ] 上传DynamicVis模型权重
- [ ] 整理论文中九个任务的微调和测试代码
- [ ] 上传基于Mamba2开发的DynamicVis模型权重


## 目录

- [简介](#简介)
- [更新日志](#更新日志)
- [TODO](#TODO)
- [目录](#目录)
- [安装](#安装)
- [数据集准备](#数据准备)
- [模型训练](#模型预训练)
- [模型微调](#模型微调)
- [常见问题](#常见问题)
- [致谢](#致谢)
- [引用](#引用)
- [开源许可证](#开源许可证)
- [联系我们](#联系我们)

## 安装

### 依赖项

- Linux 系统， Windows 无法运行 Mamba
- Python 3.10+，推荐使用 3.11
- PyTorch 2.0 或更高版本，推荐使用 2.4
- CUDA 11.7 或更高版本，推荐使用 12.1
- MMCV 2.0 或更高版本，推荐使用 2.2
- Mamba 2.2.4 版本

### 环境安装

推荐使用 Miniconda 来进行安装，以下命令将会创建一个名为 `dynamicvis` 的虚拟环境，并安装 PyTorch 和 MMCV。下述安装步骤中，默认安装的 CUDA 版本为 **12.1**，如果你的 CUDA 版本不是 12.1，请根据实际情况进行修改。

注解：如果你对 PyTorch 有经验并且已经安装了它，你可以直接跳转到下一小节。否则，你可以按照下述步骤进行准备。

<details>

**步骤 0**：安装 [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)。

**步骤 1**：创建一个名为 `dynamicvis` 的虚拟环境，并激活它。

```shell
conda create -n dynamicvis python=3.11 -y
conda activate dynamicvis
```

**步骤 2**：安装 [PyTorch2.4.x](https://pytorch.org/get-started/previous-versions/)。

Linux/Windows:

```shell
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121
```
或者
```shell
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

**步骤 3**：安装 [MMCV2.2.x](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)。

```shell
pip install -U openmim
mim install mmcv==2.2.0
#或者
pip install mmcv==2.2.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
```

**步骤 4**：安装 [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d) 和 [Mamba2.2.4](https://github.com/state-spaces/mamba)

下载对应版本安装可以避免编译或编译出错，参考 [causal-conv1d](https://github.com/Dao-AILab/causal-conv1d/releases) 和 [Mamba2.2.4](https://github.com/state-spaces/mamba/releases)

```shell
# 下载安装causal-conv1d
wget https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.5.0.post8/causal_conv1d-1.5.0.post8+cu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip install causal_conv1d-1.5.0.post8+cu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl


# 下载安装mamba
wget https://github.com/state-spaces/mamba/releases/download/v2.2.4/mamba_ssm-2.2.4+cu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
pip install mamba_ssm-2.2.4+cu12torch2.4cxx11abiTRUE-cp311-cp311-linux_x86_64.whl
```


**步骤 4**：安装其他依赖项。

```shell
pip install 
pip install ipdb -U
```


</details>


### 安装 DynamicVis


下载或克隆 DynamicVis 仓库即可。

```shell
git clone git@github.com:KyanChen/DynamicVis.git
cd DynamicVis
```

## 数据准备

<details>

### 预训练数据集

#### fMoW 数据集

##### 下载数据集

- 图片地址：[fMoW 数据集](https://github.com/fMoW/dataset)。
- 下载**fMoW-rgb**子集即可

```shell
pip install awscli

# 显示文件夹
aws s3 ls --no-sign-request s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/

# 下载文件夹
aws s3 sync --no-sign-request s3://spacenet-dataset/Hosted-Datasets/fmow/fmow-rgb/ ./data/fmow-rgb/
```

##### 组织方式

我们采用[WebDataset](https://github.com/webdataset/webdataset)来组织大规模预训练，WebDataset是一个用于大规模数据集的数据加载库，它可以有效地处理大规模数据集。


```shell
# 将数据组织成WebDataset需要的Tar包格式
python tools_DynamicVis/tools_data/fMoW/get_fmow_train_val_data.py
```

### 场景分类数据集

#### UC Merced 数据集

##### 下载数据集

- 下载地址：[UC Merced 数据集](http://weegee.vision.ucmerced.edu/datasets/landuse.html)。

#### 组织方式

```
${DATASET_ROOT} # 数据集根目录，例如：/home/username/data/UC
├── airplane
│   ├── airplane01.tif
│   ├── airplane02.tif
│   └── ...
├── ...
├── ...
├── ...
└── ...
```
注解：在项目文件夹 `datainfo` 中，我们提供了数据集的划分文件。您也可以使用 [Python 脚本](tools_DynamicVis/tools_data/UCMerced/split_trainval.py) 来划分数据集。


#### AID 数据集

##### 下载数据集

- 下载地址：[AID 数据集](https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets)。

#### 组织方式

```
${DATASET_ROOT} # 数据集根目录，例如：/home/username/data/AID
├── airplane
│   ├── airplane01.jpg
│   ├── airplane02.jpg
│   └── ...
├── ...
├── ...
├── ...
└── ...
```
注解：在项目文件夹 `datainfo` 中，我们提供了数据集的划分文件。您也可以使用 [Python 脚本](tools_DynamicVis/tools_data/AID/split_trainval.py) 来划分数据集。

</details>

## 模型预训练

### Config 文件及主要参数解析

我们提供了论文中不同参数大小的 DynamicVis 模型的配置文件，你可以在 [配置文件](configs_DynamicVis/fMoW) 文件夹中找到它们。Config 文件完全与 OpenMMLab 保持一致的 API 接口及使用方法。下面我们提供了一些主要参数的解析。如果你想了解更多参数的含义，可以参考 [OpenMMLab 相关文档](https://mmsegmentation.readthedocs.io/zh-cn/latest/user_guides/1_config.html)。

<details>

**参数解析**：

- `work_dir`：模型训练的输出路径，一般不需要修改。
- `data_root`：数据集根目录，**修改为数据集根目录的绝对路径**。
- `batch_size`：单卡的 batch size，**需要根据显存大小进行修改**。
- `max_epochs`：最大训练轮数，一般不需要修改。
- `val_interval`：验证集的间隔轮数，一般不需要修改。
- `vis_backends/WandbVisBackend`：网络端可视化工具的配置，**打开注释后，需要在 `wandb` 官网上注册账号，可以在网络浏览器中查看训练过程中的可视化结果**。
- `resume`: 是否断点续训，一般不需要修改。
- `load_from`：模型的预训练的检查点路径，一般不需要修改。
- `init_from`：模型的预训练的检查点路径，一般保持为None，除非需要断点续训，则需要修改为对应的检查点路径。
- `default_hooks/CheckpointHook`：模型训练过程中的检查点保存配置，一般不需要修改。
- `model/backbone`：DynamicVis模型的视觉骨干，**需要根据实际情况进行修改**。
- `AMP training config`：混合精度训练的配置，一般不需要修改。
- `dataset_type`：数据集的类型，一般不需要修改。
- `data_preprocessor/mean/std`：数据预处理的均值和标准差，一般不需要修改。

</details>


### 训练

```shell
# 单卡训练
python tools_mmpretrain/train.py configs_DynamicVis/fMoW/name_to_config.py  # name_to_config.py 为你想要使用的配置文件
# 多卡训练
sh tools_mmpretrain/dist_train.sh configs_DynamicVis/fMoW/name_to_config.py ${GPU_NUM}  # name_to_config.py 为你想要使用的配置文件，GPU_NUM 为使用的 GPU 数量

```


### 测试

```shell
# 单卡测试
python tools_mmpretrain/test.py configs_DynamicVis/fMoW/name_to_config.py ${CHECKPOINT_FILE}  # name_to_config.py 为你想要使用的配置文件，CHECKPOINT_FILE 为你想要使用的检查点文件
# 多卡测试
sh tools_mmpretrain/dist_test.sh configs_DynamicVis/fMoW/name_to_config.py ${CHECKPOINT_FILE} ${GPU_NUM}  # name_to_config.py 为你想要使用的配置文件，CHECKPOINT_FILE 为你想要使用的检查点文件，GPU_NUM 为使用的 GPU 数量

```



## 模型微调

### 场景分类

#### Config 文件及主要参数解析

我们提供了论文中的 UC Merced 和 AID 数据集的配置文件，你可以在 [UC 配置文件](configs_DynamicVis/UCMerced) 和 [AID 配置文件](configs_DynamicVis/AID) 文件夹中找到它们。

下面是一些除了预训练部分Config参数外的主要参数解析。

<details>

XXX

</details>

#### 微调

```shell
# 单卡微调
sh tools_mmpretrain/train.py configs_DynamicVis/UCMerced/name_to_config.py  # name_to_config.py 为你想要使用的配置文件
# 多卡微调
sh tools_mmpretrain/dist_train.sh configs_DynamicVis/UCMerced/name_to_config.py ${GPU_NUM}  # name_to_config.py 为你想要使用的配置文件，GPU_NUM 为使用的 GPU 数量

```

#### 测试

```shell
# 单卡测试
sh tools_mmpretrain/test.py configs_DynamicVis/UCMerced/name_to_config.py ${CHECKPOINT_FILE}  # name_to_config.py 为你想要使用的配置文件，CHECKPOINT_FILE 为你想要使用的检查点文件
# 多卡测试
sh tools_mmpretrain/dist_test.sh configs_DynamicVis/UCMerced/name_to_config.py ${CHECKPOINT_FILE} ${GPU_NUM}  # name_to_config.py 为你想要使用的配置文件，CHECKPOINT_FILE 为你想要使用的检查点文件，GPU_NUM 为使用的 GPU 数量

```

## 小目标检测




## 常见问题

<details>

我们在这里列出了使用时的一些常见问题及其相应的解决方案。如果您发现有一些问题被遗漏，请随时提 PR 丰富这个列表。如果您无法在此获得帮助，请使用[issue](https://github.com/KyanChen/DynamicVis/issues)来寻求帮助。请在模板中填写所有必填信息，这有助于我们更快定位问题。

### 1. 是否需要安装MM系列包？

我们建议您不要安装MM系列包（例如MMDet），因为我们已经包含了所有需要的内容，如果您安装了MM系列包，可能会导致代码运行出错。如果你出现了模块尚未被注册的错误，请检查：

- 该模块是否是一个需要安装的包，若是则安装
- 是否安装了MM系列包，若有则卸载
- 是否在类名前加上了`@MODELS.register_module()`，若没有则加上
- 是否在`__init__.py`中加入了`from .xxx import xxx`，若没有则加上
- 是否在Config文件中加入了`custom_imports = dict(imports=['dynamicvis'], allow_failed_imports=False)`，若没有则加上


### 2. dist_train.sh: Bad substitution的解决

如果您在运行`dist_train.sh`时出现了`Bad substitution`的错误，请使用`bash dist_train.sh`来运行脚本。


</details>

## 致谢

本项目基于 [OpenMMLab](https://openmmlab.com/codebase) 进行开发，感谢 OpenMMLab 项目的开发者们。

## 引用

如果你在研究中使用了本项目的代码或者性能基准，请参考如下 bibtex 引用 DynamicVis。

```
@article{chen2025dynamicvis,
  title={DynamicVis: An Efficient and General Visual Foundation Model for Remote Sensing Image Understanding},
  author={Chen, Keyan and and Liu, Chenyang and Chen, Bowen and Li, Wenyuan and Zou, Zhengxia and Shi, Zhenwei},
  journal={arXiv preprint arXiv:2501.xxxx},
  year={2025}
}
```

## 开源许可证

该项目采用 [Apache 2.0 开源许可证](LICENSE)。

## 联系我们

如果有其他问题❓，请及时与我们联系 👬
# LeRobot 环境搭建指南 — NVIDIA DGX Spark

## 硬件环境

| 项目 | 规格 |
|------|------|
| 平台 | NVIDIA DGX Spark |
| CPU | Grace (ARM aarch64) |
| GPU | NVIDIA GB10 (Blackwell), Compute Capability 12.1 |
| Driver | 580.173.02 |
| OS | Ubuntu (aarch64) |

## 关键依赖版本

| 包 | 版本 | 说明 |
|----|------|------|
| Python | 3.13 | conda 环境 |
| torch | 2.11.0+cu130 | PyPI wheel 自带 CUDA 13.0.2 |
| torchvision | 0.26.0+cu130 | 依赖 torch==2.11.0 |
| torchcodec | 0.11.1+cu130 | 视频解码，需 FFmpeg 共享库 |
| huggingface_hub | 1.31.0 | lerobot 要求 >=1.6.0 |
| datasets | 4.8.5 | lerobot[dataset] extra |
| FFmpeg | 8.0.1 | conda-forge 安装 |
| lerobot | 0.6.2 | 从源码 editable 安装 |

## 安装步骤

### 1. 创建 conda 环境

```bash
conda create -n lerobot python=3.13 -y
conda activate lerobot
```

### 2. 安装 PyTorch (CUDA 13.0)

> **重要**: torch 2.11.0 在 PyPI 上的 aarch64 wheel 自带 CUDA 13.0.2，完美适配 Blackwell GPU。
> 无需使用 `--index-url` 指定 CUDA 源。

```bash
pip install torch==2.11.0 torchvision==0.26.0
```

### 3. 从源码安装 LeRobot

```bash
cd /path/to/lerobot
pip install -e ".[test,dev,dataset]"
```

> 如果只需要基础功能，可以省略 extras：`pip install -e .`
> 可用的 extras: `test`, `dev`, `dataset`, `act`, `diffusion`, `pi`, `smolvla` 等。
> `dataset` extra 会安装 `datasets`, `pandas`, `pyarrow` 等数据处理依赖，**推荐安装**。

### 4. 安装 torchcodec

> **注意**: lerobot 的 pyproject.toml 中 torchcodec 的平台标记可能未覆盖 aarch64 + cu130 组合，
> 需要手动安装。

```bash
pip install "torchcodec>=0.11.0,<0.12.0"
```

### 5. 安装 FFmpeg

> torchcodec 运行时需要 FFmpeg 共享库 (libavutil, libavformat 等)。
> 支持 FFmpeg 4/5/6/7/8。

```bash
conda install -y -c conda-forge ffmpeg
```

### 6. 验证安装

```bash
python -c "
import torch
print(f'torch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')

x = torch.randn(1000, 1000, device='cuda')
y = torch.randn(1000, 1000, device='cuda')
z = torch.mm(x, y)
print(f'CUDA matmul: OK')

import torchvision
print(f'torchvision: {torchvision.__version__}')

import torchcodec
print(f'torchcodec: {torchcodec.__version__}')

import huggingface_hub
print(f'huggingface_hub: {huggingface_hub.__version__}')

import datasets
print(f'datasets: {datasets.__version__}')

import lerobot
print(f'lerobot: {lerobot.__version__}')

from lerobot.policies.act.modeling_act import ACTPolicy
print('ACTPolicy: OK')
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
print('DiffusionPolicy: OK')
from lerobot.datasets.lerobot_dataset import LeRobotDataset
print('LeRobotDataset: OK')
from lerobot.configs.train import TrainPipelineConfig
print('TrainPipelineConfig: OK')
print('=== All checks passed! ===')
"
```

预期输出：

```
torch: 2.11.0+cu130
CUDA available: True
CUDA version: 13.0
GPU: NVIDIA GB10
GPU memory: 121.7 GB
CUDA matmul: OK
torchvision: 0.26.0+cu130
torchcodec: 0.11.1+cu130
huggingface_hub: 1.31.0
datasets: 4.8.5
lerobot: 0.6.2
ACTPolicy: OK
DiffusionPolicy: OK
LeRobotDataset: OK
TrainPipelineConfig: OK
=== All checks passed! ===
```

## 常见问题

### Q1: 为什么不用 `pip install lerobot`？

直接 `pip install lerobot` 会从 PyPI 安装预编译包，存在以下问题：

1. **PyTorch CUDA 版本不匹配**: 默认拉取 cu128 的 torch，Blackwell GPU 推荐 cu130
2. **torchcodec 平台标记**: aarch64 上可能未自动安装
3. **源码安装更灵活**: 方便修改代码、调试、贡献

### Q2: torch 2.9.0+cu130 能用吗？

可以，但 torchcodec 0.11.x 要求 torch>=2.11。如果用 torch 2.9.0：
- 需要跳过 torchcodec 或降级到更早版本
- 视频解码会 fallback 到 PyAV（性能较差）

### Q3: 不要混用 spark_envs.txt 中的包

`~/works/spark_envs.txt` 中的部分包版本与 lerobot 不兼容：

| 包 | spark_envs.txt | lerobot 要求 | 冲突 |
|----|---------------|-------------|------|
| huggingface_hub | 0.35.0 | >=1.6.0 | ❌ 差一个大版本 |
| transformers | 4.56.1 | >=5.4.0 (可选) | ⚠️ 不装 VLA 策略无影响 |
| accelerate | 1.10.1 | >=1.14.0 (可选) | ⚠️ 同上 |

**建议**: lerobot 环境独立管理，不要与 spark_envs.txt 混装。

### Q4: torchcodec 报 FFmpeg 找不到？

```
RuntimeError: Could not load libtorchcodec.
OSError: libavutil.so.60: cannot open shared object file
```

解决：安装 FFmpeg 共享库

```bash
conda install -y -c conda-forge ffmpeg
```

### Q5: 如何安装 VLA 策略 (Pi0, SmolVLA 等)？

```bash
pip install -e ".[pi]"           # Pi0
pip install -e ".[smolvla]"      # SmolVLA
pip install -e ".[groot]"        # GR00T N1.7
```

这些 extras 会额外安装 `transformers>=5.4.0` 和 `accelerate>=1.14.0`。

## 快速开始

```bash
conda activate lerobot

# 训练 ACT 策略
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=lerobot/aloha_mobile_cabinet

# 评估策略
lerobot-eval \
  --policy.path=lerobot/act_aloha_mobile_cabinet \
  --env.type=aloha
```

## 已安装的核心包清单

```
torch                    2.11.0
torchvision              0.26.0
torchcodec               0.11.1
cuda-toolkit             13.0.2
nvidia-cublas            13.1.0.3
nvidia-cudnn-cu13        9.19.0.56
nvidia-nccl-cu13         2.28.9
huggingface_hub          1.31.0
datasets                 4.8.5
pandas                   2.3.3
pyarrow                  25.0.1
lerobot                  0.6.2
draccus                  0.11.6
einops                   0.8.2
gymnasium                1.3.0
numpy                    2.2.6
opencv-python-headless   4.13.0.92
safetensors              0.8.0
pillow                   12.3.0
```

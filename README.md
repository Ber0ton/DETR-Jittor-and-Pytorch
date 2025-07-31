# DETR - Jittor 实现

本仓库提供了DETR的 Jittor 实现版本，实现了与官方 PyTorch 版本的对齐。

## 环境配置

### 系统配置

- Python >= 3.7
- Jittor >= 1.3.8.5
- CUDA >= 12.1

### 安装步骤

**克隆仓库**

```bash
git clone https://github.com/yourusername/model-jittor.git
cd model-jittor
```

#### Jittor

```bash
# 1. 创建Jittor虚拟环境
conda create -n jt_detr python=3.7
conda activate jt_detr

# 2. 安装 Jittor
sudo apt install python3.7-dev libomp-dev
python3.7 -m pip install jittor
python3.7 -m jittor.test.test_example
# 如果您电脑包含Nvidia显卡，检查cudnn加速库
python3.7 -m jittor.test.test_cudnn_op

# 3. 安装其他依赖
cd DETR-Jittor
pip install -r requirements.txt
```

**Pytorch**

```bash
# 1. 创建Pytorch虚拟环境
conda create -n pt_detr python=3.7
conda activate pt_detr

# 2. 安装 PyTorch
pip install torch==2.4.1 torchvision==0.20.0

# 3. 安装其他依赖
cd DETR-Pytorch
pip install -r requirements.txt
```

## 数据准备

请按照如下目录准备您的数据集目录，具体标注格式json等参照COCO格式

```bash
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

## 训练脚本

**Jittor**

```bash
python main.py \
    --coco_path /path/to/coco \
    --output_dir output/detr_jittor \
    --batch_size 4 \
    --lr 0.25e-4 \
    --lr_backbone 0.25e-5 \
    --epochs 300 \
    --lr_drop 200
```

**Pytorch**

```bash
python main.py \
    --coco_path  /autodl-fs/data/data \
    --output_dir output/detr_pytorch \
    --batch_size 4 \
    --lr 0.25e-4 \
    --lr_backbone 0.25e-5 \
    --epochs 300
```



## 测试脚本

**Jittor**

```bash
python main.py \
    --batch_size 2 \
    --no_aux_loss \
    --eval \
    --resume output/path/to/checkpoint.pkl \
    --coco_path /path/to/coco
```

**Pytorch**

```
python main.py \
    --batch_size 2 \
    --no_aux_loss \
    --eval \
    --resume output/path/to/checkpoint.pth \
    --coco_path /path/to/coco
```

## 训练日志

### 训练配置

此处仅展示优化器超参数，完整参数见`main.py`

#### 优化器超参数 - Jittor

| 参数              | 默认值  | 类型  | 说明           |
| ----------------- | ------- | ----- | -------------- |
| `--lr`            | 0.25e-4 | float | 主网络学习率   |
| `--lr_backbone`   | 0.25e-5 | float | 骨干网络学习率 |
| `--batch_size`    | 4       | int   | 训练批次大小   |
| `--weight_decay`  | 1e-4    | float | 权重衰减系数   |
| `--epochs`        | 120     | int   | 训练轮数       |
| `--clip_max_norm` | 0.1     | float | 梯度裁剪阈值   |

#### 优化器超参数 - Pytorch

| 参数              | 默认值 | 类型  | 说明           |
| ----------------- | ------ | ----- | -------------- |
| `--lr1            | 1e-4   | float | 主网络学习率   |
| `--lr_backbone`   | 1e-5   | float | 骨干网络学习率 |
| `--batch_size`    | 8      | int   | 训练批次大小   |
| `--weight_decay`  | 1e-4   | float | 权重衰减系数   |
| `--epochs`        | 31     | int   | 训练轮数       |
| `--clip_max_norm` | 0.1    | float | 梯度裁剪阈值   |

### 损失曲线



### AP性能曲线

平滑后对称



### 训练过程日志示例

**Pytorch**

见''

**Jittor**

见''



## 性能对比

### 训练进度

| 框架    | 训练轮数   |
| ------- | ---------- |
| PyTorch | 31 epochs  |
| Jittor  | 120 epochs |

### 最终性能对比（最后一个 epoch）

| 指标 | PyTorch | Jittor | 差异    |
| ---- | ------- | ------ | ------- |
| AP   | 0.0002  | 0.0002 | ±0.0000 |
| AP50 | 0.0010  | 0.0012 | +0.0002 |
| AP75 | 0.0001  | 0.0000 | -0.0001 |

### 最佳性能对比

| 指标 | PyTorch 最佳值 | Jittor 最佳值 | 性能差异 |
| ---- | -------------- | ------------- | -------- |
| AP   | 0.0002         | 0.0007        | +250%    |
| AP50 | 0.0010         | 0.0019        | +90%     |
| AP75 | 0.0001         | 0.0002        | +100%    |


# DETR - Jittor Implementation

[简体中文](./README_CN.md) | English

## 📋 Project Overview

[![Jittor](https://img.shields.io/badge/Jittor-v1.3.8.5+-green.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMiA3VjE3TDEyIDIyTDIyIDE3VjdMMTIgMloiIGZpbGw9IiM0Q0FGNTAiLz4KPC9zdmc+)](https://github.com/Jittor/jittor) [![PyTorch](https://img.shields.io/badge/PyTorch-v2.4.1-orange.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/)

This project is a **Jittor** framework implementation of **DETR (Detection Transformer)**, aimed at verifying the consistency and performance comparison between Jittor and PyTorch in deep learning model training.

- Considering limited computational resources, time constraints, and the tendency of small datasets to overfit, current alignment experiments are based on a small-scale dataset containing 1,000 images. According to the official PyTorch implementation, achieving the standard performance reported in the paper (AP ~42.0) on the complete COCO dataset requires training for 300 epochs on 8 V100 GPUs. See https://github.com/facebookresearch/detr:

  > 'A single epoch takes 28 minutes, so 300 epoch training takes around 6 days on a single machine with 8 V100 cards. To ease reproduction of our results we provide results and training logs for 150 epoch schedule (3 days on a single machine), achieving 39.5/60.3 AP/AP50.'

- While Transformer's global modeling capability effectively reduces detection box redundancy, its high computational complexity also leads to longer convergence times. Nevertheless, this project provides a complete DETR model implementation using the Jittor framework based on the official PyTorch version, with detailed training logs and performance comparison data, along with complete training and testing scripts and configuration instructions.

## 🛠️ Environment Setup

### 📦 System Requirements

- Python >= 3.7
- Jittor >= 1.3.8.5
- CUDA >= 12.1

### 📥 Installation Steps

#### 🔧 Clone Repository

The following command will clone the entire project repository to your local machine:

```bash
git clone https://github.com/Ber0ton/DETR-Jittor-and-Pytorch.git
cd DETR-Jittor-and-Pytorch
```

#### 🚀 Jittor Environment Setup

These steps will help you configure a complete Jittor runtime environment, including creating a virtual environment, installing dependencies, and verifying the installation:

```bash
# 1. Create Jittor virtual environment
conda create -n jt_detr python=3.7
conda activate jt_detr

# 2. Install system dependencies
sudo apt install python3.7-dev libomp-dev

# 3. Install Jittor
python3.7 -m pip install jittor

# 4. Verify installation
python3.7 -m jittor.test.test_example

# 5. If your computer has an Nvidia GPU, check cudnn acceleration library
python3.7 -m jittor.test.test_cudnn_op

# 6. Install project dependencies
cd DETR-Jittor
pip install -r requirements.txt
```

#### 🔥 PyTorch Environment Setup (for comparison experiments)

To perform framework performance comparisons, you'll also need to configure a PyTorch environment:

```bash
# 1. Create PyTorch virtual environment
conda create -n pt_detr python=3.7
conda activate pt_detr

# 2. Install PyTorch
pip install torch==2.4.1 torchvision==0.20.0

# 3. Install project dependencies
cd DETR-Pytorch
pip install -r requirements.txt
```

## 📊 Data Preparation

### 📁 COCO Dataset Structure

Please prepare your dataset according to the following directory structure. Annotations must follow COCO format standards:

```
path/to/coco/
├── annotations/      # Annotation JSON files
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/        # Training images
└── val2017/          # Validation images
```

### 💾 Dataset Download

#### 🌐 Official Complete COCO Dataset

If you need to use the complete COCO dataset for training, you can use the following script to download:

```bash
# Download COCO 2017 dataset (optional)
mkdir -p data/coco
cd data/coco

# Download images
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip train2017.zip
unzip val2017.zip

# Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

#### 📦 Custom Dataset

Considering computational resource limitations and experimental efficiency, this project uses a streamlined COCO dataset for validation experiments:

**Dataset Scale**:

- Full version: 1,000 training images
- Minimal version: 60 training images (for quick validation)

**Dataset Access**:

- Baidu Netdisk: https://pan.baidu.com/s/1TeHnVfY88K5BGvCQ33lGgw  (Password: m74q)
- Contains pre-processed COCO format annotation files

## 🚀 Training Scripts

### 🟢 Jittor Training

The following command starts DETR model training under the Jittor framework. Note: adjust the batch_size parameter according to your GPU memory:

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

### 🟠 PyTorch Training (for comparison)

Train with the same hyperparameters under the PyTorch framework for performance comparison:

```bash
python main.py \
    --coco_path /path/to/coco \
    --output_dir output/detr_pytorch \
    --batch_size 4 \
    --lr 0.25e-4 \
    --lr_backbone 0.25e-5 \
    --epochs 300 \
    --lr_drop 200
```

## 🧪 Testing Scripts

### 🟢 Jittor Evaluation

Load the trained Jittor model for performance evaluation:

```bash
python main.py \
    --batch_size 2 \
    --no_aux_loss \
    --eval \
    --resume output/path/to/checkpoint.pkl \
    --coco_path /path/to/coco
```

### 🟠 PyTorch Evaluation

Load the trained PyTorch model for performance evaluation:

```bash
python main.py \
    --batch_size 2 \
    --no_aux_loss \
    --eval \
    --resume output/path/to/checkpoint.pth \
    --coco_path /path/to/coco
```

## 📝 Experimental Notes and Model Weights

Due to DETR model's high computational complexity and experimental resource limitations (single GPU training, small batch size, limited training epochs), current AP performance metrics have not yet reached the levels reported in the paper and should not be used as a performance benchmark. However, to verify the correct alignment between Jittor implementation and PyTorch version, corresponding model weight files are provided at https://pan.baidu.com/s/1bPnYl0jqxvm3Y5oK_VUjow?pwd=qbdf (Password: qbdf) for:

- Verifying model structure consistency
- Checking numerical computation alignment
- Serving as initialization weights for subsequent complete training

## ⚙️ Training Configuration and Logs

### 🔧 Main Training Parameters

Only optimizer hyperparameters are shown here. For complete parameter configuration, see `main.py`.

#### Jittor Training Configuration

| Parameter         | Default | Type  | Description              |
| ----------------- | ------- | ----- | ------------------------ |
| `--lr`            | 0.25e-4 | float | Main network learning rate |
| `--lr_backbone`   | 0.25e-5 | float | Backbone learning rate   |
| `--batch_size`    | 4       | int   | Training batch size      |
| `--weight_decay`  | 1e-4    | float | Weight decay coefficient |
| `--epochs`        | 120     | int   | Training epochs          |
| `--clip_max_norm` | 0.1     | float | Gradient clipping threshold |

#### PyTorch Training Configuration

| Parameter         | Default | Type  | Description              |
| ----------------- | ------- | ----- | ------------------------ |
| `--lr`            | 1e-4    | float | Main network learning rate |
| `--lr_backbone`   | 1e-5    | float | Backbone learning rate   |
| `--batch_size`    | 8       | int   | Training batch size      |
| `--weight_decay`  | 1e-4    | float | Weight decay coefficient |
| `--epochs`        | 31      | int   | Training epochs          |
| `--clip_max_norm` | 0.1     | float | Gradient clipping threshold |

### 📈 Training Curves

#### Loss Curve Comparison

The figure below shows the loss change comparison between Jittor and PyTorch frameworks during training. You can see that the loss convergence trends of both frameworks are basically consistent:

![Training Loss Comparison](./pics-and-logs/training_loss_comparison.png)

#### AP Performance Curve Comparison

The figure below shows the AP performance changes on the validation set for both frameworks. Due to the small dataset scale, performance metrics are only for framework alignment verification:

![Performance Comparison](./pics-and-logs/performance_comparison.png)

### 📄 Training Logs

For complete training logs and performance logs, please see:

- **PyTorch Training Log**: `pics-and-logs/log_torch.txt`, **Performance Log**: `pics-and-logs/eval_summary_torch.txt`
- **Jittor Log**: `pics-and-logs/log_jittor.txt`, **Performance Log**: `pics-and-logs/eval_summary_jittor.txt`

## 📊 Performance Comparison

### ⏱️ Training Progress

| Framework | Training Epochs |
| --------- | --------------- |
| PyTorch   | 31 epochs       |
| Jittor    | 120 epochs      |

### 🎯 Final Performance Comparison (last epoch)

| Metric | PyTorch | Jittor | Difference |
| ------ | ------- | ------ | ---------- |
| AP     | 0.0002  | 0.0002 | ±0.0000    |
| AP50   | 0.0010  | 0.0012 | +0.0002    |
| AP75   | 0.0001  | 0.0000 | -0.0001    |

### 🏆 Best Performance Comparison

| Metric | PyTorch Best | Jittor Best |
| ------ | ------------ | ----------- |
| AP     | 0.0002       | 0.0007      |
| AP50   | 0.0010       | 0.0019      |
| AP75   | 0.0001       | 0.0002      |

## ❓ Common Issues

### 💻 Jittor Installation Issues

#### 1. Missing libstdc++.so.6 Dynamic Library

**Error Message**:

```
ImportError: /root/miniconda3/envs/jt/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found 
(required by /root/.cache/jittor/jt1.3.9/g++11.4.0/py3.7.16/Linux-5.15.0-8xcd/IntelXeonProcex70/be50/default/cu12.1.105_sm_80/jittor_core.cpython-37m-x86_64-linux-gnu.so)
```

**Solution**:

Create a symbolic link to link the system's libstdc++.so.6 to the conda environment:

```bash
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /root/miniconda3/envs/jt/lib/libstdc++.so.6
```

#### 2. cutlass.zip Download Failure

**Error Message**:

```
MD5 mismatch between the server and the downloaded file /root/.cache/jittor/cutlass/cutlass.zip
```

**Solution**:

- Method 1: Manually download cutlass.zip

  ```bash
  wget https://cg.cs.tsinghua.edu.cn/jittor/assets/cutlass.zip -O ~/.cache/jittor/cutlass/cutlass.zip
  ```

- Method 2: Modify the download link in Jittor source code to the above address

### 🔄 PyTorch to Jittor Conversion Notes

#### 1. Advanced Indexing Behavior Differences

**Problem Description**:
PyTorch and Jittor have fundamental differences in handling advanced indexing:

- **PyTorch**: Performs pairing (zip) operation on multiple index arrays
- **Jittor/NumPy**: Performs Cartesian product operation on index arrays

**Specific Behavior**:

The following code demonstrates the different behaviors in advanced indexing between the two frameworks:

```python
# In PyTorch
src_logits[idx]  # idx = (batch_idx, src_idx)
# Takes elements pairwise: src_logits[batch_idx[0], src_idx[0]], src_logits[batch_idx[1], src_idx[1]], ...

# In Jittor
src_logits[idx]  
# Generates Cartesian product: all combinations of batch_idx with all src_idx
```

**Impact**:

- `target_classes[idx] = target_classes_o` cannot assign correctly, causing matched queries to remain as no-object class
- `src_boxes = outputs["pred_boxes"][idx]` extracts a `(Q×N_match)` cross matrix
- Loss is incorrectly distributed, resulting in `loss_bbox_unscaled ≈ 0.03`, training appears normal but actually: classification loss has almost no gradient, `class_error ≈ 10`, mAP remains 0

**Solution**:

Solve this issue by converting 2D indices to 1D linear indices:

```python
def _make_linear_idx(self, batch_idx, src_idx, num_queries):
    """Convert 2D indices (batch_idx, query_idx) to 1D linear indices"""
    return batch_idx * num_queries + src_idx

# Usage example
linear_idx = self._make_linear_idx(batch_idx, src_idx, num_queries)
src_logits_flat = src_logits.view(-1, num_classes)
selected_logits = src_logits_flat[linear_idx]
```

#### 2. argmax Return Value Differences

**Problem Description**:
Jittor's `argmax` function returns a tuple `(indices, values)`, while PyTorch only returns indices.

**Jittor argmax Behavior**:

The following example shows the return value structure of Jittor's argmax:

```python
>>> x = jt.randn(3, 2)
jt.Var([[-0.1429974  -1.1169171 ]
        [-0.35682714 -1.5031573 ]
        [ 0.66668254  1.1606413 ]], dtype=float32)
>>> jt.argmax(x, 0)
(jt.Var([2 2], dtype=int32),          # indices
 jt.Var([0.66668254 1.1606413], dtype=float32))  # corresponding max values
```

**Solution**:

To maintain compatibility with PyTorch, only take the indices part:

```python
# PyTorch code
indices = torch.argmax(x, dim=0)

# Jittor equivalent
indices, _ = jt.argmax(x, dim=0)  # Ignore returned max values
# Or
indices = jt.argmax(x, dim=0)[0]  # Take only the indices part
```

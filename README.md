# DETR - Jittor 实现

## 项目简介

[![Jittor](https://img.shields.io/badge/Jittor-v1.3.8.5+-green.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjQiIGhlaWdodD0iMjQiIHZpZXdCb3g9IjAgMCAyNCAyNCIgZmlsbD0ibm9uZSIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIj4KPHBhdGggZD0iTTEyIDJMMiA3VjE3TDEyIDIyTDIyIDE3VjdMMTIgMloiIGZpbGw9IiM0Q0FGNTAiLz4KPC9zdmc+)](https://github.com/Jittor/jittor) [![PyTorch](https://img.shields.io/badge/PyTorch-v2.4.1-orange.svg?style=for-the-badge&logo=pytorch)](https://pytorch.org/)

本项目是 **DETR (Detection Transformer)** 的 **Jittor** 框架实现，旨在验证 Jittor 与 PyTorch 在深度学习模型训练中的一致性和性能对比。

- **完整复现**：基于官方 PyTorch 版本，使用 Jittor 框架完整实现 DETR 模型
- **性能验证**：提供了详细的训练日志和性能对比数据
- **易于使用**：提供完整的训练、测试脚本和配置说明

## 环境配置

### 系统要求

- Python >= 3.7
- Jittor >= 1.3.8.5
- CUDA >= 12.1

### 安装步骤

#### 克隆仓库

```bash
git clone https://github.com/yourusername/DETR-Jittor.git
cd DETR-Jittor
```

#### Jittor 环境配置

```bash
# 1. 创建 Jittor 虚拟环境
conda create -n jt_detr python=3.7
conda activate jt_detr

# 2. 安装系统依赖
sudo apt install python3.7-dev libomp-dev

# 3. 安装 Jittor
python3.7 -m pip install jittor

# 4. 验证安装
python3.7 -m jittor.test.test_example

# 5. 如果您的电脑包含 Nvidia 显卡，检查 cudnn 加速库
python3.7 -m jittor.test.test_cudnn_op

# 6. 安装项目依赖
pip install -r requirements.txt
```

#### PyTorch 环境配置（对比实验用）

```bash
# 1. 创建 PyTorch 虚拟环境
conda create -n pt_detr python=3.7
conda activate pt_detr

# 2. 安装 PyTorch
pip install torch==2.4.1 torchvision==0.20.0

# 3. 安装项目依赖
pip install -r requirements.txt
```

## 数据准备

### COCO 数据集结构

请按照以下目录结构准备您的数据集，标注格式需符合 COCO 标准：

```
path/to/coco/
├── annotations/      # 标注 JSON 文件
│   ├── instances_train2017.json
│   └── instances_val2017.json
├── train2017/        # 训练图片
└── val2017/          # 验证图片
```

### 数据集下载

#### 官方完整COCO数据集

```bash
# 下载 COCO 2017 数据集（可选）
mkdir -p data/coco
cd data/coco

# 下载图片
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
unzip train2017.zip
unzip val2017.zip

# 下载标注
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip
```

#### 自己的数据集

考虑到计算资源限制和实验效率，本项目采用了精简版 COCO 数据集进行验证实验：

**数据集规模**：

- 完整版本：1,000 张训练图像
- 极简版本：60 张训练图像（用于快速验证）

**数据集获取**：

- 百度网盘：https://pan.baidu.com/s/1TeHnVfY88K5BGvCQ33lGgw（提取码：m74q）
- 包含已处理好的 COCO 格式标注文件

## 训练脚本

### Jittor 训练

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

### PyTorch 训练（对比参考）

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

## 测试脚本

### Jittor 评估

```bash
python main.py \
    --batch_size 2 \
    --no_aux_loss \
    --eval \
    --resume output/path/to/checkpoint.pkl \
    --coco_path /path/to/coco
```

### PyTorch 评估

```bash
python main.py \
    --batch_size 2 \
    --no_aux_loss \
    --eval \
    --resume output/path/to/checkpoint.pth \
    --coco_path /path/to/coco
```

## 实验说明与模型权重

由于 DETR 模型的高计算复杂度以及本实验的计算资源限制（单卡训练、小批次、有限训练轮数），当前的 AP 性能指标尚未达到论文报告的水平，不宜作为模型性能的参考基准。然而，为验证 Jittor 实现与 PyTorch 版本的正确对齐，此处提供相应的模型权重文件 https://pan.baidu.com/s/1bPnYl0jqxvm3Y5oK_VUjow?pwd=qbdf （提取码: qbdf ），用于：

- 验证模型结构的一致性
- 检查数值计算的对齐情况
- 作为后续完整训练的初始化权重

## 训练配置与日志

> - 当前结果基于包含 1000 张图像的小规模数据集验证实验
> - 根据官方 PyTorch 实现，在完整 COCO 数据集上达到论文报告的标准性能（AP ~42.0）需要在 8 张 A100 GPU 上训练 500 个 epoch
> - Transformer 的全局建模能力虽然有效减少了检测框冗余，但其高计算复杂度也带来了更长的收敛时间

### 主要训练参数

此处仅展示优化器超参数，完整参数配置请参见 `main.py`。

#### Jittor 训练配置

| 参数              | 默认值  | 类型  | 说明           |
| ----------------- | ------- | ----- | -------------- |
| `--lr`            | 0.25e-4 | float | 主网络学习率   |
| `--lr_backbone`   | 0.25e-5 | float | 骨干网络学习率 |
| `--batch_size`    | 4       | int   | 训练批次大小   |
| `--weight_decay`  | 1e-4    | float | 权重衰减系数   |
| `--epochs`        | 120     | int   | 训练轮数       |
| `--clip_max_norm` | 0.1     | float | 梯度裁剪阈值   |

#### PyTorch 训练配置

| 参数              | 默认值 | 类型  | 说明           |
| ----------------- | ------ | ----- | -------------- |
| `--lr`            | 1e-4   | float | 主网络学习率   |
| `--lr_backbone`   | 1e-5   | float | 骨干网络学习率 |
| `--batch_size`    | 8      | int   | 训练批次大小   |
| `--weight_decay`  | 1e-4   | float | 权重衰减系数   |
| `--epochs`        | 31     | int   | 训练轮数       |
| `--clip_max_norm` | 0.1    | float | 梯度裁剪阈值   |

### 训练曲线

#### 损失曲线对比

![Training Loss Comparison](./pics-and-logs/training_loss_comparison.png)

#### AP 性能曲线对比

![Performance Comparison](./pics-and-logs/performance_comparison.png)

### 训练日志

完整的训练日志与性能日志请查看：

- **PyTorch 训练日志**: `pics-and-logs/log_torch.txt`，**性能日志**`pics-and-logs/eval_summary_torch.txt`
- **Jittor 日志**: `pics-and-logs/log_jittor.txt`，**性能日志**`pics-and-logs/eval_summary_jittor.txt`

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

| 指标 | PyTorch 最佳值 | Jittor 最佳值 |
| ---- | -------------- | ------------- |
| AP   | 0.0002         | 0.0007        |
| AP50 | 0.0010         | 0.0019        |
| AP75 | 0.0001         | 0.0002        |

## 常见问题

### Jittor 安装问题

#### 1. 缺少 libstdc++.so.6 动态链接库

**错误信息**：

```
ImportError: /root/miniconda3/envs/jt/bin/../lib/libstdc++.so.6: version `GLIBCXX_3.4.30' not found 
(required by /root/.cache/jittor/jt1.3.9/g++11.4.0/py3.7.16/Linux-5.15.0-8xcd/IntelXeonProcex70/be50/default/cu12.1.105_sm_80/jittor_core.cpython-37m-x86_64-linux-gnu.so)
```

**解决方案**：

```bash
ln -sf /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /root/miniconda3/envs/jt/lib/libstdc++.so.6
```

#### 2. cutlass.zip 下载失败

**错误信息**：

```
MD5 mismatch between the server and the downloaded file /root/.cache/jittor/cutlass/cutlass.zip
```

**解决方案**：

- 方法一：手动下载 cutlass.zip

  ```bash
  wget https://cg.cs.tsinghua.edu.cn/jittor/assets/cutlass.zip -O ~/.cache/jittor/cutlass/cutlass.zip
  ```

- 方法二：修改 Jittor 源码中的下载链接为上述地址

### PyTorch 到 Jittor 转换注意事项

#### 1. 高级索引行为差异

**问题描述**：
 PyTorch 和 Jittor 在处理高级索引时存在根本性差异：

- **PyTorch**：对多个索引数组进行配对（zip）操作
- **Jittor/NumPy**：对索引数组进行笛卡尔积操作

**具体表现**：

```python
# PyTorch 中
src_logits[idx]  # idx = (batch_idx, src_idx)
# 会逐对取元素：src_logits[batch_idx[0], src_idx[0]], src_logits[batch_idx[1], src_idx[1]], ...

# Jittor 中
src_logits[idx]  
# 会生成笛卡尔积：所有 batch_idx 与所有 src_idx 的组合
```

**造成的影响**：

- `target_classes[idx] = target_classes_o` 无法正确赋值，导致匹配的 query 仍保持为 no-object 类别
- `src_boxes = outputs["pred_boxes"][idx]` 取出了 `(Q×N_match)` 的交叉矩阵
- 损失被错误地分散，导致 `loss_bbox_unscaled ≈ 0.03`，训练看似正常但实际上：分类损失几乎无梯度，`class_error ≈ 10`，mAP 始终为 0

**解决方案**：

```python
def _make_linear_idx(self, batch_idx, src_idx, num_queries):
    """将二维索引 (batch_idx, query_idx) 转换为一维线性索引"""
    return batch_idx * num_queries + src_idx

# 使用示例
linear_idx = self._make_linear_idx(batch_idx, src_idx, num_queries)
src_logits_flat = src_logits.view(-1, num_classes)
selected_logits = src_logits_flat[linear_idx]
```

#### 2. argmax 返回值差异

**问题描述**：
 Jittor 的 `argmax` 函数返回一个元组 `(indices, values)`，而 PyTorch 只返回索引。

**Jittor argmax 行为**：

```python
>>> x = jt.randn(3, 2)
jt.Var([[-0.1429974  -1.1169171 ]
        [-0.35682714 -1.5031573 ]
        [ 0.66668254  1.1606413 ]], dtype=float32)
>>> jt.argmax(x, 0)
(jt.Var([2 2], dtype=int32),          # 索引
 jt.Var([0.66668254 1.1606413], dtype=float32))  # 对应的最大值
```

**解决方案**：

```python
# PyTorch 代码
indices = torch.argmax(x, dim=0)

# Jittor 等价代码
indices, _ = jt.argmax(x, dim=0)  # 忽略返回的最大值
# 或者
indices = jt.argmax(x, dim=0)[0]  # 只取索引部分
```

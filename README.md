# SODA10M Faster R-CNN Object Detection

This project validates the effectiveness of Faster R-CNN on Huawei's SODA10M 2D object detection dataset.

[中文版本](#中文版本)

## English Version

### Overview

This repository contains a complete implementation for training and evaluating Faster R-CNN models on the SODA10M dataset. The project includes:

- Dataset preparation and conversion to KITTI format
- Model training with both standard and v2 versions
- Model evaluation and inference
- Result visualization

### Project Structure

```
├── train_model.py              # Training script (standard Faster R-CNN)
├── train_model_v2.py           # Training script (Faster R-CNN v2)
├── test_model.py               # Inference script (standard model)
├── test_model_v2.py            # Inference script (v2 model)
├── soda10m_to_kitti.py         # Convert SODA10M to KITTI format
├── visualize_soda10m_annotations.py  # Visualize annotations
├── get_labels_info.py          # Get dataset statistics
├── images/                     # Image directory
├── labels/                     # KITTI format annotations
├── models_split/               # Trained models (standard)
├── models_split_v2/            # Trained models (v2)
├── results_split/              # Results (standard)
├── results_split_v2/           # Results (v2)
└── split_info/                 # Train/val split information
```

### Requirements

- Python 3.8+
- PyTorch 1.9+
- torchvision 0.10+
- OpenCV
- Pillow
- scikit-learn
- matplotlib
- tqdm

Install dependencies:
```bash
pip install -r requirements.txt
```

### Setup

1. Download the SODA10M dataset (labeled_test.tar and labeled_trainval.tar)
2. Extract the dataset to the project directory
3. Run data preparation:
```bash
python soda10m_to_kitti.py
```

### Training

#### Standard Faster R-CNN
```bash
python train_model.py
```

#### Faster R-CNN v2 (recommended, no pretrained weights)
```bash
python train_model_v2.py
```

### Inference

#### Standard Model
```bash
python test_model.py --model-path models_split/latest.pt --image-dir test_images --output-dir test_results
```

#### v2 Model
```bash
python test_model_v2.py --model-path models_split_v2/latest.pt --image-dir test_images --output-dir test_results_v2
```

### Model Details

**Input Size**: 1920×1080 (resized to preserve aspect ratio within min_size=1080, max_size=1920)

**Classes**: Car, Van, Truck, Pedestrian, Person, Cyclist, Tram, Misc

### Notes

- Faster R-CNN v2: Training without pretrained weights avoids overfitting. Training loss decreases while validation mAP and recall remain stable.
- The standard Faster R-CNN with pretrained weights may overfit on this dataset.
- Training/validation split: 8:2 ratio

---

## 中文版本

### 项目概述

本项目在华为的SODA10M数据集上验证Faster R-CNN的效果。包含以下内容：

- 数据集准备和KITTI格式转换
- 模型训练（标准版和v2版）
- 模型评估和推理
- 结果可视化

### 项目结构

```
├── train_model.py              # 训练脚本（标准Faster R-CNN）
├── train_model_v2.py           # 训练脚本（Faster R-CNN v2）
├── test_model.py               # 推理脚本（标准模型）
├── test_model_v2.py            # 推理脚本（v2模型）
├── soda10m_to_kitti.py         # SODA10M转KITTI格式
├── visualize_soda10m_annotations.py  # 可视化标注
├── get_labels_info.py          # 获取数据集统计信息
├── images/                     # 图片目录
├── labels/                     # KITTI格式标注
├── models_split/               # 训练好的模型（标准）
├── models_split_v2/            # 训练好的模型（v2）
├── results_split/              # 结果（标准）
├── results_split_v2/           # 结果（v2）
└── split_info/                 # 训练/验证划分信息
```

### 环境要求

- Python 3.8+
- PyTorch 1.9+
- torchvision 0.10+
- OpenCV
- Pillow
- scikit-learn
- matplotlib
- tqdm

安装依赖：
```bash
pip install -r requirements.txt
```

### 快速开始

1. 下载SODA10M数据集（labeled_test.tar 和 labeled_trainval.tar）
2. 解压数据集到项目目录
3. 进行数据准备：
```bash
python soda10m_to_kitti.py
```

### 模型训练

#### 标准Faster R-CNN
```bash
python train_model.py
```

#### Faster R-CNN v2（推荐，不使用预训练权重）
```bash
python train_model_v2.py
```

### 模型推理

#### 标准模型
```bash
python test_model.py --model-path models_split/latest.pt --image-dir test_images --output-dir test_results
```

#### v2模型
```bash
python test_model_v2.py --model-path models_split_v2/latest.pt --image-dir test_images --output-dir test_results_v2
```

### 模型配置

**输入尺寸**：1920×1080（保持宽高比，最小边1080，最大边1920）

**检测类别**：Car, Van, Truck, Pedestrian, Person, Cyclist, Tram, Misc

### 重要说明

- **Faster R-CNN v2**：训练时不使用预训练权重，可避免过拟合。训练损失持续下降，验证集mAP和召回率保持稳定。
- **标准Faster R-CNN**：使用预训练权重在该数据集上容易过拟合。
- 训练/验证划分比例：8:2

### License

MIT License

### Contact & Contribution

欢迎提交Issue和Pull Request！

#!/usr/bin/env python3
"""
SODA10M数据集Faster R-CNN训练与评估脚本
支持训练集和验证集划分（8:2比例），每次训练15个epoch
支持从最新模型恢复训练
"""

import os
import sys
import json
import time
import random
import numpy as np
from datetime import datetime
from collections import defaultdict, OrderedDict
from pathlib import Path
import shutil

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.ops import box_iou

import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class SODA10MDataset(Dataset):
    """SODA10M数据集类，支持KITTI格式标注"""
    
    def __init__(self, image_dir, annotation_dir, file_list=None, transform=None, 
                 classes=None, max_samples=None):
        """
        初始化数据集
        
        Args:
            image_dir: 图片目录
            annotation_dir: KITTI格式标注目录
            file_list: 指定的文件列表（如果为None，则读取目录下所有文件）
            transform: 数据增强变换
            classes: 类别列表
            max_samples: 最大样本数（用于测试）
        """
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.transform = transform
        
        # 默认类别列表（KITTI格式）
        self.default_classes = [
            'Car', 'Van', 'Truck', 'Pedestrian', 
            'Person', 'Cyclist', 'Tram', 'Misc', 'DontCare'
        ]
        
        self.classes = classes or self.default_classes
        self.class_to_idx = {cls: idx+1 for idx, cls in enumerate(self.classes)}  # 0为背景
        self.idx_to_class = {idx+1: cls for idx, cls in enumerate(self.classes)}
        
        # 收集所有图片文件
        if file_list is not None:
            # 使用指定的文件列表
            self.image_files = file_list
        else:
            # 读取目录下所有图片文件
            self.image_files = []
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            
            for ext in image_extensions:
                files = list(Path(image_dir).glob(f'*{ext}'))
                self.image_files.extend([str(f) for f in files])
            
            # 按字母顺序排序
            self.image_files.sort()
        
        # 限制样本数量（用于快速测试）
        if max_samples and max_samples < len(self.image_files):
            self.image_files = self.image_files[:max_samples]
        
        print(f"数据集初始化完成: {len(self.image_files)} 张图片")
        print(f"类别: {self.classes}")
        
        # 统计有标注的图片数量
        self.annotated_count = 0
        for img_file in self.image_files:
            img_name = os.path.splitext(os.path.basename(img_file))[0]
            anno_path = os.path.join(annotation_dir, f"{img_name}.txt")
            if os.path.exists(anno_path):
                with open(anno_path, 'r') as f:
                    content = f.read().strip()
                    if content:  # 文件非空
                        self.annotated_count += 1
        
        print(f"有标注的图片数量: {self.annotated_count}/{len(self.image_files)}")
    
    def __len__(self):
        return len(self.image_files)
    
    def parse_kitti_annotation(self, annotation_path):
        """
        解析KITTI格式的标注文件
        
        Args:
            annotation_path: 标注文件路径
            
        Returns:
            boxes: 边界框列表 [N, 4] (x1, y1, x2, y2)
            labels: 类别标签列表 [N]
        """
        boxes = []
        labels = []
        
        if not os.path.exists(annotation_path):
            return boxes, labels
        
        try:
            with open(annotation_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 8:  # 至少需要8个字段才能解析边界框
                    continue
                
                # 解析KITTI格式
                obj_type = parts[0]  # 类别
                truncated = float(parts[1])  # 截断程度
                occluded = int(parts[2])  # 遮挡状态
                
                # 跳过DontCare类别
                if obj_type == 'DontCare':
                    continue
                
                # 获取边界框坐标
                try:
                    x1 = float(parts[4])
                    y1 = float(parts[5])
                    x2 = float(parts[6])
                    y2 = float(parts[7])
                except (ValueError, IndexError):
                    continue
                
                # 确保边界框有效
                if x2 <= x1 or y2 <= y1 or x1 < 0 or y1 < 0:
                    continue
                
                # 映射到类别索引
                if obj_type in self.class_to_idx:
                    label = self.class_to_idx[obj_type]
                else:
                    # 如果类别不在列表中，映射到Misc
                    label = self.class_to_idx.get('Misc', 0)
                
                if label == 0:  # 背景类，跳过
                    continue
                
                boxes.append([x1, y1, x2, y2])
                labels.append(label)
        
        except Exception as e:
            print(f"解析标注文件 {annotation_path} 失败: {e}")
        
        return boxes, labels
    
    def __getitem__(self, idx):
        """
        获取数据样本
        
        Returns:
            image: 图片张量 [C, H, W]
            target: 包含边界框和标签的字典
        """
        # 获取图片路径
        img_path = self.image_files[idx]
        
        # 读取图片
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"读取图片失败 {img_path}: {e}")
            # 创建空白图片
            image = Image.new('RGB', (1920, 1080), color='white')
        
        # 获取对应的标注文件路径
        img_name = os.path.splitext(os.path.basename(img_path))[0]
        anno_path = os.path.join(self.annotation_dir, f"{img_name}.txt")
        
        # 解析标注
        boxes, labels = self.parse_kitti_annotation(anno_path)
        
        # 转换为张量
        if len(boxes) > 0:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            
            # 确保边界框维度正确
            if boxes.dim() == 1:
                boxes = boxes.unsqueeze(0)
            if boxes.shape[1] != 4:
                # 修复边界框形状
                boxes = boxes.view(-1, 4)
        else:
            # 没有边界框的情况
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        
        # 构建目标字典
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([idx]),
        }
        
        # 计算面积（只计算有边界框的情况）
        if len(boxes) > 0:
            target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            target['area'] = torch.zeros((0,), dtype=torch.float32)
        
        # 设置iscrowd
        target['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.int64)
        
        # 应用数据增强
        if self.transform:
            image = self.transform(image)
        else:
            # 默认转换
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            image = transform(image)
        
        return image, target

def get_transform(train=True):
    """
    获取数据转换管道
    
    Args:
        train: 是否为训练模式
        
    Returns:
        transform: 数据转换管道
    """
    transforms_list = []
    
    if train:
        # 训练时的数据增强
        transforms_list.extend([
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    
    transforms_list.extend([
        transforms.ToTensor(),
    ])
    
    return transforms.Compose(transforms_list)

def collate_fn(batch):
    """
    自定义批处理函数
    
    Args:
        batch: 批数据
        
    Returns:
        tuple: (images, targets)
    """
    return tuple(zip(*batch))

def get_faster_rcnn_model(num_classes, pretrained=True):
    """
    获取Faster R-CNN模型
    
    Args:
        num_classes: 类别数量（包括背景）
        pretrained: 是否使用预训练权重
        
    Returns:
        model: Faster R-CNN模型
    """
    # 加载预训练的Faster R-CNN模型（输入尺寸改为1920x1080）
    model = fasterrcnn_resnet50_fpn(pretrained=pretrained, min_size=1080, max_size=1920)
    
    # 获取输入特征数
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # 替换预测头以适应自定义类别数
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model

def load_checkpoint(model, checkpoint_path, device='cpu', optimizer=None, lr_scheduler=None):
    """
    从checkpoint文件加载模型参数和训练状态
    
    Args:
        model: 模型实例
        checkpoint_path: checkpoint文件路径
        device: 设备
        optimizer: 优化器实例（可选）
        lr_scheduler: 学习率调度器实例（可选）
        
    Returns:
        dict: 加载的状态字典，包含:
            - epoch: 当前的epoch
            - best_map: 最佳mAP
            - train_losses: 训练损失历史
            - val_metrics: 验证指标历史
    """
    if not os.path.exists(checkpoint_path):
        print(f"警告: Checkpoint文件不存在: {checkpoint_path}")
        return None
    
    print(f"从 {checkpoint_path} 加载checkpoint...")
    
    try:
        # 加载checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # 加载模型参数
        if 'model_state_dict' in checkpoint:
            # 处理不匹配的键
            model_state_dict = checkpoint['model_state_dict']
            current_model_state_dict = model.state_dict()
            
            # 检查类别数是否匹配
            if 'roi_heads.box_predictor.cls_score.weight' in model_state_dict:
                checkpoint_num_classes = model_state_dict['roi_heads.box_predictor.cls_score.weight'].shape[0]
                current_num_classes = current_model_state_dict['roi_heads.box_predictor.cls_score.weight'].shape[0]
                
                if checkpoint_num_classes != current_num_classes:
                    print(f"警告: 类别数不匹配! Checkpoint: {checkpoint_num_classes}, 当前模型: {current_num_classes}")
                    print("这可能是由于使用了不同的类别定义。")
                    print("将尝试加载除分类头外的其他参数...")
                    
                    # 只加载backbone和RPN的参数
                    filtered_state_dict = {}
                    for key, value in model_state_dict.items():
                        if 'roi_heads.box_predictor' not in key:  # 跳过分类头
                            filtered_state_dict[key] = value
                    
                    # 加载过滤后的参数
                    model.load_state_dict(filtered_state_dict, strict=False)
                else:
                    # 类别数匹配，正常加载
                    model.load_state_dict(model_state_dict)
            else:
                # 加载整个模型
                model.load_state_dict(model_state_dict)
        else:
            # 旧格式的checkpoint，直接加载
            model.load_state_dict(checkpoint)
        
        # 加载优化器状态
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("优化器状态已加载")
        
        # 加载学习率调度器状态
        if lr_scheduler is not None and 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            print("学习率调度器状态已加载")
        
        # 提取训练状态
        state = {
            'epoch': checkpoint.get('epoch', 0),
            'best_map': checkpoint.get('best_map', 0.0),
            'train_losses': checkpoint.get('train_losses', []),
            'val_metrics': checkpoint.get('val_metrics', []),
            'config': checkpoint.get('config', {}),
            'classes': checkpoint.get('classes', []),
            'metrics': checkpoint.get('metrics', {})
        }
        
        print(f"成功加载checkpoint:")
        print(f"  - 训练轮次: {state['epoch']}")
        print(f"  - 最佳mAP: {state['best_map']:.4f}")
        print(f"  - 训练损失记录: {len(state['train_losses'])} 个epoch")
        print(f"  - 验证指标记录: {len(state['val_metrics'])} 个epoch")
        
        return state
    
    except Exception as e:
        print(f"加载checkpoint失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def get_all_image_files(image_dir, annotation_dir, test_mode=False, max_samples=None):
    """
    获取所有图片文件列表，并过滤掉没有标注的图片
    
    Args:
        image_dir: 图片目录
        annotation_dir: 标注目录
        test_mode: 是否为测试模式
        max_samples: 最大样本数
        
    Returns:
        list: 有标注的图片文件列表
    """
    # 收集所有图片文件
    image_files = []
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    
    for ext in image_extensions:
        files = list(Path(image_dir).glob(f'*{ext}'))
        image_files.extend([str(f) for f in files])
    
    # 按字母顺序排序
    image_files.sort()
    
    # 过滤掉没有标注的图片
    valid_image_files = []
    print(f"开始过滤图片，总共 {len(image_files)} 张图片...")
    
    for img_file in tqdm(image_files, desc="检查标注文件"):
        img_name = os.path.splitext(os.path.basename(img_file))[0]
        anno_path = os.path.join(annotation_dir, f"{img_name}.txt")
        
        if os.path.exists(anno_path):
            with open(anno_path, 'r') as f:
                content = f.read().strip()
                if content:  # 文件非空
                    valid_image_files.append(img_file)
    
    print(f"有效图片数量: {len(valid_image_files)}/{len(image_files)}")
    
    # 限制样本数量（用于快速测试）
    if test_mode and max_samples and max_samples < len(valid_image_files):
        valid_image_files = valid_image_files[:max_samples]
        print(f"测试模式: 使用 {len(valid_image_files)} 个样本")
    
    return valid_image_files

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10):
    """
    训练一个epoch
    
    Args:
        model: 模型
        optimizer: 优化器
        data_loader: 数据加载器
        device: 设备
        epoch: 当前epoch
        print_freq: 打印频率
        
    Returns:
        float: 平均损失
    """
    model.train()
    total_loss = 0
    loss_classifier = 0
    loss_box_reg = 0
    loss_objectness = 0
    loss_rpn_box_reg = 0
    
    # 进度条
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, (images, targets) in enumerate(pbar):
        # 过滤掉没有目标的样本
        valid_indices = []
        for i, target in enumerate(targets):
            if len(target['boxes']) > 0:
                valid_indices.append(i)
        
        if len(valid_indices) == 0:
            # 如果整个批次都没有目标，跳过
            continue
        
        # 只保留有目标的样本
        images = [images[i] for i in valid_indices]
        targets = [targets[i] for i in valid_indices]
        
        # 将图片和标注移到设备
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # 前向传播
        loss_dict = model(images, targets)
        
        # 计算总损失
        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        
        # 累加各项损失
        total_loss += loss_value
        if 'loss_classifier' in loss_dict:
            loss_classifier += loss_dict['loss_classifier'].item()
        if 'loss_box_reg' in loss_dict:
            loss_box_reg += loss_dict['loss_box_reg'].item()
        if 'loss_objectness' in loss_dict:
            loss_objectness += loss_dict['loss_objectness'].item()
        if 'loss_rpn_box_reg' in loss_dict:
            loss_rpn_box_reg += loss_dict['loss_rpn_box_reg'].item()
        
        # 反向传播
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # 更新进度条
        if (batch_idx + 1) % print_freq == 0 or batch_idx + 1 == len(data_loader):
            pbar.set_postfix({
                'loss': f'{loss_value:.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
    
    # 计算平均损失
    if len(data_loader) > 0:
        avg_loss = total_loss / len(data_loader)
        avg_loss_classifier = loss_classifier / len(data_loader) if loss_classifier > 0 else 0
        avg_loss_box_reg = loss_box_reg / len(data_loader) if loss_box_reg > 0 else 0
        avg_loss_objectness = loss_objectness / len(data_loader) if loss_objectness > 0 else 0
        avg_loss_rpn_box_reg = loss_rpn_box_reg / len(data_loader) if loss_rpn_box_reg > 0 else 0
    else:
        avg_loss = 0
        avg_loss_classifier = 0
        avg_loss_box_reg = 0
        avg_loss_objectness = 0
        avg_loss_rpn_box_reg = 0
    
    print(f"\nEpoch {epoch} 训练完成:")
    print(f"  总平均损失: {avg_loss:.4f}")
    print(f"  分类损失: {avg_loss_classifier:.4f}")
    print(f"  边界框回归损失: {avg_loss_box_reg:.4f}")
    print(f"  目标性损失: {avg_loss_objectness:.4f}")
    print(f"  RPN边界框回归损失: {avg_loss_rpn_box_reg:.4f}")
    
    return avg_loss

def evaluate_model(model, data_loader, device, iou_threshold=0.5, score_threshold=0.5):
    """
    评估模型性能
    
    Args:
        model: 模型
        data_loader: 数据加载器
        device: 设备
        iou_threshold: IoU阈值
        score_threshold: 置信度阈值
        
    Returns:
        dict: 评估指标
    """
    model.eval()
    
    # 验证集损失统计
    total_val_loss = 0
    val_loss_count = 0
    
    # 统计变量
    total_gt_boxes = 0
    total_pred_boxes = 0
    total_true_positives = 0
    
    # 按类别统计
    class_stats = defaultdict(lambda: {
        'gt_count': 0,
        'pred_count': 0,
        'tp_count': 0
    })
    
    # 用于计算mAP
    all_predictions = []
    all_ground_truths = []
    
    print("开始评估模型...")
    
    # 切换到evaluation模式但计算loss
    model.train()  # 保持训练模式以计算loss
    
    with torch.no_grad():
        for batch_idx, (images, targets) in enumerate(tqdm(data_loader, desc="评估")):
            # 过滤掉没有目标的样本
            valid_indices = []
            for i, target in enumerate(targets):
                if len(target['boxes']) > 0:
                    valid_indices.append(i)
            
            if len(valid_indices) == 0:
                # 如果整个批次都没有目标，跳过
                continue
            
            # 只保留有目标的样本
            images = [images[i] for i in valid_indices]
            targets = [targets[i] for i in valid_indices]
            
            # 将图片和标注移到设备
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            # 计算验证集损失
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            total_val_loss += losses.item()
            val_loss_count += 1
            
            # 切换到eval模式获取预测
            model.eval()
            predictions = model(images)
            model.train()  # 切换回train模式
            
            for i, (pred, target) in enumerate(zip(predictions, targets)):
                # 获取预测结果
                pred_boxes = pred['boxes'].cpu()
                pred_scores = pred['scores'].cpu()
                pred_labels = pred['labels'].cpu()
                
                # 获取真实标注
                gt_boxes = target['boxes'].cpu()
                gt_labels = target['labels'].cpu()
                
                # 过滤低置信度预测
                if len(pred_boxes) > 0:
                    keep = pred_scores >= score_threshold
                    pred_boxes = pred_boxes[keep]
                    pred_scores = pred_scores[keep]
                    pred_labels = pred_labels[keep]
                else:
                    pred_boxes = torch.zeros((0, 4))
                    pred_scores = torch.zeros((0,))
                    pred_labels = torch.zeros((0,), dtype=torch.int64)
                
                # 统计数量
                total_gt_boxes += len(gt_boxes)
                total_pred_boxes += len(pred_boxes)
                
                # 按类别统计真实标注数量
                for label in gt_labels:
                    class_idx = label.item()
                    class_stats[class_idx]['gt_count'] += 1
                
                # 按类别统计预测数量
                for label in pred_labels:
                    class_idx = label.item()
                    class_stats[class_idx]['pred_count'] += 1
                
                # 计算真阳性
                if len(gt_boxes) > 0 and len(pred_boxes) > 0:
                    # 计算IoU矩阵
                    iou_matrix = box_iou(pred_boxes, gt_boxes)
                    
                    # 为每个真实框找到匹配的预测框
                    for gt_idx in range(len(gt_boxes)):
                        # 找到与该真实框IoU最大的预测框
                        if len(pred_boxes) > 0:
                            max_iou, pred_idx = iou_matrix[:, gt_idx].max(0)
                            
                            if max_iou >= iou_threshold:
                                # 检查类别是否匹配
                                if pred_labels[pred_idx] == gt_labels[gt_idx]:
                                    total_true_positives += 1
                                    class_idx = gt_labels[gt_idx].item()
                                    class_stats[class_idx]['tp_count'] += 1
                
                # 保存预测结果用于计算mAP
                all_predictions.append({
                    'boxes': pred_boxes.numpy(),
                    'scores': pred_scores.numpy(),
                    'labels': pred_labels.numpy()
                })
                
                all_ground_truths.append({
                    'boxes': gt_boxes.numpy(),
                    'labels': gt_labels.numpy()
                })
    
    # 计算总体指标
    precision = total_true_positives / total_pred_boxes if total_pred_boxes > 0 else 0
    recall = total_true_positives / total_gt_boxes if total_gt_boxes > 0 else 0
    
    # 计算F1分数
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 按类别计算指标
    class_metrics = {}
    for class_idx, stats in class_stats.items():
        gt_count = stats['gt_count']
        pred_count = stats['pred_count']
        tp_count = stats['tp_count']
        
        class_precision = tp_count / pred_count if pred_count > 0 else 0
        class_recall = tp_count / gt_count if gt_count > 0 else 0
        class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall) if (class_precision + class_recall) > 0 else 0
        
        class_metrics[class_idx] = {
            'precision': class_precision,
            'recall': class_recall,
            'f1': class_f1,
            'gt_count': gt_count,
            'pred_count': pred_count,
            'tp_count': tp_count
        }
    
    # 计算mAP
    mean_ap = calculate_map(all_predictions, all_ground_truths, iou_threshold, device)
    
    # 计算平均验证损失
    avg_val_loss = total_val_loss / val_loss_count if val_loss_count > 0 else 0
    
    # 恢复到eval模式
    model.eval()
    
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mean_average_precision': mean_ap,
        'val_loss': avg_val_loss,
        'total_gt_boxes': total_gt_boxes,
        'total_pred_boxes': total_pred_boxes,
        'total_true_positives': total_true_positives,
        'class_metrics': class_metrics
    }
    
    return metrics

def calculate_map(predictions, ground_truths, iou_threshold, device):
    """
    计算mAP（平均精度均值）
    
    Args:
        predictions: 预测结果列表
        ground_truths: 真实标注列表
        iou_threshold: IoU阈值
        device: 设备
        
    Returns:
        float: mAP值
    """
    # 如果没有预测结果，返回0
    if not predictions or not ground_truths:
        return 0.0
    
    # 按类别收集预测和真实标注
    class_predictions = defaultdict(list)
    class_ground_truths = defaultdict(list)
    
    for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
        # 处理预测
        for box, score, label in zip(pred['boxes'], pred['scores'], pred['labels']):
            class_predictions[label].append({
                'img_idx': img_idx,
                'box': box,
                'score': score,
                'matched': False
            })
        
        # 处理真实标注
        for box, label in zip(gt['boxes'], gt['labels']):
            class_ground_truths[label].append({
                'img_idx': img_idx,
                'box': box,
                'matched': False
            })
    
    # 计算每个类别的AP
    ap_scores = []
    
    for class_idx in class_predictions.keys():
        if class_idx not in class_ground_truths:
            continue
        
        # 获取该类别的预测和真实标注
        preds = class_predictions[class_idx]
        gts = class_ground_truths[class_idx]
        
        # 按置信度降序排序
        preds.sort(key=lambda x: x['score'], reverse=True)
        
        # 计算精度和召回率
        tp = np.zeros(len(preds))
        fp = np.zeros(len(preds))
        
        # 统计每个预测框是否匹配
        for i, pred in enumerate(preds):
            # 找到同张图片中的真实标注
            img_gts = [gt for gt in gts if gt['img_idx'] == pred['img_idx'] and not gt['matched']]
            
            if len(img_gts) == 0:
                fp[i] = 1
                continue
            
            # 计算与所有真实标注的IoU
            ious = []
            for gt in img_gts:
                # 计算IoU
                pred_box = torch.tensor(pred['box']).unsqueeze(0)
                gt_box = torch.tensor(gt['box']).unsqueeze(0)
                iou = box_iou(pred_box, gt_box).item()
                ious.append(iou)
            
            max_iou = max(ious) if ious else 0
            max_iou_idx = ious.index(max_iou) if ious else -1
            
            if max_iou >= iou_threshold:
                # 匹配成功
                tp[i] = 1
                img_gts[max_iou_idx]['matched'] = True
            else:
                # 匹配失败
                fp[i] = 1
        
        # 计算累计的精度和召回率
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / len(gts) if len(gts) > 0 else np.zeros_like(tp_cumsum)
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
        
        # 计算AP（11点插值法）
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            mask = recalls >= t
            if mask.any():
                p = precisions[mask].max()
            else:
                p = 0
            ap += p / 11
        
        ap_scores.append(ap)
    
    # 计算mAP
    mean_ap = np.mean(ap_scores) if ap_scores else 0
    
    return mean_ap

def save_model(model, optimizer, epoch, metrics, save_path, classes, config, train_losses, val_metrics, best_map, lr_scheduler=None):
    """
    保存模型和训练状态
    
    Args:
        model: 模型
        optimizer: 优化器
        epoch: 当前epoch
        metrics: 评估指标
        save_path: 保存路径
        classes: 类别列表
        config: 训练配置
        train_losses: 训练损失历史
        val_metrics: 验证指标历史
        best_map: 最佳mAP
        lr_scheduler: 学习率调度器
    """
    checkpoint = {
        'epoch': epoch,
        'best_map': best_map,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'lr_scheduler_state_dict': lr_scheduler.state_dict() if lr_scheduler else None,
        'metrics': metrics,
        'train_losses': train_losses,
        'val_metrics': val_metrics,
        'classes': classes,
        'config': config,
        'save_time': datetime.now().isoformat()
    }
    
    torch.save(checkpoint, save_path)
    print(f"模型已保存到: {save_path}")

def plot_training_metrics(train_losses, val_metrics, save_dir):
    """
    绘制训练指标
    
    Args:
        train_losses: 训练损失列表
        val_metrics: 验证指标列表
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 绘制训练损失
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.plot(train_losses, label='Training Loss', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.grid(True)
    
    # 提取验证指标
    epochs = list(range(1, len(val_metrics) + 1))
    precisions = [m['precision'] for m in val_metrics]
    recalls = [m['recall'] for m in val_metrics]
    maps = [m['mean_average_precision'] for m in val_metrics]
    val_losses = [m.get('val_loss', 0) for m in val_metrics]
    
    plt.subplot(2, 3, 2)
    plt.plot(epochs, precisions, label='Precision', marker='o')
    plt.plot(epochs, recalls, label='Recall', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.title('Precision and Recall')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 3)
    plt.plot(epochs, maps, label='mAP', color='green', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('mAP')
    plt.title('Mean Average Precision')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 4)
    plt.plot(train_losses, label='Training Loss', color='blue', marker='o')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 5)
    f1_scores = [m['f1_score'] for m in val_metrics]
    plt.plot(epochs, f1_scores, label='F1 Score', color='purple', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.title('F1 Score During Training')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(2, 3, 6)
    plt.plot(epochs, [m['total_true_positives'] for m in val_metrics], label='True Positives', marker='o', color='orange')
    plt.plot(epochs, [m['total_pred_boxes'] for m in val_metrics], label='Predicted Boxes', marker='s', color='green')
    plt.plot(epochs, [m['total_gt_boxes'] for m in val_metrics], label='Ground Truth Boxes', marker='^', color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Count')
    plt.title('Detection Statistics')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"训练指标图已保存到: {os.path.join(save_dir, 'training_metrics.png')}")

def save_split_info(train_files, val_files, save_dir):
    """
    保存数据集划分信息
    
    Args:
        train_files: 训练集文件列表
        val_files: 验证集文件列表
        save_dir: 保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存训练集文件列表
    with open(os.path.join(save_dir, 'train_files.txt'), 'w') as f:
        for file in train_files:
            f.write(f"{file}\n")
    
    # 保存验证集文件列表
    with open(os.path.join(save_dir, 'val_files.txt'), 'w') as f:
        for file in val_files:
            f.write(f"{file}\n")
    
    # 保存划分统计信息
    split_info = {
        'train_count': len(train_files),
        'val_count': len(val_files),
        'total_count': len(train_files) + len(val_files),
        'train_ratio': len(train_files) / (len(train_files) + len(val_files)),
        'val_ratio': len(val_files) / (len(train_files) + len(val_files)),
        'split_date': datetime.now().isoformat()
    }
    
    with open(os.path.join(save_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"数据集划分信息已保存到: {save_dir}")

def main():
    """主函数：训练和评估Faster R-CNN模型"""
    import argparse
    
    parser = argparse.ArgumentParser(description='训练Faster R-CNN模型，支持从现有模型继续训练')
    parser.add_argument('--model_path', type=str, default='models_export/faster_rcnn_best.pth',
                       help='从指定的模型文件恢复训练 (默认: models_export/faster_rcnn_best.pth)')
    parser.add_argument('--epochs_per_session', type=int, default=25,
                       help='每次训练轮次 (默认: 25)')
    parser.add_argument('--force_restart', action='store_true',
                       help='强制重新开始训练，忽略现有模型')
    
    args = parser.parse_args()
    
    # 设置参数
    config = {
        # 数据路径
        'image_dir': 'images',
        'annotation_dir': 'labels',
        
        # 训练参数
        'batch_size': 2,  # 减小批大小以节省显存
        'epochs_per_session': args.epochs_per_session,  # 每次训练15个epoch
        'learning_rate': 0.005,
        'weight_decay': 0.0005,
        
        # 数据集划分参数
        'train_ratio': 0.8,  # 训练集比例
        'val_ratio': 0.2,    # 验证集比例
        'random_seed': 42,   # 随机种子，每次随即划分结果
        
        # 评估参数
        'iou_threshold': 0.5,
        'score_threshold': 0.5,
        
        # 设备
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        
        # 输出路径
        'model_save_dir': 'models_export',
        'results_save_dir': 'results_export',
        'split_info_dir': 'split_info',
        
        # 类别
        'classes': ['Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck', 'Bus', 'Misc'],
        
        # 测试模式
        'test_mode': False,  # 设置为True使用少量数据快速测试
        'max_samples': 200  # 测试时使用的最大样本数
    }
    
    print("=" * 80)
    print("SODA10M Faster R-CNN 训练与评估 (8:2 划分)")
    print("每次训练15个epoch，支持从现有模型恢复训练")
    print("=" * 80)
    print(f"设备: {config['device']}")
    print(f"图片目录: {config['image_dir']}")
    print(f"标注目录: {config['annotation_dir']}")
    print(f"类别: {config['classes']}")
    print(f"每次训练轮次: {config['epochs_per_session']}")
    print(f"批大小: {config['batch_size']}")
    print(f"学习率: {config['learning_rate']}")
    print(f"是否从现有模型恢复: {'是' if not args.force_restart else '否'}")
    if not args.force_restart:
        print(f"模型文件: {args.model_path}")
    print("=" * 80)
    
    # 检查路径
    if not os.path.exists(config['image_dir']):
        print(f"错误: 图片目录不存在: {config['image_dir']}")
        return
    
    if not os.path.exists(config['annotation_dir']):
        print(f"错误: 标注目录不存在: {config['annotation_dir']}")
        return
    
    # 创建输出目录
    os.makedirs(config['model_save_dir'], exist_ok=True)
    os.makedirs(config['results_save_dir'], exist_ok=True)
    os.makedirs(config['split_info_dir'], exist_ok=True)
    
    # 准备数据集
    print("准备数据集...")
    
    # 获取所有有效图片文件
    all_image_files = get_all_image_files(
        config['image_dir'],
        config['annotation_dir'],
        test_mode=config['test_mode'],
        max_samples=config['max_samples']
    )
    
    if len(all_image_files) == 0:
        print("错误: 没有找到有效的图片文件")
        return
    
    print(f"总共找到 {len(all_image_files)} 个有效样本")
    
    # 划分训练集和验证集 (8:2)
    print(f"划分训练集和验证集 ({config['train_ratio']}:{config['val_ratio']})...")
    train_files, val_files = train_test_split(
        all_image_files,
        test_size=config['val_ratio'],
        train_size=config['train_ratio'],
        random_state=config['random_seed'],
        shuffle=True
    )
    
    print(f"训练集大小: {len(train_files)}")
    print(f"验证集大小: {len(val_files)}")
    
    # 保存划分信息
    save_split_info(train_files, val_files, config['split_info_dir'])
    
    # 创建训练集
    print("创建训练集...")
    train_dataset = SODA10MDataset(
        image_dir=config['image_dir'],
        annotation_dir=config['annotation_dir'],
        file_list=train_files,
        transform=get_transform(train=True),
        classes=config['classes']
    )
    
    # 创建验证集
    print("创建验证集...")
    val_dataset = SODA10MDataset(
        image_dir=config['image_dir'],
        annotation_dir=config['annotation_dir'],
        file_list=val_files,
        transform=get_transform(train=False),
        classes=config['classes']
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,  # 设置为0以避免多进程问题
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,  # 验证时使用批大小为1
        shuffle=False,
        num_workers=0,  # 设置为0以避免多进程问题
        collate_fn=collate_fn
    )
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    
    # 创建模型
    print("创建模型...")
    num_classes = len(config['classes']) + 1  # 包括背景类
    model = get_faster_rcnn_model(num_classes, pretrained=True)
    model.to(config['device'])
    
    # 创建优化器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=config['learning_rate'],
        momentum=0.9,
        weight_decay=config['weight_decay']
    )
    
    # 创建学习率调度器
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    # 初始化训练状态
    start_epoch = 0
    best_map = 0.0
    train_losses = []
    val_metrics_list = []
    
    # 从现有模型恢复训练（除非强制重新开始）
    if not args.force_restart and os.path.exists(args.model_path):
        print(f"尝试从现有模型恢复训练: {args.model_path}")
        checkpoint_state = load_checkpoint(
            model, 
            args.model_path, 
            device=config['device'],
            optimizer=optimizer,
            lr_scheduler=lr_scheduler
        )
        
        if checkpoint_state:
            # 恢复训练状态
            start_epoch = checkpoint_state['epoch'] + 1
            best_map = checkpoint_state.get('best_map', 0.0)
            train_losses = checkpoint_state.get('train_losses', [])
            val_metrics_list = checkpoint_state.get('val_metrics', [])
            
            # 检查类别是否一致
            checkpoint_classes = checkpoint_state.get('classes', [])
            if checkpoint_classes != config['classes']:
                print(f"警告: 类别不一致!")
                print(f"Checkpoint类别: {checkpoint_classes}")
                print(f"当前类别: {config['classes']}")
            
            print(f"将从第 {start_epoch} 个epoch继续训练")
            print(f"当前最佳mAP: {best_map:.4f}")
        else:
            print("加载checkpoint失败，将从第0个epoch开始训练")
    else:
        if args.force_restart:
            print("强制重新开始训练，忽略现有模型")
        else:
            print(f"未找到模型文件: {args.model_path}")
        print("将从第0个epoch开始训练")
    
    # 训练和评估
    print("开始训练...")
    
    for epoch in range(start_epoch, start_epoch + config['epochs_per_session']):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{start_epoch + config['epochs_per_session']} (总epoch: {len(train_losses)+1})")
        print(f"{'='*60}")
        
        # 训练
        train_loss = train_one_epoch(model, optimizer, train_loader, config['device'], epoch+1)
        train_losses.append(train_loss)
        
        # 更新学习率
        lr_scheduler.step()
        
        # 评估
        print("\n评估模型...")
        val_metrics = evaluate_model(
            model, 
            val_loader, 
            config['device'],
            iou_threshold=config['iou_threshold'],
            score_threshold=config['score_threshold']
        )
        
        # 评估后恢复训练模式
        model.train()
        
        val_metrics_list.append(val_metrics)
        
        # 打印评估结果
        print(f"\n评估结果 (Epoch {epoch+1}):")
        print(f"  验证集损失 (Val Loss): {val_metrics['val_loss']:.4f}")
        print(f"  精确度 (Precision): {val_metrics['precision']:.4f}")
        print(f"  召回率 (Recall): {val_metrics['recall']:.4f}")
        print(f"  F1分数: {val_metrics['f1_score']:.4f}")
        print(f"  mAP: {val_metrics['mean_average_precision']:.4f}")
        print(f"  真实框总数: {val_metrics['total_gt_boxes']}")
        print(f"  预测框总数: {val_metrics['total_pred_boxes']}")
        print(f"  真阳性数: {val_metrics['total_true_positives']}")
        
        # 保存当前epoch的模型
        model_path = os.path.join(config['model_save_dir'], f"faster_rcnn_epoch_{epoch+1}.pth")
        save_model(model, optimizer, epoch+1, val_metrics, model_path, 
                  config['classes'], config, train_losses, val_metrics_list, best_map, lr_scheduler)
        
        # 保存最佳模型
        if val_metrics['mean_average_precision'] > best_map:
            best_map = val_metrics['mean_average_precision']
            best_model_path = os.path.join(config['model_save_dir'], "faster_rcnn_best.pth")
            
            save_model(model, optimizer, epoch+1, val_metrics, best_model_path, 
                      config['classes'], config, train_losses, val_metrics_list, best_map, lr_scheduler)
    
    # 训练完成
    print(f"\n{'='*80}")
    print("本次训练完成!")
    print(f"{'='*80}")
    print(f"训练轮次: {len(train_losses)}")
    print(f"最佳mAP: {best_map:.4f}")
    print(f"最佳模型: {os.path.join(config['model_save_dir'], 'faster_rcnn_best.pth')}")
    
    # 保存最终结果
    final_results = {
        'config': config,
        'train_losses': train_losses,
        'val_metrics': val_metrics_list,
        'best_map': best_map,
        'best_model_path': os.path.join(config['model_save_dir'], 'faster_rcnn_best.pth'),
        'training_date': datetime.now().isoformat()
    }
    
    results_path = os.path.join(config['results_save_dir'], 'training_results.json')
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    print(f"训练结果已保存到: {results_path}")
    
    # 绘制训练指标
    plot_training_metrics(train_losses, val_metrics_list, config['results_save_dir'])
    
    # 打印类别级别的详细结果
    print(f"\n{'='*80}")
    print("类别级别详细结果:")
    print(f"{'='*80}")
    
    if val_metrics_list:
        last_metrics = val_metrics_list[-1]
        class_metrics = last_metrics['class_metrics']
        
        for class_idx, metrics in class_metrics.items():
            class_name = train_dataset.idx_to_class.get(class_idx, f"Class_{class_idx}")
            print(f"\n{class_name}:")
            print(f"  精确度: {metrics['precision']:.4f}")
            print(f"  召回率: {metrics['recall']:.4f}")
            print(f"  F1分数: {metrics['f1']:.4f}")
            print(f"  真实框数: {metrics['gt_count']}")
            print(f"  预测框数: {metrics['pred_count']}")
            print(f"  真阳性数: {metrics['tp_count']}")
    
    print(f"\n{'='*80}")
    print("训练完成!")
    print(f"模型保存在: {config['model_save_dir']}")
    print(f"结果保存在: {config['results_save_dir']}")
    print(f"划分信息保存在: {config['split_info_dir']}")
    print(f"{'='*80}")

if __name__ == '__main__':
    # 运行主函数
    main()
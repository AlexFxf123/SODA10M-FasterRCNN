#!/usr/bin/env python3
"""
Faster R-CNN 目标检测完整测试脚本
加载训练好的模型，对测试图片进行检测，并可视化结果
"""

import os
import sys
import json
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import cv2
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_trained_model(model_path, device='cuda'):
    """
    加载训练好的模型
    
    Args:
        model_path: 模型文件路径
        device: 运行设备
        
    Returns:
        tuple: (模型, 类别列表, checkpoint信息)
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    print(f"加载模型: {model_path}")
    
    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # 获取类别信息
    classes = checkpoint.get('classes', [])
    if not classes:
        config = checkpoint.get('config', {})
        classes = config.get('classes', ['Car', 'Pedestrian', 'Cyclist', 'Van', 'Truck', 'Bus', 'Misc'])
    
    print(f"检测类别: {classes}")
    
    # 创建模型 - 使用新的API避免弃用警告
    num_classes = len(classes) + 1  # 包括背景类
    model = fasterrcnn_resnet50_fpn(weights=None, min_size=1080, max_size=1920)  # 输入尺寸改为1920x1080
    
    # 替换预测头
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # 加载模型权重
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # 移动到设备
    model.to(device)
    model.eval()
    
    return model, classes, checkpoint

def get_image_files(image_dir, extensions=None):
    """
    获取目录中的所有图片文件
    
    Args:
        image_dir: 图片目录
        extensions: 支持的图片扩展名
        
    Returns:
        list: 图片文件路径列表
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
    
    image_files = []
    
    for ext in extensions:
        # 查找不区分大小写的文件
        for pattern in [f'*{ext}', f'*{ext.upper()}']:
            files = list(Path(image_dir).glob(pattern))
            image_files.extend([str(f) for f in files])
    
    # 去重并排序
    image_files = list(set(image_files))
    image_files.sort()
    
    return image_files

def preprocess_image(image_path, target_size=None):
    """
    预处理图片
    
    Args:
        image_path: 图片路径
        target_size: 目标尺寸 (width, height)
        
    Returns:
        tuple: (原始图片, 预处理后的tensor, 原始尺寸)
    """
    # 读取图片
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (width, height)
    
    # 调整尺寸（如果需要）
    if target_size:
        image = image.resize(target_size, Image.Resampling.LANCZOS)
    
    # 转换为tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image)
    
    return image, image_tensor, original_size

def detect_objects(model, image_tensor, device, score_threshold=0.5):
    """
    使用模型检测物体
    
    Args:
        model: 模型
        image_tensor: 图片tensor
        device: 设备
        score_threshold: 置信度阈值
        
    Returns:
        dict: 检测结果
    """
    with torch.no_grad():
        # 添加batch维度
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        
        # 移动到设备
        image_tensor = image_tensor.to(device)
        
        # 预测
        predictions = model(image_tensor)
        
        # 提取结果
        pred = predictions[0]
        
        # 过滤低置信度结果
        keep = pred['scores'] >= score_threshold
        boxes = pred['boxes'][keep].cpu().numpy()
        scores = pred['scores'][keep].cpu().numpy()
        labels = pred['labels'][keep].cpu().numpy()
        
        return {
            'boxes': boxes,
            'scores': scores,
            'labels': labels
        }

def draw_detections(image, detections, classes, score_threshold=0.5):
    """
    在图片上绘制检测框
    
    Args:
        image: PIL图片
        detections: 检测结果
        classes: 类别列表
        score_threshold: 置信度阈值
        
    Returns:
        Image: 绘制了检测框的图片
    """
    # 转换为numpy数组 (BGR格式)
    img_np = np.array(image)
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    img_height, img_width = img_np.shape[:2]
    
    # 类别颜色映射
    class_colors = {
        'Car': (0, 0, 255),        # 红色
        'Pedestrian': (0, 255, 0),  # 绿色
        'Cyclist': (255, 0, 0),     # 蓝色
        'Van': (0, 255, 255),       # 黄色
        'Truck': (0, 165, 255),     # 橙色
        'Bus': (255, 0, 255),       # 紫色
        'Misc': (192, 192, 192),    # 银色
    }
    
    boxes = detections['boxes']
    scores = detections['scores']
    labels = detections['labels']
    
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if score < score_threshold:
            continue
        
        # 获取类别名称
        if 0 < label <= len(classes):
            class_name = classes[label - 1]
        else:
            class_name = f"Class_{label}"
        
        # 获取颜色
        color = class_colors.get(class_name, (255, 255, 255))
        
        # 确保坐标在范围内
        x1, y1, x2, y2 = [int(coord) for coord in box]
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))
        
        # 绘制边界框
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
        
        # 准备标签文本
        label_text = f"{class_name}: {score:.2f}"
        
        # 计算文本位置
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), _ = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # 绘制文本背景
        text_y = y1 - 5 if y1 > 20 else y1 + 20
        cv2.rectangle(img_np,
                     (x1, text_y - text_height - 5),
                     (x1 + text_width, text_y + 5),
                     color, -1)
        
        # 绘制文本
        cv2.putText(img_np, label_text,
                   (x1, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, (255, 255, 255), thickness)
    
    # 转换回RGB格式
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(img_rgb)

def save_results(image_path, detections, output_dir, classes):
    """
    保存检测结果
    
    Args:
        image_path: 原始图片路径
        detections: 检测结果
        output_dir: 输出目录
        classes: 类别列表
        
    Returns:
        tuple: (图片保存路径, JSON保存路径)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取图片基本信息
    image_name = os.path.splitext(os.path.basename(image_path))[0]
    
    # 保存JSON结果
    json_path = os.path.join(output_dir, f"{image_name}.json")
    
    results = {
        'image': os.path.basename(image_path),
        'timestamp': datetime.now().isoformat(),
        'detections': []
    }
    
    boxes = detections['boxes']
    scores = detections['scores']
    labels = detections['labels']
    
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        if 0 < label <= len(classes):
            class_name = classes[label - 1]
        else:
            class_name = f"Class_{label}"
        
        x1, y1, x2, y2 = box
        
        detection = {
            'id': i + 1,
            'class': class_name,
            'confidence': float(score),
            'bbox': [float(x1), float(y1), float(x2), float(y2)]
        }
        
        results['detections'].append(detection)
    
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return json_path

def process_single_image(model, image_path, output_dir, classes, device, 
                        score_threshold=0.5, save_image=True):
    """
    处理单张图片
    
    Returns:
        dict: 处理结果
    """
    try:
        # 预处理图片
        original_image, image_tensor, original_size = preprocess_image(image_path)
        
        # 检测物体
        detections = detect_objects(model, image_tensor, device, score_threshold)
        
        print(f"检测到 {len(detections['boxes'])} 个物体")
        
        # 绘制检测结果
        if save_image:
            result_image = draw_detections(original_image, detections, classes, score_threshold)
            
            # 保存图片
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            image_output_path = os.path.join(output_dir, f"{image_name}_detection.jpg")
            result_image.save(image_output_path)
        else:
            image_output_path = None
        
        # 保存JSON结果
        json_path = save_results(image_path, detections, output_dir, classes)
        
        return {
            'image_path': image_path,
            'detections': detections,
            'image_output_path': image_output_path,
            'json_output_path': json_path,
            'detection_count': len(detections['boxes'])
        }
        
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")
        return None

def process_directory(model, image_dir, output_dir, classes, device, 
                     score_threshold=0.5, max_images=None):
    """
    处理整个目录的图片
    
    Returns:
        list: 所有图片的处理结果
    """
    # 获取所有图片文件
    image_files = get_image_files(image_dir)
    
    if not image_files:
        print(f"在目录 {image_dir} 中没有找到图片文件")
        return []
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 限制处理数量
    if max_images and max_images < len(image_files):
        image_files = image_files[:max_images]
        print(f"将处理前 {len(image_files)} 张图片")
    
    # 处理每张图片
    all_results = []
    
    for image_path in tqdm(image_files, desc="处理图片"):
        result = process_single_image(
            model, image_path, output_dir, classes, device,
            score_threshold, save_image=True
        )
        
        if result:
            all_results.append(result)
    
    return all_results

def generate_statistics(results, output_dir, classes):
    """
    生成统计报告
    
    Args:
        results: 处理结果列表
        output_dir: 输出目录
        classes: 类别列表
    """
    if not results:
        return
    
    # 统计信息
    total_images = len(results)
    total_detections = sum(r['detection_count'] for r in results)
    avg_detections = total_detections / total_images if total_images > 0 else 0
    
    # 按类别统计 - 修复：将标签转换为字符串
    class_counts = defaultdict(int)
    for result in results:
        detections = result['detections']
        for label in detections['labels']:
            # 将numpy.int64转换为Python int
            label_int = int(label)
            if 0 < label_int <= len(classes):
                class_name = classes[label_int - 1]
            else:
                class_name = f"Unknown_{label_int}"
            class_counts[class_name] += 1
    
    # 创建统计报告
    stats = {
        'total_images_processed': total_images,
        'total_detections': total_detections,
        'average_detections_per_image': float(avg_detections),  # 转换为float
        'class_distribution': dict(class_counts),  # 现在键是字符串
        'processing_date': datetime.now().isoformat(),
        'classes': classes
    }
    
    # 保存统计报告
    stats_path = os.path.join(output_dir, 'detection_statistics.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n统计报告已保存到: {stats_path}")
    
    # 打印摘要
    print("\n" + "="*60)
    print("检测结果统计摘要")
    print("="*60)
    print(f"处理图片数量: {stats['total_images_processed']}")
    print(f"总检测物体数: {stats['total_detections']}")
    print(f"平均每张图片检测数: {stats['average_detections_per_image']:.2f}")
    print("\n类别分布:")
    for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {class_name}: {count} 个")
    print("="*60)
    
    return stats_path

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Faster R-CNN目标检测测试')
    parser.add_argument('--model', type=str, default='models_split/faster_rcnn_best.pth',
                       help='训练好的模型路径 (默认: models_split/faster_rcnn_best.pth)')
    parser.add_argument('--test_dir', type=str, default='test_images',
                       help='测试图片目录 (默认: test_images)')
    parser.add_argument('--output_dir', type=str, default='test_results',
                       help='结果输出目录 (默认: test_results)')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                       help='置信度阈值 (默认: 0.5)')
    parser.add_argument('--max_images', type=int, default=None,
                       help='最大处理图片数量 (默认: None, 处理所有图片)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                       help='运行设备 (cuda/cpu) (默认: 自动选择)')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("="*80)
    print("Faster R-CNN 目标检测测试")
    print("="*80)
    print(f"模型文件: {args.model}")
    print(f"测试目录: {args.test_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"置信度阈值: {args.score_threshold}")
    print(f"运行设备: {device}")
    print("="*80)
    
    # 检查路径
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        print(f"请确保模型文件存在，或使用 --model 参数指定正确的路径")
        return
    
    if not os.path.exists(args.test_dir):
        print(f"错误: 测试目录不存在: {args.test_dir}")
        print(f"请确保测试图片目录存在，或使用 --test_dir 参数指定正确的路径")
        return
    
    # 加载模型
    try:
        model, classes, checkpoint = load_trained_model(args.model, device)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 处理图片
    results = process_directory(
        model=model,
        image_dir=args.test_dir,
        output_dir=args.output_dir,
        classes=classes,
        device=device,
        score_threshold=args.score_threshold,
        max_images=args.max_images
    )
    
    if results:
        # 生成统计报告
        generate_statistics(results, args.output_dir, classes)
        
        print(f"\n处理完成! 共处理 {len(results)} 张图片")
        print(f"结果保存在: {args.output_dir}")
    else:
        print("\n没有处理任何图片")
    
    print("="*80)

if __name__ == '__main__':
    # 如果直接运行，使用示例参数
    if len(sys.argv) == 1:
        print("使用示例参数运行...")
        print("您可以指定参数: python test_detection.py --model model.pth --test_dir images --output_dir results")
        print()
        
        # 检查是否存在测试图片目录
        test_dir = "test_images"
        if os.path.exists(test_dir):
            main()
        else:
            print(f"测试图片目录不存在: {test_dir}")
            print("请创建测试图片目录或使用 --test_dir 参数指定")
    else:
        main()

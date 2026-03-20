#!/usr/bin/env python3
"""
可视化验证集推理结果，并与真实标注进行对比
从split_info/val_files.txt中读取验证图片列表，对每张图片运行模型推理，
将推理结果用红色框画在原图上，然后从labels找到对应图片的标注，用绿色框画在原图上，
所有结果保存在results_val_images目录中。
"""

import os
import sys
import argparse
import numpy as np
from pathlib import Path
import torch
from torchvision import transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from PIL import Image
import cv2
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import matplotlib.pyplot as plt
from tqdm import tqdm

def load_trained_model(model_path, device='cuda'):
    """
    加载训练好的模型（从test_model.py复制）
    
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

def preprocess_image(image_path, target_size=None):
    """
    预处理图片（从test_model.py复制）
    
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
    使用模型检测物体（从test_model.py复制）
    
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

def parse_kitti_annotation(kitti_file_path):
    """
    解析KITTI格式的标注文件（从visualize_soda10m_annotations.py复制）
    
    Args:
        kitti_file_path: KITTI格式标注文件路径
        
    Returns:
        list: 标注对象列表，每个对象是一个字典
    """
    annotations = []
    
    if not os.path.exists(kitti_file_path):
        # print(f"警告: KITTI标注文件不存在: {kitti_file_path}")
        return annotations
    
    try:
        with open(kitti_file_path, 'r') as f:
            lines = f.readlines()
            
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            parts = line.split()
            if len(parts) < 15:
                print(f"警告: 第{line_num+1}行格式错误，跳过")
                continue
            
            # 解析KITTI格式的15个字段
            annotation = {
                'type': parts[0],  # 类别
                'truncated': float(parts[1]),  # 截断程度
                'occluded': int(parts[2]),  # 遮挡状态
                'alpha': float(parts[3]),  # 观察角度
                'bbox_left': float(parts[4]),  # 左上角x
                'bbox_top': float(parts[5]),  # 左上角y
                'bbox_right': float(parts[6]),  # 右下角x
                'bbox_bottom': float(parts[7]),  # 右下角y
                'dimensions_height': float(parts[8]),  # 3D高度
                'dimensions_width': float(parts[9]),  # 3D宽度
                'dimensions_length': float(parts[10]),  # 3D长度
                'location_x': float(parts[11]),  # 3D位置x
                'location_y': float(parts[12]),  # 3D位置y
                'location_z': float(parts[13]),  # 3D位置z
                'rotation_y': float(parts[14]),  # 旋转角度
            }
            
            # 如果有多余字段，可能是置信度分数
            if len(parts) > 15:
                annotation['score'] = float(parts[15])
            else:
                annotation['score'] = 1.0
            
            annotations.append(annotation)
            
        # print(f"从 {kitti_file_path} 读取了 {len(annotations)} 个标注")
        
    except Exception as e:
        print(f"解析KITTI标注文件失败: {e}")
        
    return annotations

def draw_boxes_on_image(image, pred_boxes, pred_labels, pred_scores, gt_boxes, gt_labels, classes):
    """
    在图片上绘制预测框（红色）和真实标注框（绿色）
    
    Args:
        image: PIL图片
        pred_boxes: 预测框数组 (N, 4)
        pred_labels: 预测标签数组 (N,)
        pred_scores: 预测置信度数组 (N,)
        gt_boxes: 真实标注框列表，每个元素是包含bbox_left/top/right/bottom的字典
        gt_labels: 真实标注类别列表（字符串）
        classes: 类别名称列表
        
    Returns:
        PIL图片: 绘制了框的图片
    """
    # 转换为numpy数组 (BGR格式)
    img_np = np.array(image)
    if len(img_np.shape) == 3 and img_np.shape[2] == 3:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    img_height, img_width = img_np.shape[:2]
    
    # 绘制真实标注框（绿色）
    for i, (gt_box, gt_label) in enumerate(zip(gt_boxes, gt_labels)):
        x1 = int(round(gt_box['bbox_left']))
        y1 = int(round(gt_box['bbox_top']))
        x2 = int(round(gt_box['bbox_right']))
        y2 = int(round(gt_box['bbox_bottom']))
        
        # 确保坐标在范围内
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))
        
        # 绘制绿色边界框
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # 准备标签文本
        label_text = f"GT: {gt_label}"
        
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
                     (0, 255, 0), -1)
        
        # 绘制文本
        cv2.putText(img_np, label_text,
                   (x1, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, (255, 255, 255), thickness)
    
    # 绘制预测框（红色）
    for i, (box, score, label) in enumerate(zip(pred_boxes, pred_scores, pred_labels)):
        # 获取类别名称
        if 0 < label <= len(classes):
            class_name = classes[label - 1]
        else:
            class_name = f"Class_{label}"
        
        # 确保坐标在范围内
        x1, y1, x2, y2 = [int(coord) for coord in box]
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))
        
        # 绘制红色边界框
        cv2.rectangle(img_np, (x1, y1), (x2, y2), (0, 0, 255), 2)
        
        # 准备标签文本
        label_text = f"Pred: {class_name}: {score:.2f}"
        
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
                     (0, 0, 255), -1)
        
        # 绘制文本
        cv2.putText(img_np, label_text,
                   (x1, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX,
                   font_scale, (255, 255, 255), thickness)
    
    # 转换回RGB格式
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    
    return Image.fromarray(img_rgb)

def process_image(model, image_path, label_dir, output_dir, classes, device, score_threshold=0.5):
    """
    处理单张图片：推理、加载标注、绘制并保存
    
    Args:
        model: 模型
        image_path: 图片路径
        label_dir: 标注目录
        output_dir: 输出目录
        classes: 类别列表
        device: 设备
        score_threshold: 置信度阈值
        
    Returns:
        dict: 处理结果统计
    """
    try:
        # 预处理图片
        original_image, image_tensor, original_size = preprocess_image(image_path)
        
        # 检测物体
        detections = detect_objects(model, image_tensor, device, score_threshold)
        pred_boxes = detections['boxes']
        pred_scores = detections['scores']
        pred_labels = detections['labels']
        
        # 获取对应的标注文件路径
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        label_path = os.path.join(label_dir, f"{base_name}.txt")
        
        # 解析KITTI标注
        gt_annotations = parse_kitti_annotation(label_path)
        gt_boxes = [ann for ann in gt_annotations]
        gt_labels = [ann['type'] for ann in gt_annotations]
        
        # 绘制框
        result_image = draw_boxes_on_image(
            original_image, pred_boxes, pred_labels, pred_scores, 
            gt_boxes, gt_labels, classes
        )
        
        # 保存图片
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"{base_name}_vis.jpg")
        result_image.save(output_path)
        
        return {
            'image_path': image_path,
            'output_path': output_path,
            'pred_count': len(pred_boxes),
            'gt_count': len(gt_boxes),
            'success': True
        }
        
    except Exception as e:
        print(f"处理图片 {image_path} 时出错: {e}")
        return {
            'image_path': image_path,
            'success': False,
            'error': str(e)
        }

def read_val_files(val_files_path):
    """
    读取验证图片列表文件
    
    Args:
        val_files_path: val_files.txt路径
        
    Returns:
        list: 图片路径列表
    """
    if not os.path.exists(val_files_path):
        raise FileNotFoundError(f"验证文件列表不存在: {val_files_path}")
    
    with open(val_files_path, 'r') as f:
        lines = f.readlines()
    
    # 去除空白字符，过滤空行
    image_paths = [line.strip() for line in lines if line.strip()]
    
    # 确保路径是相对于当前工作目录的绝对路径
    base_dir = os.path.dirname(os.path.dirname(val_files_path))  # split_info的父目录
    abs_paths = []
    for path in image_paths:
        if os.path.isabs(path):
            abs_paths.append(path)
        else:
            abs_paths.append(os.path.join(base_dir, path))
    
    return abs_paths

def main():
    parser = argparse.ArgumentParser(description='可视化验证集推理结果并与真实标注对比')
    parser.add_argument('--model', type=str, default='models_export/faster_rcnn_best.pth',
                       help='训练好的模型路径 (默认: models_export/faster_rcnn_best.pth)')
    parser.add_argument('--val_files', type=str, default='split_info/val_files.txt',
                       help='验证图片列表文件 (默认: split_info/val_files.txt)')
    parser.add_argument('--label_dir', type=str, default='labels',
                       help='标注文件目录 (默认: labels)')
    parser.add_argument('--output_dir', type=str, default='results_val_images',
                       help='结果输出目录 (默认: results_val_images)')
    parser.add_argument('--score_threshold', type=float, default=0.5,
                       help='置信度阈值 (默认: 0.5)')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                       help='运行设备 (cuda/cpu) (默认: 自动选择)')
    parser.add_argument('--max_images', type=int, default=None,
                       help='最大处理图片数量 (默认: None, 处理所有图片)')
    
    args = parser.parse_args()
    
    # 设置设备
    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    
    print("="*80)
    print("验证集推理结果可视化")
    print("="*80)
    print(f"模型文件: {args.model}")
    print(f"验证列表: {args.val_files}")
    print(f"标注目录: {args.label_dir}")
    print(f"输出目录: {args.output_dir}")
    print(f"置信度阈值: {args.score_threshold}")
    print(f"运行设备: {device}")
    print("="*80)
    
    # 检查路径
    if not os.path.exists(args.model):
        print(f"错误: 模型文件不存在: {args.model}")
        print(f"请确保模型文件存在，或使用 --model 参数指定正确的路径")
        return
    
    if not os.path.exists(args.val_files):
        print(f"错误: 验证列表文件不存在: {args.val_files}")
        print(f"请确保文件存在，或使用 --val_files 参数指定正确的路径")
        return
    
    if not os.path.exists(args.label_dir):
        print(f"错误: 标注目录不存在: {args.label_dir}")
        print(f"请确保目录存在，或使用 --label_dir 参数指定正确的路径")
        return
    
    # 加载模型
    try:
        model, classes, checkpoint = load_trained_model(args.model, device)
    except Exception as e:
        print(f"加载模型失败: {e}")
        return
    
    # 读取验证图片列表
    try:
        image_paths = read_val_files(args.val_files)
    except Exception as e:
        print(f"读取验证文件列表失败: {e}")
        return
    
    print(f"找到 {len(image_paths)} 张验证图片")
    
    # 限制处理数量
    if args.max_images and args.max_images < len(image_paths):
        image_paths = image_paths[:args.max_images]
        print(f"将处理前 {len(image_paths)} 张图片")
    
    # 处理每张图片
    results = []
    success_count = 0
    
    for image_path in tqdm(image_paths, desc="处理图片"):
        if not os.path.exists(image_path):
            print(f"警告: 图片不存在，跳过: {image_path}")
            continue
        
        result = process_image(
            model=model,
            image_path=image_path,
            label_dir=args.label_dir,
            output_dir=args.output_dir,
            classes=classes,
            device=device,
            score_threshold=args.score_threshold
        )
        
        if result['success']:
            success_count += 1
            # 打印进度信息
            tqdm.write(f"处理完成: {os.path.basename(image_path)} -> {result['pred_count']}预测/{result['gt_count']}标注")
        
        results.append(result)
    
    # 打印统计信息
    print("\n" + "="*80)
    print("处理完成!")
    print("="*80)
    print(f"总图片数: {len(results)}")
    print(f"成功处理: {success_count}")
    print(f"失败: {len(results) - success_count}")
    
    if success_count > 0:
        total_pred = sum(r.get('pred_count', 0) for r in results if r.get('success', False))
        total_gt = sum(r.get('gt_count', 0) for r in results if r.get('success', False))
        print(f"总预测框数: {total_pred}")
        print(f"总标注框数: {total_gt}")
        print(f"平均每张图片预测框数: {total_pred / success_count:.2f}")
        print(f"平均每张图片标注框数: {total_gt / success_count:.2f}")
        print(f"\n结果已保存到: {args.output_dir}")
    
    print("="*80)

if __name__ == '__main__':
    main()

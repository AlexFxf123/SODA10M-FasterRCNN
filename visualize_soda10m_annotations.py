#!/usr/bin/env python3
"""
SODA10M数据集标注可视化工具
从KITTI格式标注文件中读取边界框，并将其绘制在图片上
输入：图片路径和KITTI格式标注文件
输出：带标注框的可视化图片
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
import argparse
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib

# 设置中文字体支持
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

class SODA10MVisualizer:
    def __init__(self, category_colors=None):
        """
        初始化可视化器
        
        Args:
            category_colors: 类别颜色映射字典
        """
        # 默认类别颜色映射
        self.default_category_colors = {
            'Car': (255, 0, 0),        # 红色
            'Pedestrian': (0, 255, 0),  # 绿色
            'Cyclist': (0, 0, 255),     # 蓝色
            'Van': (255, 255, 0),       # 黄色
            'Truck': (255, 165, 0),     # 橙色
            'Bus': (128, 0, 128),       # 紫色
            'Tram': (0, 255, 255),      # 青色
            'Misc': (192, 192, 192),    # 银色
            'DontCare': (128, 128, 128) # 灰色
        }
        
        # 使用用户提供的颜色映射或默认映射
        self.category_colors = category_colors or self.default_category_colors
        
        # 类别显示名称映射
        self.category_names = {
            'Car': '汽车',
            'Pedestrian': '行人',
            'Cyclist': '骑车人',
            'Van': '厢式车',
            'Truck': '卡车',
            'Bus': '公交车',
            'Tram': '电车',
            'Misc': '其他',
            'DontCare': '忽略'
        }
    
    def parse_kitti_annotation(self, kitti_file_path):
        """
        解析KITTI格式的标注文件
        
        Args:
            kitti_file_path: KITTI格式标注文件路径
            
        Returns:
            list: 标注对象列表，每个对象是一个字典
        """
        annotations = []
        
        if not os.path.exists(kitti_file_path):
            print(f"警告: KITTI标注文件不存在: {kitti_file_path}")
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
                
            print(f"从 {kitti_file_path} 读取了 {len(annotations)} 个标注")
            
        except Exception as e:
            print(f"解析KITTI标注文件失败: {e}")
            
        return annotations
    
    def get_annotation_file_path(self, image_path, kitti_output_dir="kitti_output"):
        """
        根据图片路径获取对应的KITTI标注文件路径
        
        Args:
            image_path: 图片文件路径
            kitti_output_dir: KITTI格式标注输出目录
            
        Returns:
            str: 对应的KITTI标注文件路径
        """
        # 获取图片文件名（不含扩展名）
        image_name = os.path.basename(image_path)
        base_name = os.path.splitext(image_name)[0]
        
        # 构建KITTI标注文件路径
        kitti_annotation_path = os.path.join(kitti_output_dir, f"{base_name}.txt")
        
        return kitti_annotation_path
    
    def visualize_annotations_cv2(self, image_path, annotations, output_path, 
                                 show_labels=True, show_scores=True, line_width=2):
        """
        使用OpenCV在图片上可视化标注框
        
        Args:
            image_path: 图片文件路径
            annotations: 标注对象列表
            output_path: 输出图片路径
            show_labels: 是否显示类别标签
            show_scores: 是否显示置信度分数
            line_width: 边界框线宽
        """
        # 读取图片
        if not os.path.exists(image_path):
            print(f"错误: 图片文件不存在: {image_path}")
            return False
        
        # 使用OpenCV读取图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"错误: 无法读取图片: {image_path}")
            return False
        
        # 创建图片副本用于绘制
        img_with_boxes = image.copy()
        
        # 获取图片尺寸
        img_height, img_width = image.shape[:2]
        
        # 绘制每个标注框
        for i, ann in enumerate(annotations):
            # 获取边界框坐标
            x1 = int(round(ann['bbox_left']))
            y1 = int(round(ann['bbox_top']))
            x2 = int(round(ann['bbox_right']))
            y2 = int(round(ann['bbox_bottom']))
            
            # 确保坐标在图片范围内
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            x2 = max(0, min(x2, img_width - 1))
            y2 = max(0, min(y2, img_height - 1))
            
            # 获取类别和颜色
            obj_type = ann['type']
            color = self.category_colors.get(obj_type, (255, 255, 255))  # 默认白色
            
            # 绘制边界框
            cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, line_width)
            
            # 准备标签文本
            label = self.category_names.get(obj_type, obj_type)
            
            # 添加置信度分数（如果有）
            if show_scores and 'score' in ann and ann['score'] != 1.0:
                label += f" {ann['score']:.2f}"
            
            # 添加截断和遮挡信息
            if ann['truncated'] > 0 or ann['occluded'] > 0:
                label += f" (T:{ann['truncated']:.1f},O:{ann['occluded']})"
            
            if show_labels:
                # 计算文本位置
                text_y = y1 - 5 if y1 > 20 else y1 + 20
                
                # 获取文本尺寸
                font_scale = 0.5
                thickness = 1
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                
                # 绘制文本背景
                cv2.rectangle(img_with_boxes, 
                             (x1, text_y - text_height - 5), 
                             (x1 + text_width, text_y + 5), 
                             color, -1)
                
                # 绘制文本
                cv2.putText(img_with_boxes, label, 
                           (x1, text_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, (255, 255, 255), thickness)
        
        # 保存图片
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        success = cv2.imwrite(output_path, img_with_boxes)
        if success:
            print(f"图片已保存到: {output_path}")
        else:
            print(f"保存图片失败: {output_path}")
        
        return success
    
    def visualize_annotations_matplotlib(self, image_path, annotations, output_path, 
                                        show_labels=True, show_scores=True, figsize=(15, 10)):
        """
        使用matplotlib在图片上可视化标注框
        
        Args:
            image_path: 图片文件路径
            annotations: 标注对象列表
            output_path: 输出图片路径
            show_labels: 是否显示类别标签
            show_scores: 是否显示置信度分数
            figsize: 图形尺寸
        """
        # 读取图片
        if not os.path.exists(image_path):
            print(f"错误: 图片文件不存在: {image_path}")
            return False
        
        # 使用matplotlib读取图片
        image = plt.imread(image_path)
        
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)
        ax.imshow(image)
        
        # 获取图片尺寸
        img_height, img_width = image.shape[:2]
        
        # 绘制每个标注框
        for i, ann in enumerate(annotations):
            # 获取边界框坐标
            x1 = ann['bbox_left']
            y1 = ann['bbox_top']
            x2 = ann['bbox_right']
            y2 = ann['bbox_bottom']
            width = x2 - x1
            height = y2 - y1
            
            # 确保坐标在图片范围内
            x1 = max(0, min(x1, img_width - 1))
            y1 = max(0, min(y1, img_height - 1))
            width = max(1, min(width, img_width - x1))
            height = max(1, min(height, img_height - y1))
            
            # 获取类别和颜色
            obj_type = ann['type']
            color = np.array(self.category_colors.get(obj_type, (1.0, 1.0, 1.0))) / 255.0
            
            # 创建矩形边界框
            rect = Rectangle((x1, y1), width, height, 
                            linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            
            # 准备标签文本
            label = self.category_names.get(obj_type, obj_type)
            
            # 添加置信度分数（如果有）
            if show_scores and 'score' in ann and ann['score'] != 1.0:
                label += f"\n{ann['score']:.2f}"
            
            # 添加截断和遮挡信息
            if ann['truncated'] > 0 or ann['occluded'] > 0:
                label += f"\nT:{ann['truncated']:.1f},O:{ann['occluded']}"
            
            if show_labels:
                # 添加文本标签
                text_x = x1
                text_y = y1 - 5 if y1 > 20 else y1 + height + 10
                
                # 绘制文本背景
                ax.text(text_x, text_y, label, 
                       fontsize=10, color='white',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=color, edgecolor=color, alpha=0.7))
        
        # 隐藏坐标轴
        ax.axis('off')
        
        # 添加标题
        image_name = os.path.basename(image_path)
        ax.set_title(f'SODA10M_Visualize - {image_name}', fontsize=16)
        
        # 创建图例
        unique_types = set([ann['type'] for ann in annotations])
        legend_elements = []
        for obj_type in unique_types:
            color = np.array(self.category_colors.get(obj_type, (1.0, 1.0, 1.0))) / 255.0
            label = self.category_names.get(obj_type, obj_type)
            legend_elements.append(Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color, label=label))
        
        if legend_elements:
            ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        print(f"图片已保存到: {output_path}")
        return True
    
    def visualize_single_image(self, image_path, kitti_output_dir="kitti_output", 
                              output_dir="test_kitti_anno", use_matplotlib=True):
        """
        可视化单张图片的标注框
        
        Args:
            image_path: 图片文件路径
            kitti_output_dir: KITTI格式标注目录
            output_dir: 输出图片目录
            use_matplotlib: 是否使用matplotlib（True）或OpenCV（False）
            
        Returns:
            bool: 可视化是否成功
        """
        print(f"开始处理图片: {image_path}")
        
        # 检查图片文件是否存在
        if not os.path.exists(image_path):
            print(f"错误: 图片文件不存在: {image_path}")
            return False
        
        # 获取对应的KITTI标注文件路径
        annotation_path = self.get_annotation_file_path(image_path, kitti_output_dir)
        
        # 解析KITTI标注文件
        annotations = self.parse_kitti_annotation(annotation_path)
        
        if not annotations:
            print(f"警告: 没有找到标注信息，可能原因:")
            print(f"  1. 标注文件不存在: {annotation_path}")
            print(f"  2. 标注文件格式错误")
            print(f"  3. 该图片没有标注")
            
            # 创建测试标注（用于演示）
            print("创建测试标注用于演示...")
            annotations = self.create_test_annotations(image_path)
        
        # 创建输出文件路径
        image_name = os.path.basename(image_path)
        base_name, ext = os.path.splitext(image_name)
        
        if use_matplotlib:
            output_filename = f"{base_name}_matplotlib{ext}"
        else:
            output_filename = f"{base_name}_opencv{ext}"
        
        output_path = os.path.join(output_dir, output_filename)
        
        # 可视化标注框
        if use_matplotlib:
            success = self.visualize_annotations_matplotlib(
                image_path, annotations, output_path, 
                show_labels=True, show_scores=True)
        else:
            success = self.visualize_annotations_cv2(
                image_path, annotations, output_path, 
                show_labels=True, show_scores=True, line_width=2)
        
        if success:
            print(f"成功可视化图片: {image_path}")
            print(f"标注数量: {len(annotations)}")
            print(f"输出文件: {output_path}")
            
            # 打印标注统计信息
            self.print_annotation_stats(annotations)
        
        return success
    
    def create_test_annotations(self, image_path):
        """
        创建测试标注（当没有找到真实标注时使用）
        
        Args:
            image_path: 图片文件路径
            
        Returns:
            list: 测试标注列表
        """
        # 使用PIL获取图片尺寸
        try:
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except:
            img_width, img_height = 1920, 1080
        
        # 创建一些测试标注
        test_annotations = [
            {
                'type': 'Car',
                'truncated': 0.0,
                'occluded': 0,
                'alpha': -1.0,
                'bbox_left': img_width * 0.3,
                'bbox_top': img_height * 0.4,
                'bbox_right': img_width * 0.5,
                'bbox_bottom': img_height * 0.6,
                'dimensions_height': -1.0,
                'dimensions_width': -1.0,
                'dimensions_length': -1.0,
                'location_x': -1000.0,
                'location_y': -1000.0,
                'location_z': -1000.0,
                'rotation_y': -10.0,
                'score': 0.95
            },
            {
                'type': 'Pedestrian',
                'truncated': 0.2,
                'occluded': 1,
                'alpha': -1.0,
                'bbox_left': img_width * 0.6,
                'bbox_top': img_height * 0.5,
                'bbox_right': img_width * 0.7,
                'bbox_bottom': img_height * 0.8,
                'dimensions_height': -1.0,
                'dimensions_width': -1.0,
                'dimensions_length': -1.0,
                'location_x': -1000.0,
                'location_y': -1000.0,
                'location_z': -1000.0,
                'rotation_y': -10.0,
                'score': 0.88
            }
        ]
        
        print(f"创建了 {len(test_annotations)} 个测试标注")
        return test_annotations
    
    def print_annotation_stats(self, annotations):
        """
        打印标注统计信息
        
        Args:
            annotations: 标注对象列表
        """
        if not annotations:
            print("没有标注信息")
            return
        
        # 统计各类别的数量
        category_counts = {}
        for ann in annotations:
            obj_type = ann['type']
            category_counts[obj_type] = category_counts.get(obj_type, 0) + 1
        
        print("\n标注统计信息:")
        print("-" * 40)
        for obj_type, count in category_counts.items():
            display_name = self.category_names.get(obj_type, obj_type)
            print(f"  {display_name}: {count} 个")
        
        # 计算平均截断程度
        if annotations:
            avg_truncated = sum(ann['truncated'] for ann in annotations) / len(annotations)
            print(f"  平均截断程度: {avg_truncated:.2f}")
            
            # 统计遮挡情况
            occlusion_counts = {0: 0, 1: 0, 2: 0, 3: 0}
            for ann in annotations:
                occluded = ann['occluded']
                if occluded in occlusion_counts:
                    occlusion_counts[occluded] += 1
            
            print(f"  遮挡情况:")
            print(f"    完全可见: {occlusion_counts[0]} 个")
            print(f"    部分遮挡: {occlusion_counts[1]} 个")
            print(f"    严重遮挡: {occlusion_counts[2]} 个")
            print(f"    未知遮挡: {occlusion_counts[3]} 个")
        print("-" * 40)


def main():
    """
    主函数：可视化SODA10M数据集中的图片标注
    """
    parser = argparse.ArgumentParser(description='可视化SODA10M数据集的标注框')
    parser.add_argument('--image_path', type=str, 
                       default="labeled_trainval/SSLAD-2D/labeled/train/HT_TRAIN_000001_SH_000.jpg",
                       help='图片文件路径')
    parser.add_argument('--kitti_output_dir', type=str, default="kitti_output",
                       help='KITTI格式标注目录')
    parser.add_argument('--output_dir', type=str, default="test_kitti_anno",
                       help='输出图片目录')
    parser.add_argument('--use_matplotlib', action='store_true', default=True,
                       help='使用matplotlib进行可视化（默认）')
    parser.add_argument('--use_opencv', action='store_true',
                       help='使用OpenCV进行可视化')
    parser.add_argument('--custom_colors', type=str, default=None,
                       help='自定义颜色映射JSON文件路径')
    
    args = parser.parse_args()
    
    # 确定使用哪种可视化方法
    use_matplotlib = args.use_matplotlib
    if args.use_opencv:
        use_matplotlib = False
    
    # 加载自定义颜色映射（如果提供）
    category_colors = None
    if args.custom_colors:
        try:
            with open(args.custom_colors, 'r') as f:
                category_colors = json.load(f)
            print(f"已加载自定义颜色映射: {args.custom_colors}")
        except Exception as e:
            print(f"加载自定义颜色映射失败: {e}")
    
    # 创建可视化器
    visualizer = SODA10MVisualizer(category_colors)
    
    # 可视化单张图片
    print(f"{'='*60}")
    print("SODA10M数据集标注可视化工具")
    print(f"{'='*60}")
    
    success = visualizer.visualize_single_image(
        args.image_path, 
        args.kitti_output_dir, 
        args.output_dir, 
        use_matplotlib
    )
    
    if success:
        print(f"\n{'='*60}")
        print("可视化完成!")
        print(f"图片: {args.image_path}")
        print(f"输出目录: {args.output_dir}")
        print(f"可视化方法: {'matplotlib' if use_matplotlib else 'OpenCV'}")
        print(f"{'='*60}")
    else:
        print(f"\n{'='*60}")
        print("可视化失败!")
        print(f"请检查:")
        print(f"  1. 图片文件是否存在: {args.image_path}")
        print(f"  2. KITTI标注目录是否存在: {args.kitti_output_dir}")
        print(f"  3. 是否有对应的KITTI标注文件")
        print(f"{'='*60}")
    
    return success


if __name__ == '__main__':
    # 直接运行示例：可视化指定图片
    print("SODA10M数据集标注可视化工具")
    print("=" * 60)
    
    # 检查必要的库
    try:
        import cv2
        print("✓ OpenCV库已安装")
    except ImportError:
        print("⚠ OpenCV库未安装，将使用matplotlib进行可视化")
        print("  安装命令: pip install opencv-python")
    
    try:
        import matplotlib
        print("✓ matplotlib库已安装")
    except ImportError:
        print("⚠ matplotlib库未安装")
        print("  安装命令: pip install matplotlib")
    
    try:
        from PIL import Image
        print("✓ PIL(Pillow)库已安装")
    except ImportError:
        print("⚠ PIL(Pillow)库未安装")
        print("  安装命令: pip install pillow")
    
    print("=" * 60)
    
    # 设置具体路径
    # image_path = "labeled_trainval/SSLAD-2D/labeled/train/HT_TRAIN_000003_SH_000.jpg"
    image_path = "labeled_trainval/SSLAD-2D/labeled/val/HT_VAL_000002_SH_001.jpg"
    kitti_output_dir = "kitti_output_val"
    output_dir = "test_kitti_anno_anno"
    
    print(f"图片路径: {image_path}")
    print(f"KITTI标注目录: {kitti_output_dir}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    # 创建可视化器
    visualizer = SODA10MVisualizer()
    
    # 可视化图片
    print("开始可视化图片...")
    success = visualizer.visualize_single_image(
        image_path, 
        kitti_output_dir, 
        output_dir, 
        use_matplotlib=True
    )
    
    if success:
        print("\n可视化成功完成!")
        print("您可以:")
        print("  1. 查看输出目录中的可视化图片")
        print("  2. 修改参数可视化其他图片")
        print("  3. 运行 python visualize_soda10m_annotations.py --help 查看更多选项")
    else:
        print("\n可视化失败!")
        print("请确保:")
        print("  1. 图片文件存在")
        print("  2. 已运行之前的转换脚本生成KITTI格式标注")
        print("  3. 必要的Python库已安装")
    
    print("=" * 60)
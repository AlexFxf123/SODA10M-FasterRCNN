#!/usr/bin/env python3
"""
SODA10M数据集标注格式转换工具 - 针对具体路径优化版
将SODA10M的COCO格式标注转换为KITTI格式
具体处理：labeled_trainval/SSLAD-2D/labeled/annotations/instance_val.json
输出到：kitti_output
"""

import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime
import sys

class SODA10MToKITTIConverter:
    def __init__(self, category_mapping=None):
        """
        初始化转换器
        
        Args:
            category_mapping: 可选的类别映射字典
        """
        # SODA10M类别到KITTI类别的默认映射
        self.default_category_mapping = {
            # 车辆相关
            'car': 'Car',
            'truck': 'Truck',
            'bus': 'Bus',
            'van': 'Van',
            'trailer': 'Trailer',
            'construction vehicle': 'Misc',
            
            # 行人相关
            'pedestrian': 'Pedestrian',
            'person': 'Pedestrian',
            'rider': 'Cyclist',
            'rider-bicyclist': 'Cyclist',
            'rider-motorcyclist': 'Cyclist',
            
            # 两轮车相关
            'bicycle': 'Cyclist',
            'motorcycle': 'Cyclist',
            'bike': 'Cyclist',
            
            # 交通设施
            'traffic cone': 'Misc',
            'traffic light': 'Misc',
            'traffic sign': 'Misc',
            'barrier': 'Misc',
            
            # 其他
            'other': 'DontCare',
            'ignore': 'DontCare',
            'unknown': 'DontCare'
        }
        
        # 使用用户提供的映射或默认映射
        self.category_mapping = category_mapping or self.default_category_mapping
        
        # KITTI格式字段
        self.kitti_fields = [
            'type', 'truncated', 'occluded', 'alpha', 
            'bbox_left', 'bbox_top', 'bbox_right', 'bbox_bottom',
            'dimensions_height', 'dimensions_width', 'dimensions_length',
            'location_x', 'location_y', 'location_z', 'rotation_y',
            'score'
        ]
    
    def convert_bbox_coco_to_kitti(self, bbox_coco, img_width, img_height):
        """
        将COCO格式的bbox[x, y, width, height]转换为KITTI格式[x1, y1, x2, y2]
        
        Args:
            bbox_coco: [x, y, width, height]
            img_width: 图像宽度
            img_height: 图像高度
            
        Returns:
            bbox_kitti: [x1, y1, x2, y2]
        """
        x, y, w, h = bbox_coco
        x1 = float(x)
        y1 = float(y)
        x2 = float(x + w)
        y2 = float(y + h)
        
        # 确保边界框在图像范围内
        x1 = max(0, min(x1, img_width - 1))
        y1 = max(0, min(y1, img_height - 1))
        x2 = max(0, min(x2, img_width - 1))
        y2 = max(0, min(y2, img_height - 1))
        
        return [x1, y1, x2, y2]
    
    def calculate_truncation(self, bbox, img_width, img_height):
        """
        计算截断程度
        
        Args:
            bbox: KITTI格式边界框[x1, y1, x2, y2]
            img_width: 图像宽度
            img_height: 图像高度
            
        Returns:
            truncation: 截断程度(0-1)
        """
        x1, y1, x2, y2 = bbox
        bbox_area = (x2 - x1) * (y2 - y1)
        
        # 计算边界框与图像边界的重叠
        img_bbox = [0, 0, img_width, img_height]
        
        # 计算交集
        inter_x1 = max(x1, img_bbox[0])
        inter_y1 = max(y1, img_bbox[1])
        inter_x2 = min(x2, img_bbox[2])
        inter_y2 = min(y2, img_bbox[3])
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 1.0  # 完全在图像外
        
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        
        if bbox_area == 0:
            return 1.0
        
        truncation = 1.0 - (inter_area / bbox_area)
        return min(1.0, max(0.0, truncation))
    
    def parse_coco_annotation(self, coco_data, img_id, category_mapping=None):
        """
        解析COCO格式的标注
        
        Args:
            coco_data: COCO格式数据字典
            img_id: 图像ID
            category_mapping: 类别映射
            
        Returns:
            annotations: 该图像的所有标注列表
            img_info: 图像信息字典
        """
        if category_mapping is None:
            category_mapping = self.category_mapping
        
        # 获取图像信息
        img_info = None
        for img in coco_data.get('images', []):
            if img.get('id') == img_id:
                img_info = img
                break
        
        if img_info is None:
            return [], None
        
        img_width = img_info.get('width', 1920)
        img_height = img_info.get('height', 1080)
        img_filename = img_info.get('file_name', '')
        
        # 获取该图像的所有标注
        img_annotations = []
        for ann in coco_data.get('annotations', []):
            if ann.get('image_id') == img_id:
                # 获取类别信息
                category_id = ann.get('category_id', 0)
                category_name = None
                
                # 查找类别名称
                for cat in coco_data.get('categories', []):
                    if cat.get('id') == category_id:
                        category_name = cat.get('name', '').lower()
                        break
                
                if category_name is None:
                    continue
                
                # 映射到KITTI类别
                kitti_category = category_mapping.get(category_name, 'DontCare')
                
                # 获取边界框
                bbox_coco = ann.get('bbox', [0, 0, 0, 0])
                bbox_kitti = self.convert_bbox_coco_to_kitti(bbox_coco, img_width, img_height)
                
                # 计算截断程度
                truncated = self.calculate_truncation(bbox_kitti, img_width, img_height)
                
                # 获取遮挡状态（SODA10M可能没有，用默认值）
                occluded = ann.get('occluded', 0)
                if occluded not in [0, 1, 2, 3]:
                    occluded = 0
                
                # 获取分割信息（如果有）
                segmentation = ann.get('segmentation', None)
                
                # 获取面积
                area = ann.get('area', 0.0)
                
                # 创建KITTI格式标注
                kitti_ann = {
                    'type': kitti_category,
                    'truncated': truncated,
                    'occluded': occluded,
                    'alpha': -10.0,  # 默认值，SODA10M通常没有alpha角度
                    'bbox_left': bbox_kitti[0],
                    'bbox_top': bbox_kitti[1],
                    'bbox_right': bbox_kitti[2],
                    'bbox_bottom': bbox_kitti[3],
                    'dimensions_height': -1.0,  # 2D检测没有3D尺寸
                    'dimensions_width': -1.0,
                    'dimensions_length': -1.0,
                    'location_x': -1000.0,  # 2D检测没有3D位置
                    'location_y': -1000.0,
                    'location_z': -1000.0,
                    'rotation_y': -10.0,  # 默认值
                    'score': 1.0,  # 默认置信度
                    'original_category': category_name,  # 保存原始类别
                    'area': area,  # 保存原始面积
                    'segmentation': segmentation  # 保存分割信息
                }
                
                img_annotations.append(kitti_ann)
        
        return img_annotations, img_info
    
    def convert_single_image(self, coco_data, img_id, output_dir, category_mapping=None):
        """
        转换单个图像的标注
        
        Args:
            coco_data: COCO格式数据
            img_id: 图像ID
            output_dir: 输出目录
            category_mapping: 类别映射
            
        Returns:
            bool: 转换是否成功
        """
        annotations, img_info = self.parse_coco_annotation(coco_data, img_id, category_mapping)
        
        if not annotations or img_info is None:
            return False
        
        # 获取图像文件名（不含扩展名）
        img_filename = img_info.get('file_name', f'{img_id:06d}.jpg')
        base_name = os.path.splitext(os.path.basename(img_filename))[0]
        
        # 创建输出文件路径
        output_path = os.path.join(output_dir, f'{base_name}.txt')
        
        # 写入KITTI格式
        with open(output_path, 'w') as f:
            for ann in annotations:
                # 格式化每一行（标准的KITTI 15字段格式）
                line = f"{ann['type']} {ann['truncated']:.2f} {ann['occluded']} {ann['alpha']:.2f} "
                line += f"{ann['bbox_left']:.2f} {ann['bbox_top']:.2f} {ann['bbox_right']:.2f} {ann['bbox_bottom']:.2f} "
                line += f"{ann['dimensions_height']:.2f} {ann['dimensions_width']:.2f} {ann['dimensions_length']:.2f} "
                line += f"{ann['location_x']:.2f} {ann['location_y']:.2f} {ann['location_z']:.2f} "
                line += f"{ann['rotation_y']:.2f} {ann['score']:.2f}\n"
                f.write(line)
        
        return True
    
    def convert_dataset(self, coco_json_path, output_dir, split='train'):
        """
        转换整个数据集
        
        Args:
            coco_json_path: COCO格式JSON文件路径
            output_dir: 输出目录
            split: 数据集分割（train/val/test）
            
        Returns:
            dict: 转换统计信息
        """
        # 检查输入文件是否存在
        if not os.path.exists(coco_json_path):
            print(f"错误: 输入文件不存在: {coco_json_path}")
            return None
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取COCO JSON文件
        print(f"读取COCO标注文件: {coco_json_path}")
        try:
            with open(coco_json_path, 'r') as f:
                coco_data = json.load(f)
        except Exception as e:
            print(f"读取JSON文件失败: {e}")
            return None
        
        # 获取数据集信息
        dataset_info = coco_data.get('info', {})
        print(f"数据集: {dataset_info.get('description', '未知数据集')}")
        print(f"版本: {dataset_info.get('version', '未知版本')}")
        print(f"年份: {dataset_info.get('year', '未知年份')}")
        
        # 获取所有图像ID
        image_ids = []
        for img in coco_data.get('images', []):
            img_id = img.get('id')
            if img_id is not None:
                image_ids.append(img_id)
        
        print(f"找到 {len(image_ids)} 张图像")
        
        # 统计类别信息
        category_stats = {}
        for cat in coco_data.get('categories', []):
            cat_id = cat.get('id')
            cat_name = cat.get('name', 'unknown')
            category_stats[cat_id] = {'name': cat_name, 'count': 0}
        
        # 转换每个图像
        successful = 0
        failed = 0
        total_annotations = 0
        
        print(f"开始转换{len(image_ids)}张图像的标注...")
        for img_id in tqdm(image_ids, desc=f"转换{split}集标注"):
            try:
                annotations, img_info = self.parse_coco_annotation(coco_data, img_id)
                total_annotations += len(annotations)
                
                if self.convert_single_image(coco_data, img_id, output_dir):
                    successful += 1
                else:
                    failed += 1
                    
                # 更新类别统计
                for ann in annotations:
                    original_cat = ann.get('original_category', 'unknown')
                    for cat_id, cat_info in category_stats.items():
                        if cat_info['name'].lower() == original_cat:
                            category_stats[cat_id]['count'] += 1
                            break
            except Exception as e:
                print(f"\n转换图像 {img_id} 时出错: {e}")
                failed += 1
        
        # 保存类别映射信息
        mapping_file = os.path.join(output_dir, 'category_mapping.json')
        with open(mapping_file, 'w') as f:
            json.dump(self.category_mapping, f, indent=2, ensure_ascii=False)
        
        # 保存类别统计信息
        category_stats_list = []
        for cat_id, cat_info in category_stats.items():
            category_stats_list.append({
                'id': cat_id,
                'name': cat_info['name'],
                'count': cat_info['count']
            })
        
        stats_file = os.path.join(output_dir, 'category_stats.json')
        with open(stats_file, 'w') as f:
            json.dump(category_stats_list, f, indent=2, ensure_ascii=False)
        
        # 保存转换统计信息
        conversion_stats = {
            'dataset': 'SODA10M',
            'split': split,
            'source_file': coco_json_path,
            'output_dir': output_dir,
            'total_images': len(image_ids),
            'successful_conversions': successful,
            'failed_conversions': failed,
            'total_annotations': total_annotations,
            'conversion_date': datetime.now().isoformat(),
            'category_mapping_file': mapping_file,
            'category_stats_file': stats_file
        }
        
        stats_file_path = os.path.join(output_dir, 'conversion_stats.json')
        with open(stats_file_path, 'w') as f:
            json.dump(conversion_stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n{'='*60}")
        print(f"转换完成!")
        print(f"{'='*60}")
        print(f"输入文件: {coco_json_path}")
        print(f"输出目录: {output_dir}")
        print(f"数据集分割: {split}")
        print(f"总图像数: {len(image_ids)}")
        print(f"成功转换: {successful}")
        print(f"失败转换: {failed}")
        print(f"总标注数: {total_annotations}")
        print(f"类别映射文件: {mapping_file}")
        print(f"类别统计文件: {stats_file}")
        print(f"转换统计文件: {stats_file_path}")
        print(f"{'='*60}")
        
        # 打印类别统计
        print("\n类别统计:")
        for cat_stat in category_stats_list:
            if cat_stat['count'] > 0:
                print(f"  {cat_stat['name']} (ID: {cat_stat['id']}): {cat_stat['count']} 个标注")
        
        return conversion_stats


def convert_specific_soda10m_dataset():
    """
    转换特定的SODA10M数据集
    输入: labeled_trainval/SSLAD-2D/labeled/annotations/instance_val.json
    输出: kitti_output
    """
    # 设置具体路径
    input_path = "labeled_trainval/SSLAD-2D/labeled/annotations/instance_val.json"
    output_path = "kitti_output"
    
    print(f"开始转换SODA10M数据集...")
    print(f"输入文件: {input_path}")
    print(f"输出目录: {output_path}")
    print(f"{'='*60}")
    
    # 检查输入文件是否存在
    if not os.path.exists(input_path):
        print(f"错误: 输入文件不存在: {input_path}")
        print("请确保路径正确: labeled_trainval/SSLAD-2D/labeled/annotations/instance_val.json")
        return False
    
    # 创建转换器
    converter = SODA10MToKITTIConverter()
    
    # 执行转换
    try:
        stats = converter.convert_dataset(input_path, output_path, split='train')
        if stats:
            print(f"\n转换成功完成!")
            print(f"KITTI格式标注文件已保存在: {output_path}/")
            print(f"每个.txt文件对应一个图像的KITTI格式标注")
            return True
        else:
            print(f"\n转换失败!")
            return False
    except Exception as e:
        print(f"\n转换过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """
    主函数：支持两种使用方式
    1. 无参数运行：使用默认路径转换
    2. 带参数运行：自定义输入输出路径
    """
    parser = argparse.ArgumentParser(description='将SODA10M的COCO格式标注转换为KITTI格式')
    parser.add_argument('--coco_json', type=str, default=None,
                       help='COCO格式的JSON标注文件路径 (默认: labeled_trainval/SSLAD-2D/labeled/annotations/instance_val.json)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出KITTI格式标注的目录 (默认: kitti_output)')
    parser.add_argument('--split', type=str, default='train',
                       choices=['train', 'val', 'test'],
                       help='数据集分割类型 (train/val/test)')
    parser.add_argument('--custom_mapping', type=str, default=None,
                       help='自定义类别映射JSON文件路径')
    
    args = parser.parse_args()
    
    # 如果没有提供参数，使用默认路径
    if args.coco_json is None and args.output_dir is None:
        print("使用默认路径进行转换...")
        print("输入: labeled_trainval/SSLAD-2D/labeled/annotations/instance_val.json")
        print("输出: kitti_output")
        print()
        success = convert_specific_soda10m_dataset()
        if not success:
            print("\n使用默认路径转换失败，请检查文件路径是否正确。")
        return
    
    # 如果提供了参数，使用参数指定的路径
    input_path = args.coco_json or "labeled_trainval/SSLAD-2D/labeled/annotations/instance_val.json"
    output_dir = args.output_dir or "kitti_output"
    
    # 加载自定义类别映射（如果提供）
    category_mapping = None
    if args.custom_mapping:
        try:
            with open(args.custom_mapping, 'r') as f:
                category_mapping = json.load(f)
            print(f"已加载自定义类别映射: {args.custom_mapping}")
        except Exception as e:
            print(f"加载自定义类别映射失败: {e}")
            return
    
    # 创建转换器
    converter = SODA10MToKITTIConverter(category_mapping)
    
    # 执行转换
    print(f"开始转换...")
    print(f"输入文件: {input_path}")
    print(f"输出目录: {output_dir}")
    print(f"数据集分割: {args.split}")
    
    stats = converter.convert_dataset(input_path, output_dir, args.split)
    
    if stats:
        print(f"\nKITTI格式文件已生成在: {output_dir}")
        print("每个图像对应一个.txt文件，格式为:")
        print("类别 截断程度 遮挡状态 alpha bbox_left bbox_top bbox_right bbox_bottom")
        print("高度 宽度 长度 位置x 位置y 位置z 旋转y 置信度")


if __name__ == '__main__':
    # 测试转换器的基本功能
    print("SODA10M到KITTI格式转换器")
    print("=" * 60)
    
    # 检查是否安装了必要的库
    try:
        from tqdm import tqdm
        print("✓ tqdm库已安装")
    except ImportError:
        print("⚠ tqdm库未安装，安装命令: pip install tqdm")
        print("正在使用简单进度显示...")
        # 简单的进度显示替代函数
        def tqdm(iterable, **kwargs):
            return iterable
    
    # 运行主函数
    main()

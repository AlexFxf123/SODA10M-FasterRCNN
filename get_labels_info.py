#!/usr/bin/env python3
"""
KITTI格式标注数据统计脚本
统计每个类别的目标数量、包含该类别的图片数量
支持自定义类别过滤和多种输出格式
增加标注框面积信息统计，包括每个类别的最大最小面积
"""

import os
import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

def parse_kitti_annotation_with_area(annotation_path, valid_classes=None, skip_classes=None):
    """
    解析KITTI格式的标注文件，同时提取边界框面积信息
    
    Args:
        annotation_path: 标注文件路径
        valid_classes: 有效的类别列表，如果为None则统计所有类别
        skip_classes: 需要跳过的类别列表
        
    Returns:
        list: 包含(类别, 面积)的元组列表
    """
    annotations = []
    
    if not os.path.exists(annotation_path):
        return annotations
    
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
            
            # 第一个字段是类别
            obj_type = parts[0]
            
            # 跳过指定的类别
            if skip_classes and obj_type in skip_classes:
                continue
            
            # 如果指定了有效类别，只统计有效类别
            if valid_classes and obj_type not in valid_classes:
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
            
            # 计算边界框面积
            area = (x2 - x1) * (y2 - y1)
            
            annotations.append((obj_type, area))
    
    except Exception as e:
        print(f"解析标注文件 {annotation_path} 失败: {e}")
    
    return annotations

def collect_statistics(labels_dir, valid_classes=None, skip_classes=None, 
                      kitti_default_skip=False, verbose=False):
    """
    收集统计信息
    
    Args:
        labels_dir: 标注文件目录
        valid_classes: 有效的类别列表
        skip_classes: 需要跳过的类别列表
        kitti_default_skip: 是否跳过KITTI默认的DontCare类别
        verbose: 是否显示详细信息
        
    Returns:
        dict: 统计结果字典
    """
    # 处理类别过滤
    if skip_classes is None:
        skip_classes = []
    
    if kitti_default_skip:
        skip_classes.append('DontCare')
    
    # 获取所有标注文件
    annotation_files = list(Path(labels_dir).glob('*.txt'))
    
    if not annotation_files:
        print(f"错误: 在 {labels_dir} 中没有找到标注文件")
        return None
    
    print(f"找到 {len(annotation_files)} 个标注文件")
    
    # 初始化统计变量
    class_object_count = defaultdict(int)  # 各类别目标总数
    class_image_count = defaultdict(int)   # 包含各类别的图片数量
    class_area_sum = defaultdict(float)    # 各类别总面积
    class_area_list = defaultdict(list)    # 各类别所有面积列表
    class_min_area = defaultdict(lambda: float('inf'))  # 各类别最小面积
    class_max_area = defaultdict(float)    # 各类别最大面积
    image_counts = []  # 每张图片的目标数量
    all_areas = []     # 所有边界框面积
    all_categories = set()  # 所有出现的类别
    
    # 统计信息
    empty_files = 0
    total_objects = 0
    total_area = 0.0
    
    # 遍历所有标注文件
    for anno_file in tqdm(annotation_files, desc="Processing annotation files"):
        # 解析标注文件
        annotations = parse_kitti_annotation_with_area(str(anno_file), valid_classes, skip_classes)
        
        # 统计
        if not annotations:
            empty_files += 1
            image_counts.append(0)
            continue
        
        # 提取类别和面积
        categories = [ann[0] for ann in annotations]
        areas = [ann[1] for ann in annotations]
        
        # 更新各类别目标数量
        category_counter = Counter(categories)
        for category, count in category_counter.items():
            class_object_count[category] += count
            all_categories.add(category)
        
        # 更新包含各类别的图片数量
        unique_categories = set(categories)
        for category in unique_categories:
            class_image_count[category] += 1
        
        # 更新各类别面积信息
        for category, area in zip(categories, areas):
            class_area_sum[category] += area
            class_area_list[category].append(area)
            
            # 更新最小面积
            if area < class_min_area[category]:
                class_min_area[category] = area
            
            # 更新最大面积
            if area > class_max_area[category]:
                class_max_area[category] = area
            
            total_area += area
        
        # 统计每张图片的目标数量和面积
        image_counts.append(len(categories))
        all_areas.extend(areas)
        total_objects += len(categories)
    
    # 计算各类别在图片中的出现频率
    class_frequency = {}
    for category in all_categories:
        if class_image_count[category] > 0:
            class_frequency[category] = class_image_count[category] / len(annotation_files)
    
    # 计算各类别面积统计
    class_area_stats = {}
    for category in all_categories:
        if class_object_count[category] > 0:
            areas = class_area_list[category]
            if areas:
                class_area_stats[category] = {
                    'total_area': class_area_sum[category],
                    'avg_area': class_area_sum[category] / len(areas),
                    'min_area': class_min_area[category],
                    'max_area': class_max_area[category],
                    'median_area': np.median(areas),
                    'std_area': np.std(areas) if len(areas) > 1 else 0.0,
                    'area_percentiles': {
                        '10th': np.percentile(areas, 10),
                        '25th': np.percentile(areas, 25),
                        '50th': np.percentile(areas, 50),
                        '75th': np.percentile(areas, 75),
                        '90th': np.percentile(areas, 90)
                    },
                    'areas': areas  # 保存原始面积列表用于后续分析
                }
    
    # 计算平均每张图片的目标数和平均面积
    avg_objects_per_image = total_objects / len(annotation_files) if len(annotation_files) > 0 else 0
    avg_area_per_object = total_area / total_objects if total_objects > 0 else 0
    avg_area_per_image = total_area / len(annotation_files) if len(annotation_files) > 0 else 0
    
    # 整体面积统计
    overall_area_stats = {}
    if all_areas:
        overall_area_stats = {
            'total_area': total_area,
            'avg_area': np.mean(all_areas),
            'min_area': min(all_areas),
            'max_area': max(all_areas),
            'median_area': np.median(all_areas),
            'std_area': np.std(all_areas),
            'area_percentiles': {
                '10th': np.percentile(all_areas, 10),
                '25th': np.percentile(all_areas, 25),
                '50th': np.percentile(all_areas, 50),
                '75th': np.percentile(all_areas, 75),
                '90th': np.percentile(all_areas, 90)
            }
        }
    
    # 整理结果
    results = {
        'labels_dir': labels_dir,
        'total_files': len(annotation_files),
        'total_objects': total_objects,
        'empty_files': empty_files,
        'avg_objects_per_image': avg_objects_per_image,
        'avg_area_per_object': avg_area_per_object,
        'avg_area_per_image': avg_area_per_image,
        'overall_area_stats': overall_area_stats,
        'class_object_count': dict(class_object_count),
        'class_image_count': dict(class_image_count),
        'class_frequency': class_frequency,
        'class_area_stats': class_area_stats,
        'class_min_area': dict(class_min_area),
        'class_max_area': dict(class_max_area),
        'image_counts': image_counts,
        'all_areas': all_areas,
        'valid_classes': valid_classes,
        'skip_classes': skip_classes
    }
    
    return results

def print_statistics(results, show_details=False, sort_by='count'):
    """
    打印统计结果
    
    Args:
        results: 统计结果字典
        show_details: 是否显示详细信息
        sort_by: 排序方式，'count'按数量，'name'按名称
    """
    if not results:
        print("没有统计结果")
        return
    
    print("\n" + "="*60)
    print("KITTI标注数据统计结果")
    print("="*60)
    
    # 基本信息
    print(f"标注目录: {results['labels_dir']}")
    print(f"总文件数: {results['total_files']}")
    print(f"总目标数: {results['total_objects']}")
    print(f"空文件数: {results['empty_files']}")
    print(f"平均每张图片目标数: {results['avg_objects_per_image']:.2f}")
    print(f"平均每个目标面积: {results['avg_area_per_object']:.2f}")
    print(f"平均每张图片总面积: {results['avg_area_per_image']:.2f}")
    
    # 整体面积统计
    if results['overall_area_stats']:
        print("\n整体边界框面积统计:")
        stats = results['overall_area_stats']
        print(f"  总面积: {stats['total_area']:.2f}")
        print(f"  平均面积: {stats['avg_area']:.2f}")
        print(f"  最小面积: {stats['min_area']:.2f}")
        print(f"  最大面积: {stats['max_area']:.2f}")
        print(f"  中位数面积: {stats['median_area']:.2f}")
        print(f"  面积标准差: {stats['std_area']:.2f}")
    
    # 类别统计
    class_object_count = results['class_object_count']
    if not class_object_count:
        print("\n没有找到任何目标")
        return
    
    print(f"\n总类别数: {len(class_object_count)}")
    
    # 按指定方式排序
    if sort_by == 'count':
        sorted_categories = sorted(class_object_count.items(), key=lambda x: x[1], reverse=True)
    else:  # sort_by == 'name'
        sorted_categories = sorted(class_object_count.items(), key=lambda x: x[0])
    
    # 打印类别统计表
    print("\n" + "-"*100)
    print(f"{'类别':<15} {'目标数量':<12} {'图片数量':<12} {'平均面积':<12} {'最小面积':<12} {'最大面积':<12} {'总面积':<12}")
    print("-"*100)
    
    for category, count in sorted_categories:
        img_count = results['class_image_count'].get(category, 0)
        frequency = results['class_frequency'].get(category, 0)
        area_stats = results['class_area_stats'].get(category, {})
        avg_area = area_stats.get('avg_area', 0) if area_stats else 0
        min_area = area_stats.get('min_area', 0) if area_stats else 0
        max_area = area_stats.get('max_area', 0) if area_stats else 0
        total_area = area_stats.get('total_area', 0) if area_stats else 0
        
        print(f"{category:<15} {count:<12} {img_count:<12} {avg_area:<12.2f} {min_area:<12.2f} {max_area:<12.2f} {total_area:<12.2f}")
    
    print("-"*100)
    
    # 显示详细信息
    if show_details:
        print("\n详细信息:")
        print(f"有效类别: {results['valid_classes']}")
        print(f"跳过类别: {results['skip_classes']}")
        
        # 计算每张图片目标数的统计信息
        image_counts = results['image_counts']
        if image_counts:
            print(f"\n每张图片目标数统计:")
            print(f"  最小值: {min(image_counts)}")
            print(f"  最大值: {max(image_counts)}")
            print(f"  平均值: {np.mean(image_counts):.2f}")
            print(f"  中位数: {np.median(image_counts):.2f}")
            print(f"  标准差: {np.std(image_counts):.2f}")
        
        # 统计类别分布
        print(f"\n类别分布:")
        total_objects = results['total_objects']
        for category, count in sorted_categories:
            percentage = (count / total_objects) * 100 if total_objects > 0 else 0
            print(f"  {category}: {count} ({percentage:.2f}%)")
        
        # 详细面积统计
        print(f"\n详细面积统计:")
        for category, count in sorted_categories[:10]:  # 只显示前10个类别
            area_stats = results['class_area_stats'].get(category)
            if area_stats:
                print(f"\n  {category}:")
                print(f"    最小面积: {area_stats['min_area']:.2f}")
                print(f"    最大面积: {area_stats['max_area']:.2f}")
                print(f"    平均面积: {area_stats['avg_area']:.2f}")
                print(f"    中位数面积: {area_stats['median_area']:.2f}")
                print(f"    面积标准差: {area_stats['std_area']:.2f}")
                print(f"    10%分位数: {area_stats['area_percentiles']['10th']:.2f}")
                print(f"    90%分位数: {area_stats['area_percentiles']['90th']:.2f}")
                
                # 计算面积范围
                area_range = area_stats['max_area'] - area_stats['min_area']
                print(f"    面积范围: {area_range:.2f}")
                
                # 计算面积变异系数
                if area_stats['avg_area'] > 0:
                    cv = (area_stats['std_area'] / area_stats['avg_area']) * 100
                    print(f"    变异系数: {cv:.2f}%")

def save_statistics(results, output_dir, formats=['json', 'txt']):
    """
    保存统计结果
    
    Args:
        results: 统计结果字典
        output_dir: 输出目录
        formats: 输出格式列表
    """
    os.makedirs(output_dir, exist_ok=True)
    
    base_name = "kitti_statistics"
    
    # 保存为JSON格式
    if 'json' in formats:
        json_path = os.path.join(output_dir, f"{base_name}.json")
        with open(json_path, 'w') as f:
            # 处理numpy类型
            def convert_to_python(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                return obj
            
            json.dump(results, f, indent=2, default=convert_to_python)
        print(f"统计结果已保存为JSON: {json_path}")
    
    # 保存为TXT格式
    if 'txt' in formats:
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(txt_path, 'w') as f:
            f.write("KITTI标注数据统计结果\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"标注目录: {results['labels_dir']}\n")
            f.write(f"总文件数: {results['total_files']}\n")
            f.write(f"总目标数: {results['total_objects']}\n")
            f.write(f"空文件数: {results['empty_files']}\n")
            f.write(f"平均每张图片目标数: {results['avg_objects_per_image']:.2f}\n")
            f.write(f"平均每个目标面积: {results['avg_area_per_object']:.2f}\n")
            f.write(f"平均每张图片总面积: {results['avg_area_per_image']:.2f}\n\n")
            
            f.write("类别统计:\n")
            f.write("-"*100 + "\n")
            f.write(f"{'类别':<15} {'目标数量':<12} {'图片数量':<12} {'平均面积':<12} {'最小面积':<12} {'最大面积':<12} {'总面积':<12}\n")
            f.write("-"*100 + "\n")
            
            class_object_count = results['class_object_count']
            sorted_categories = sorted(class_object_count.items(), key=lambda x: x[1], reverse=True)
            
            for category, count in sorted_categories:
                img_count = results['class_image_count'].get(category, 0)
                frequency = results['class_frequency'].get(category, 0)
                area_stats = results['class_area_stats'].get(category, {})
                avg_area = area_stats.get('avg_area', 0) if area_stats else 0
                min_area = area_stats.get('min_area', 0) if area_stats else 0
                max_area = area_stats.get('max_area', 0) if area_stats else 0
                total_area = area_stats.get('total_area', 0) if area_stats else 0
                
                f.write(f"{category:<15} {count:<12} {img_count:<12} {avg_area:<12.2f} {min_area:<12.2f} {max_area:<12.2f} {total_area:<12.2f}\n")
            
            f.write("-"*100 + "\n")
        
        print(f"统计结果已保存为TXT: {txt_path}")

def plot_statistics(results, output_dir, figsize=(16, 12)):
    """
    绘制统计图表
    
    Args:
        results: 统计结果字典
        output_dir: 输出目录
        figsize: 图表尺寸
    """
    os.makedirs(output_dir, exist_ok=True)
    
    class_object_count = results['class_object_count']
    if not class_object_count:
        print("没有数据可绘制图表")
        return
    
    # 准备数据
    categories = list(class_object_count.keys())
    object_counts = list(class_object_count.values())
    
    # 按数量排序
    sorted_indices = np.argsort(object_counts)[::-1]
    categories_sorted = [categories[i] for i in sorted_indices]
    object_counts_sorted = [object_counts[i] for i in sorted_indices]
    
    # 获取面积数据
    avg_areas = []
    min_areas = []
    max_areas = []
    for category in categories_sorted:
        area_stats = results['class_area_stats'].get(category, {})
        avg_area = area_stats.get('avg_area', 0) if area_stats else 0
        min_area = area_stats.get('min_area', 0) if area_stats else 0
        max_area = area_stats.get('max_area', 0) if area_stats else 0
        avg_areas.append(avg_area)
        min_areas.append(min_area)
        max_areas.append(max_area)
    
    # 创建图表 (3x2布局)
    fig, axes = plt.subplots(3, 2, figsize=figsize)
    
    # 1. Number of objects per category bar chart
    ax1 = axes[0, 0]
    bars = ax1.bar(range(len(categories_sorted)), object_counts_sorted, color='skyblue')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Number of Objects')
    ax1.set_title('Number of Objects per Category')
    ax1.set_xticks(range(len(categories_sorted)))
    ax1.set_xticklabels(categories_sorted, rotation=45, ha='right')
    
    # 在柱状图上添加数值
    for bar, count in zip(bars, object_counts_sorted):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1 * max(object_counts_sorted),
                f'{count}', ha='center', va='bottom', fontsize=9)
    
    # 2. Category distribution pie chart
    ax2 = axes[0, 1]
    # 只显示前8个类别，其他的合并为"其他"
    if len(object_counts_sorted) > 8:
        top_categories = categories_sorted[:7]
        top_counts = object_counts_sorted[:7]
        other_count = sum(object_counts_sorted[7:])
        top_categories.append('Other')
        top_counts.append(other_count)
    else:
        top_categories = categories_sorted
        top_counts = object_counts_sorted
    
    wedges, texts, autotexts = ax2.pie(top_counts, labels=top_categories, autopct='%1.1f%%', 
                                      startangle=90, textprops={'fontsize': 9})
    ax2.set_title('Category Distribution')
    ax2.axis('equal')
    
    # 3. Objects per image distribution histogram
    ax3 = axes[1, 0]
    image_counts = results['image_counts']
    ax3.hist(image_counts, bins=30, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Objects per Image')
    ax3.set_ylabel('Number of Images')
    ax3.set_title('Distribution of Objects per Image')
    ax3.grid(True, alpha=0.3)
    
    # 添加平均线
    avg_count = results['avg_objects_per_image']
    ax3.axvline(avg_count, color='red', linestyle='--', linewidth=2, 
               label=f'Average: {avg_count:.2f}')
    ax3.legend()
    
    # 4. Average bounding box area per category
    ax4 = axes[1, 1]
    if any(avg_areas):
        # 过滤掉面积为0的类别
        valid_indices = [i for i, area in enumerate(avg_areas) if area > 0]
        if valid_indices:
            valid_categories = [categories_sorted[i] for i in valid_indices]
            valid_avg_areas = [avg_areas[i] for i in valid_indices]
            valid_min_areas = [min_areas[i] for i in valid_indices]
            valid_max_areas = [max_areas[i] for i in valid_indices]
            
            # 创建分组柱状图
            x = np.arange(len(valid_categories))
            width = 0.25
            
            bars1 = ax4.bar(x - width, valid_min_areas, width, label='Min Area', color='lightblue')
            bars2 = ax4.bar(x, valid_avg_areas, width, label='Avg Area', color='lightcoral')
            bars3 = ax4.bar(x + width, valid_max_areas, width, label='Max Area', color='lightgreen')
            
            ax4.set_xlabel('Category')
            ax4.set_ylabel('Bounding Box Area')
            ax4.set_title('Min/Avg/Max Bounding Box Area per Category')
            ax4.set_xticks(x)
            ax4.set_xticklabels(valid_categories, rotation=45, ha='right')
            ax4.legend()
            
            # 在柱状图上添加数值
            for bar, value in zip(bars1, valid_min_areas):
                height = bar.get_height()
                if height > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1 * max(valid_max_areas),
                            f'{value:.0f}', ha='center', va='bottom', fontsize=7)
            
            for bar, value in zip(bars2, valid_avg_areas):
                height = bar.get_height()
                if height > 0:
                    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.1 * max(valid_max_areas),
                            f'{value:.0f}', ha='center', va='bottom', fontsize=7)
        else:
            ax4.text(0.5, 0.5, 'No area data', ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Min/Avg/Max Bounding Box Area per Category')
    else:
        ax4.text(0.5, 0.5, 'No area data', ha='center', va='center', transform=ax4.transAxes)
        ax4.set_title('Min/Avg/Max Bounding Box Area per Category')
    
    # 5. Bounding box area distribution
    ax5 = axes[2, 0]
    all_areas = results.get('all_areas', [])
    if all_areas:
        ax5.hist(all_areas, bins=50, edgecolor='black', alpha=0.7, color='lightgreen')
        ax5.set_xlabel('Bounding Box Area')
        ax5.set_ylabel('Frequency')
        ax5.set_title('Bounding Box Area Distribution')
        ax5.grid(True, alpha=0.3)
        
        # 添加平均线和中位数线
        avg_area = np.mean(all_areas)
        median_area = np.median(all_areas)
        min_area = min(all_areas)
        max_area = max(all_areas)
        
        ax5.axvline(avg_area, color='red', linestyle='--', linewidth=2, 
                   label=f'Average: {avg_area:.2f}')
        ax5.axvline(median_area, color='blue', linestyle='--', linewidth=2,
                   label=f'Median: {median_area:.2f}')
        ax5.axvline(min_area, color='green', linestyle=':', linewidth=1.5,
                   label=f'Min: {min_area:.2f}')
        ax5.axvline(max_area, color='purple', linestyle=':', linewidth=1.5,
                   label=f'Max: {max_area:.2f}')
        ax5.legend()
    else:
        ax5.text(0.5, 0.5, 'No area data', ha='center', va='center', transform=ax5.transAxes)
        ax5.set_title('Bounding Box Area Distribution')
    
    # 6. Category occurrence frequency
    ax6 = axes[2, 1]
    class_frequency = results['class_frequency']
    sorted_freq = sorted(class_frequency.items(), key=lambda x: x[1], reverse=True)
    freq_categories = [item[0] for item in sorted_freq]
    freq_values = [item[1] for item in sorted_freq]
    
    bars3 = ax6.bar(range(len(freq_categories)), freq_values, color='lightgreen')
    ax6.set_xlabel('Category')
    ax6.set_ylabel('Occurrence Frequency')
    ax6.set_title('Category Occurrence Frequency')
    ax6.set_xticks(range(len(freq_categories)))
    ax6.set_xticklabels(freq_categories, rotation=45, ha='right')
    ax6.set_ylim([0, 1.1])
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plot_path = os.path.join(output_dir, "kitti_statistics.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"统计图表已保存: {plot_path}")
    
    # 单独保存面积相关图表
    if all_areas:
        fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
        
        # 面积分布直方图（详细）
        ax1 = axes2[0]
        ax1.hist(all_areas, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
        ax1.set_xlabel('Bounding Box Area')
        ax1.set_ylabel('Frequency (log scale)')
        ax1.set_title('Bounding Box Area Distribution (Log Scale)')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # 面积箱线图（按类别）
        ax2 = axes2[1]
        area_data = []
        area_labels = []
        for category in categories_sorted[:10]:  # 只显示前10个类别
            area_stats = results['class_area_stats'].get(category)
            if area_stats and len(area_stats.get('areas', [])) > 0:
                area_data.append(area_stats['areas'])
                area_labels.append(category)
        
        if area_data:
            ax2.boxplot(area_data, labels=area_labels)
            ax2.set_xlabel('Category')
            ax2.set_ylabel('Bounding Box Area')
            ax2.set_title('Bounding Box Area by Category (Box Plot)')
            ax2.set_xticklabels(area_labels, rotation=45, ha='right')
            ax2.set_yscale('log')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'No area data by category', ha='center', va='center', transform=ax2.transAxes)
        
        plt.tight_layout()
        area_plot_path = os.path.join(output_dir, "area_statistics.png")
        plt.savefig(area_plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"面积统计图表已保存: {area_plot_path}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='KITTI格式标注数据统计脚本')
    parser.add_argument('--labels-dir', type=str, default='labels',
                       help='KITTI标注目录 (默认: labels)')
    parser.add_argument('--output-dir', type=str, default='statistics',
                       help='输出目录 (默认: statistics)')
    parser.add_argument('--valid-classes', type=str, nargs='+', default=None,
                       help='有效的类别列表，只统计这些类别 (默认: 统计所有类别)')
    parser.add_argument('--skip-classes', type=str, nargs='+', default=None,
                       help='需要跳过的类别列表')
    parser.add_argument('--kitti-default-skip', action='store_true',
                       help='跳过KITTI默认的DontCare类别')
    parser.add_argument('--no-plot', action='store_true',
                       help='不生成图表')
    parser.add_argument('--sort-by', type=str, choices=['count', 'name'], default='count',
                       help='排序方式: count按数量, name按名称 (默认: count)')
    parser.add_argument('--verbose', action='store_true',
                       help='显示详细信息')
    parser.add_argument('--formats', type=str, nargs='+', default=['json', 'txt'],
                       help='输出格式: json, txt (默认: json txt)')
    
    args = parser.parse_args()
    
    print("="*60)
    print("KITTI标注数据统计")
    print("="*60)
    
    # 检查输入目录
    if not os.path.exists(args.labels_dir):
        print(f"错误: 标注目录不存在: {args.labels_dir}")
        return
    
    # 收集统计信息
    results = collect_statistics(
        args.labels_dir,
        valid_classes=args.valid_classes,
        skip_classes=args.skip_classes,
        kitti_default_skip=args.kitti_default_skip,
        verbose=args.verbose
    )
    
    if not results:
        print("统计失败")
        return
    
    # 打印统计结果
    print_statistics(results, show_details=args.verbose, sort_by=args.sort_by)
    
    # 保存统计结果
    save_statistics(results, args.output_dir, formats=args.formats)
    
    # 生成图表
    if not args.no_plot:
        print("\n生成统计图表...")
        plot_statistics(results, args.output_dir)
    
    print(f"\n统计完成! 结果保存在: {args.output_dir}")

if __name__ == '__main__':
    main()

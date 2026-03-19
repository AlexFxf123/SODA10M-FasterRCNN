# 筛选标注，选择面积较大的目标，删除小目标
import os
import shutil

def calculate_bbox_area(bbox_line):
    """
    计算KITTI格式标注中边界框的面积
    
    参数:
    bbox_line: KITTI标注文件中的一行字符串
    
    返回:
    float: 边界框的面积
    """
    # 分割行，KITTI格式: type truncated occluded alpha bbox dimensions location rotation_y score
    parts = bbox_line.strip().split()
    
    if len(parts) < 15:
        return 0.0
    
    try:
        # 提取边界框坐标: xmin, ymin, xmax, ymax (KITTI格式中的4-7列)
        xmin = float(parts[4])
        ymin = float(parts[5])
        xmax = float(parts[6])
        ymax = float(parts[7])
        
        # 计算面积
        width = xmax - xmin
        height = ymax - ymin
        
        # 确保宽度和高度非负
        if width < 0 or height < 0:
            return 0.0
            
        return width * height
    except (ValueError, IndexError):
        return 0.0

def filter_kitti_by_area(input_folder, output_folder, min_area=5000):
    """
    过滤KITTI标注文件，只保留面积大于等于min_area的标注
    
    参数:
    input_folder: 输入标注文件夹路径
    output_folder: 输出标注文件夹路径
    min_area: 最小面积阈值，默认5000
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 获取所有标注文件
    annotation_files = [f for f in os.listdir(input_folder) if f.endswith('.txt')]
    
    if not annotation_files:
        print(f"在文件夹 {input_folder} 中未找到任何.txt标注文件")
        return
    
    processed_count = 0
    total_removed = 0
    total_kept = 0
    
    for filename in annotation_files:
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)
        
        kept_lines = []
        removed_count = 0
        
        # 读取并处理每个文件
        with open(input_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        for line in lines:
            line = line.strip()
            if not line:  # 跳过空行
                continue
                
            # 计算当前边界框的面积
            area = calculate_bbox_area(line)
            
            if area >= min_area:
                kept_lines.append(line)
            else:
                removed_count += 1
        
        # 保存过滤后的结果
        if kept_lines:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(kept_lines))
        
        # 统计信息
        processed_count += 1
        total_removed += removed_count
        total_kept += len(kept_lines)
        
        print(f"已处理: {filename} - 保留: {len(kept_lines)} 个, 删除: {removed_count} 个")
    
    # 输出汇总信息
    print(f"\n处理完成!")
    print(f"总共处理文件数: {processed_count}")
    print(f"总共保留标注数: {total_kept}")
    print(f"总共删除标注数: {total_removed}")
    print(f"输出文件夹: {output_folder}")

def main():
    """
    主函数，设置路径和参数并执行过滤
    """
    # 设置路径 - 请根据实际情况修改
    input_labels_folder = "labels"  # 原始标注文件夹
    output_labels_folder = "labels_choose"  # 过滤后的标注文件夹
    
    # 最小面积阈值
    min_area_threshold = 2500
    
    # 检查输入文件夹是否存在
    if not os.path.exists(input_labels_folder):
        print(f"错误: 输入文件夹 '{input_labels_folder}' 不存在!")
        return
    
    # 执行过滤
    filter_kitti_by_area(input_labels_folder, output_labels_folder, min_area_threshold)
    
    # 测试函数
    test_calculate_area()

def test_calculate_area():
    """
    测试面积计算函数
    """
    print("\n--- 测试面积计算 ---")
    
    # 测试用例1: 正常边界框
    test_line1 = "Car 0.0 0 0.0 0.0 0.0 0.0 0.0 100.0 200.0 300.0 400.0 0.0 0.0 0.0 0.0"
    # bbox: (100, 200, 300, 400) -> 宽度:200, 高度:200, 面积:40000
    area1 = calculate_bbox_area(test_line1)
    print(f"测试1 - 正常边界框面积: {area1} (期望: 40000.0)")
    
    # 测试用例2: 小面积边界框
    test_line2 = "Car 0.0 0 0.0 0.0 0.0 0.0 0.0 100.0 200.0 120.0 210.0 0.0 0.0 0.0 0.0"
    # bbox: (100, 200, 120, 210) -> 宽度:20, 高度:10, 面积:200
    area2 = calculate_bbox_area(test_line2)
    print(f"测试2 - 小边界框面积: {area2} (期望: 200.0)")
    
    # 测试用例3: 无效行
    test_line3 = "Invalid line with fewer parts"
    area3 = calculate_bbox_area(test_line3)
    print(f"测试3 - 无效行面积: {area3} (期望: 0.0)")
    
    # 测试用例4: 负宽度边界框
    test_line4 = "Car 0.0 0 0.0 0.0 0.0 0.0 0.0 300.0 400.0 100.0 200.0 0.0 0.0 0.0 0.0"
    # bbox: (300, 400, 100, 200) -> 负宽度
    area4 = calculate_bbox_area(test_line4)
    print(f"测试4 - 负宽度边界框面积: {area4} (期望: 0.0)")

if __name__ == "__main__":
    main()

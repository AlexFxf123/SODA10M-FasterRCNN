# 格式化json脚本
import json

def format_json_file(input_file, output_file=None):
    """
    格式化JSON文件
    """
    # 读取JSON文件
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果未指定输出文件，则覆盖原文件
    if output_file is None:
        output_file = input_file
    
    # 写入格式化后的JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=False)
    
    print(f"JSON文件已格式化: {output_file}")

# 使用示例
format_json_file('labeled_trainval/SSLAD-2D/labeled/annotations/instance_val.json')
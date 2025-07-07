import os
import pandas as pd
import re

# 基础目录路径
base_directory = "/home/yzhong/gits/interpretable-pd/splits/gita-splitmono"

# 用于存储所有找到的CSV文件的列表
all_csv_files = []

# 递归查找所有子文件夹中的CSV文件
for root, dirs, files in os.walk(base_directory):
    for file in files:
        if file.endswith('.csv'):
            all_csv_files.append(os.path.join(root, file))

print(f"找到 {len(all_csv_files)} 个CSV文件")

# 处理每个CSV文件
total_replacements = 0

for file_path in all_csv_files:
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        file_replacements = 0
        
        # 对每一列进行处理
        for column in df.columns:
            if df[column].dtype == 'object':  # 只处理字符串类型列
                # 对于每个part编号(1-10)，替换_partX_为__partX
                for i in range(1, 11):
                    old_pattern = f"_part{i}_"
                    new_pattern = f"__part{i}"
                    
                    # 计算每列中替换的数量
                    col_replacements = df[column].astype(str).str.count(old_pattern).sum()
                    file_replacements += col_replacements
                    
                    # 执行替换
                    df[column] = df[column].astype(str).str.replace(old_pattern, new_pattern, regex=False)
        
        total_replacements += file_replacements
        
        # 保存修改后的CSV文件
        df.to_csv(file_path, index=False)
        print(f"已处理: {file_path} (替换了 {file_replacements} 处)")
    
    except Exception as e:
        print(f"处理 {file_path} 时出错: {e}")

print(f"所有CSV文件处理完成！总共替换了 {total_replacements} 处 '_partX_' 为 '__partX'")
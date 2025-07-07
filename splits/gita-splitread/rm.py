#!/usr/bin/env python3
# filepath: /home/yzhong/gits/interpretable-pd/scripts/filter_mono_csv.py

import os
import pandas as pd
import glob

def process_csv_files(root_dir):
    """
    处理指定目录及其子文件夹中的所有CSV文件，
    只保留标题行和第三列为'MONOLOGUE'的行
    """
    # 查找所有子文件夹中的CSV文件
    csv_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.lower().endswith('.csv'):
                csv_files.append(os.path.join(dirpath, filename))
    
    # 处理每个CSV文件
    for file_path in csv_files:
        try:
            print(f"处理文件: {file_path}")
            
            # 读取CSV文件，不将第一行作为数据
            df = pd.read_csv(file_path)
            
            # 确保文件至少有3列
            if df.shape[1] < 3:
                print(f"警告：{file_path} 列数少于3，跳过处理")
                continue
            
            # 获取标题行
            header = df.columns
            
            # 过滤第三列为'MONOLOGUE'的行
            monologue_rows = df[df.iloc[:, 2] == 'READ']
            
            # 创建新的DataFrame，只包含标题行和MONOLOGUE行
            result_df = pd.DataFrame(columns=header)
            result_df = pd.concat([result_df, monologue_rows])
            
            # 将结果写回原文件
            result_df.to_csv(file_path, index=False)
            
            print(f"文件已处理完成: {file_path}")
            print(f"保留了 {len(result_df)} 行MONOLOGUE数据 + 1个标题行")
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {str(e)}")

if __name__ == "__main__":
    # 指定目录路径
    import sys
    base_dir = sys.argv[1] 
    print(f"开始处理目录: {base_dir}")
    
    # 处理CSV文件
    process_csv_files(base_dir)
    
    print("所有CSV文件处理完成")
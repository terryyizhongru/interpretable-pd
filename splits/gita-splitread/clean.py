import os
import pandas as pd

# 需要移除的sample_id列表
sample_ids_to_remove = [
    "PD_READ_TEXT_AVPEPUDEA0003_READTEXT_part8",
    "PD_READ_TEXT_AVPEPUDEA0024_READTEXT_part5",
    "PD_READ_TEXT_AVPEPUDEA0058_READTEXT_part9"
]

# 基础目录路径
base_directory = os.getcwd()  # 获取当前工作目录

# 用于存储所有找到的CSV文件的列表
all_csv_files = []

# 递归查找所有子文件夹中的CSV文件
for root, dirs, files in os.walk(base_directory):
    for file in files:
        if file.endswith('.csv'):
            all_csv_files.append(os.path.join(root, file))

print(f"找到 {len(all_csv_files)} 个CSV文件")

# 处理每个CSV文件
for file_path in all_csv_files:
    try:
        # 读取CSV文件
        df = pd.read_csv(file_path)
        
        # 检查是否有sample_id列
        if 'sample_id' in df.columns:
            # 过滤掉特定sample_id的行
            original_row_count = len(df)
            df = df[~df['sample_id'].isin(sample_ids_to_remove)]
            removed_row_count = original_row_count - len(df)
            
            if removed_row_count > 0:
                print(f"{file_path}: 移除了 {removed_row_count} 行")
        
        # 替换所有列中的路径字符串
        for column in df.columns:
            if df[column].dtype == 'object':  # 只处理字符串类型列
                df[column] = df[column].astype(str).str.replace('./data/gita/', './data/gita-splitread/')
        
        # 保存修改后的CSV文件
        df.to_csv(file_path, index=False)
        print(f"已处理: {file_path}")
    
    except Exception as e:
        print(f"处理 {file_path} 时出错: {e}")

print("所有CSV文件处理完成！")
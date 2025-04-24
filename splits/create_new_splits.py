import pandas as pd
import os
import sys
import glob
from tqdm import tqdm

def generate_sample_id(row):
    """为 train.csv 或 test.csv 中的行生成 sample_id"""
    # 获取基本信息
    subject_id = row['speaker_id']
    status = row['status'].upper()
    
    # 从音频路径中提取任务类型和详情
    audio_path = row['audio_path']
    filename = os.path.basename(audio_path)
    task_detail = filename.split('_')[-1].split('.')[0].upper()
    
    # 确定任务类型
    if 'DDK_analysis' in audio_path:
        task_type = 'DDK_ANALYSIS'
    elif 'read_text' in audio_path:
        task_type = 'READ_TEXT'
    elif 'monologue' in audio_path:
        task_type = 'MONOLOGUE'
    elif 'sentences' in audio_path or 'sentences2' in audio_path:
        task_type = 'SENTENCES'
    else:
        task_type = 'WORDS'
    
    # 特殊情况处理
    if task_detail == 'READTEXT':
        task_type = 'READ_TEXT'
    elif 'MONOLOGO' in task_detail:
        task_detail = '-MONOLOGO-NR'
    elif task_detail == 'PRECUPADO':
        task_detail = 'PREOCUPADO'
        
    # 处理一些特殊情况
    if 'AVPEPUDEA0002READTEXT' in task_detail:
        task_detail = 'READTEXT'
    elif 'AVPEPUDEAC0012-TA' in task_detail:
        task_detail = '-TA'
    
    # 创建sample_id
    sample_id = f"{status}_{task_type}_{subject_id}_{task_detail}"
    return sample_id

def process_split_file(file_path, dataset_df, output_dir):
    """处理单个分割文件，生成 sample_id 并匹配 dataset.csv"""
    print(f"处理文件: {file_path}")
    
    # 读取分割文件
    split_df = pd.read_csv(file_path)
    
    # 生成 sample_id
    split_df['generated_sample_id'] = split_df.apply(generate_sample_id, axis=1)
    
    # 匹配 dataset.csv
    matched_rows = []
    for sample_id in tqdm(split_df['generated_sample_id']):
        matched = dataset_df[dataset_df['sample_id'] == sample_id]
        if not matched.empty:
            matched_rows.append(matched.iloc[0])
    
    # 创建新的 DataFrame
    if matched_rows:
        new_split_df = pd.DataFrame(matched_rows)
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_dir), exist_ok=True)
        
        # 保存到新的 CSV 文件
        output_file = os.path.join(output_dir, os.path.basename(file_path))
        
        columns_to_keep = ['subject_id', 'sample_id', 'task_id', 'label', 'prosody', 'wav2vec', 'phonation', 'articulation', 'glottal']

        new_split_df[columns_to_keep].to_csv(output_file, index=False)
        print(f"已保存到: {output_file}")
        print(f"匹配到 {len(matched_rows)} 行，总共 {len(split_df)} 行")
    else:
        print(f"警告: 没有匹配到任何行 - {file_path}")

def main():
    # 加载 dataset.csv
    dataset_path = './gita/dataset.csv'
    dataset_df = pd.read_csv(dataset_path)
    
    # 创建输出目录
    output_base_dir = 'pcgita_splits_10foldnew'
    os.makedirs(output_base_dir, exist_ok=True)
    
    # 查找所有的分割目录
    for fold in range(1, 11):
        input_dir = f'pcgita_splits/TRAIN_TEST_{fold}'
        output_dir = os.path.join(output_base_dir, f'fold_{fold-1}')
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理 train.csv 和 test.csv
        for split_type in ['train.csv', 'test.csv']:
            split_file = os.path.join(input_dir, split_type)
            if os.path.exists(split_file):
                process_split_file(split_file, dataset_df, output_dir)
            else:
                print(f"警告: 文件不存在 - {split_file}")

if __name__ == "__main__":
    main()

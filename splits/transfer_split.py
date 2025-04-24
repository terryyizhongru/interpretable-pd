import pandas as pd
import os
import sys

def main():
    # 确保提供了参数
    if len(sys.argv) < 2:
        print("Usage: python check_sample_ids.py [path to train.csv]")
        sys.exit(1)
    
    # 加载训练数据和数据集
    train_path = sys.argv[1]
    dataset_path = './gita/dataset.csv'
    
    train_df = pd.read_csv(train_path)
    dataset_df = pd.read_csv(dataset_path)
    
    # 获取数据集中所有的sample_id
    dataset_sample_ids = set(dataset_df['sample_id'].values)
    
    # 为train.csv数据生成sample_id
    def generate_sample_id(row):
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
        if 'MONOLOGO' in task_detail:
            task_detail = '-MONOLOGO-NR'
        
        # 创建sample_id
        sample_id = f"{status}_{task_type}_{subject_id}_{task_detail}"
        if sample_id == 'PD_READ_TEXT_AVPEPUDEA0002_AVPEPUDEA0002READTEXT':
            sample_id = 'PD_READ_TEXT_AVPEPUDEA0002_READTEXT'
        if sample_id == 'HC_DDK_ANALYSIS_AVPEPUDEAC0012_AVPEPUDEAC0012-TA':
            sample_id = 'HC_DDK_ANALYSIS_AVPEPUDEAC0012_-TA'
        return sample_id.replace('_PRECUPADO', '_PREOCUPADO')
    
    # 生成sample_ids并检查
    train_df['generated_sample_id'] = train_df.apply(generate_sample_id, axis=1)
    
    # 检查每个生成的sample_id是否在dataset.csv中
    found = []
    not_found = []
    
    for idx, row in train_df.iterrows():
        sample_id = row['generated_sample_id']
        if sample_id in dataset_sample_ids:
            found.append(sample_id)
        else:
            not_found.append((sample_id, row['audio_path']))
    
    # 打印结果
    print(f"总共检查了 {len(train_df)} 个样本")
    print(f"在dataset.csv中找到 {len(found)} 个样本")
    print(f"未找到 {len(not_found)} 个样本")
    
    # 如果有未找到的样本，打印前10个
    if not_found:
        print("\n未找到的样本示例:")
        for i, (sample_id, audio_path) in enumerate(not_found[:10]):
            print(f"{i+1}. {sample_id} (来自 {audio_path})")
    
    # 保存生成的sample_ids
    train_df[['audio_path', 'speaker_id', 'generated_sample_id']].to_csv('train_with_sample_ids.csv', index=False)
    
    # 保存未找到的sample_ids
    if not_found:
        with open('not_found_sample_ids.txt', 'w') as f:
            for sample_id, audio_path in not_found:
                f.write(f"{sample_id},{audio_path}\n")

if __name__ == "__main__":
    main()
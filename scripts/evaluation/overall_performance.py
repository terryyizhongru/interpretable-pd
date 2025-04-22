import os
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

def compute_average_report_across_runs(reports):
    overall_report = {
        'HC': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
        'PD': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
        'accuracy': [],
        'macro avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
        'weighted avg': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    }

    # Spoiler: It's gonna be inefficient :)

    for report in reports:
        for key in report.keys():
            if key == 'accuracy':
                overall_report[key].append(report[key])
            else:
                for key2 in report[key].keys():
                    overall_report[key][key2].append(report[key][key2])

    for key in overall_report.keys():
            if key == 'accuracy':
                overall_report[key] = f'{round(np.array(overall_report[key]).mean(), 4)}±{round(np.array(overall_report[key]).std(), 4)}'
            else:
                for key2 in report[key].keys():
                    overall_report[key][key2] = f'{round(np.array(overall_report[key][key2]).mean(), 4)}±{round(np.array(overall_report[key][key2]).std(), 4)}'

    # -- just for a more clean output
    overall_report = pd.DataFrame.from_dict(overall_report).T
    overall_report.iloc[2,0] = ''
    overall_report.iloc[2,1] = ''
    overall_report.iloc[2,3] = overall_report.iloc[3,3]

    return overall_report

def get_reports(exps_dir):
    val_reports = []
    test_reports = []
    run_dirs = os.listdir(exps_dir)
    for run_dir in run_dirs:
        val_preds = []
        val_labels = []
        test_preds = []
        test_labels = []

        run_dir_path = os.path.join(exps_dir, run_dir)
        fold_dirs = os.listdir(run_dir_path)
        
        for fold_dir in fold_dirs:
            model_output_dir = os.path.join(run_dir_path, fold_dir, 'model_output')

            # -- validation set
            val_report_path = os.path.join(os.path.join(model_output_dir, 'validation_classification.pkl'))
            with open(val_report_path, 'rb') as f:
                val_model_output = pickle.load(f)

            val_preds += val_model_output['preds']
            val_labels += val_model_output['labels']

            # -- test set
            test_report_path = os.path.join(os.path.join(model_output_dir, 'test_classification.pkl'))
            with open(test_report_path, 'rb') as f:
                test_model_output = pickle.load(f)

            test_preds += test_model_output['preds']
            test_labels += test_model_output['labels']

        # -- computing reports
        val_reports.append(
            classification_report(
                val_labels,
                val_preds,
                target_names=['HC', 'PD'],
                output_dict=True,
            )
        )

        test_reports.append(
            classification_report(
                test_labels,
                test_preds,
                target_names=['HC', 'PD'],
                output_dict=True,
            )
        )

    return val_reports, test_reports


def get_report_perfold(exps_dir):
    """
    获取每个run的每个fold的报告
    
    Args:
        exps_dir: 存储实验结果的目录路径
        
    Returns:
        val_fold_reports: 字典，键为run_dir，值为该run下每个fold的验证集报告列表
        test_fold_reports: 字典，键为run_dir，值为该run下每个fold的测试集报告列表
    """
    val_fold_reports = {}
    test_fold_reports = {}
    run_dirs = sorted(os.listdir(exps_dir))
    
    for run_dir in run_dirs:
        run_dir_path = os.path.join(exps_dir, run_dir)
        fold_dirs = sorted(os.listdir(run_dir_path))
        
        val_fold_reports[run_dir] = []
        test_fold_reports[run_dir] = []
        
        for fold_dir in fold_dirs:
            model_output_dir = os.path.join(run_dir_path, fold_dir, 'model_output')
            
            # -- validation set
            val_report_path = os.path.join(model_output_dir, 'validation_classification.pkl')
            if os.path.exists(val_report_path):
                with open(val_report_path, 'rb') as f:
                    val_model_output = pickle.load(f)
                
                # 计算单个fold的验证集分类报告
                val_fold_report = classification_report(
                    val_model_output['labels'],
                    val_model_output['preds'],
                    target_names=['HC', 'PD'],
                    output_dict=True,
                )
                val_fold_reports[run_dir].append({
                    'fold': fold_dir,
                    'report': val_fold_report,
                    'preds': val_model_output['preds'],
                    'labels': val_model_output['labels']
                })
            
            # -- test set
            test_report_path = os.path.join(model_output_dir, 'test_classification.pkl')
            if os.path.exists(test_report_path):
                with open(test_report_path, 'rb') as f:
                    test_model_output = pickle.load(f)
                
                # 计算单个fold的测试集分类报告
                test_fold_report = classification_report(
                    test_model_output['labels'],
                    test_model_output['preds'],
                    target_names=['HC', 'PD'],
                    output_dict=True,
                )
                test_fold_reports[run_dir].append({
                    'fold': fold_dir,
                    'report': test_fold_report,
                    'preds': test_model_output['preds'],
                    'labels': test_model_output['labels']
                })
    
    return val_fold_reports, test_fold_reports

def compute_fold_stats(fold_reports):
    """
    计算每个run中所有fold的性能指标的均值和标准差
    
    Args:
        fold_reports: 字典，键为run_dir，值为该run下每个fold的报告列表
        
    Returns:
        fold_stats: 字典，键为run_dir，值为包含accuracy和f1的均值和标准差的字典
    """
    fold_stats = {}
    
    for run_dir, fold_results in fold_reports.items():
        accuracies = [fold_result['report']['accuracy'] for fold_result in fold_results]
        f1_scores = [fold_result['report']['PD']['f1-score'] for fold_result in fold_results]
        precisions = [fold_result['report']['PD']['precision'] for fold_result in fold_results]
        recalls = [fold_result['report']['PD']['recall'] for fold_result in fold_results]
        
        fold_stats[run_dir] = {
            'accuracy': {
                'mean': np.mean(accuracies),
                'std': np.std(accuracies)
            },
            'f1': {
                'mean': np.mean(f1_scores),
                'std': np.std(f1_scores)
            },
            'precision': {
                'mean': np.mean(precisions),
                'std': np.std(precisions)
            },
            'recall': {
                'mean': np.mean(recalls),
                'std': np.std(recalls)
            }
        }
    
    return fold_stats

if __name__ == '__main__':
    # -- command-line arguments
    parser = argparse.ArgumentParser(description='Performance analysis across runs and folds.',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exps-dir', required=True, type=str, help='Directory where the experiments are stored.')

    args = parser.parse_args()

    # 获取每个fold的报告
    val_fold_reports, test_fold_reports = get_report_perfold(args.exps_dir)
    
    # 计算每个run的统计数据
    val_fold_stats = compute_fold_stats(val_fold_reports)
    test_fold_stats = compute_fold_stats(test_fold_reports)
    
    # 打印每个run每个fold的结果
    print("\nValidation Performance Per Fold:")
    for run_dir, fold_results in val_fold_reports.items():
        print(f"\nRun: {run_dir}")
        for fold_result in fold_results:
            fold = fold_result['fold']
            report = fold_result['report']
            print(f"  Fold: {fold}, Accuracy: {report['accuracy']:.4f}, F1 (PD): {report['PD']['f1-score']:.4f}")
        
        # 打印该run的所有fold的均值和标准差
        stats = val_fold_stats[run_dir]
        print(f"  Mean Accuracy: {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")
        print(f"  Mean F1 (PD): {stats['f1']['mean']:.4f} ± {stats['f1']['std']:.4f}")
        print(f"  Mean Precision (PD): {stats['precision']['mean']:.4f} ± {stats['precision']['std']:.4f}")
        print(f"  Mean Recall (PD): {stats['recall']['mean']:.4f} ± {stats['recall']['std']:.4f}")
    
    print("\nTest Performance Per Fold:")
    for run_dir, fold_results in test_fold_reports.items():
        print(f"\nRun: {run_dir}")
        for fold_result in fold_results:
            fold = fold_result['fold']
            report = fold_result['report']
            print(f"  Fold: {fold}, Accuracy: {report['accuracy']:.4f}, F1 (PD): {report['PD']['f1-score']:.4f}")
        
        # 打印该run的所有fold的均值和标准差
        stats = test_fold_stats[run_dir]
        print(f"  Mean Accuracy: {stats['accuracy']['mean']:.4f} ± {stats['accuracy']['std']:.4f}")
        print(f"  Mean F1 (PD): {stats['f1']['mean']:.4f} ± {stats['f1']['std']:.4f}")
        print(f"  Mean Precision (PD): {stats['precision']['mean']:.4f} ± {stats['precision']['std']:.4f}")
        print(f"  Mean Recall (PD): {stats['recall']['mean']:.4f} ± {stats['recall']['std']:.4f}")
    
    # 也可以继续使用原来的函数获取汇总报告
    val_reports, test_reports = get_reports(args.exps_dir)
    val_overall_report = compute_average_report_across_runs(val_reports)
    test_overall_report = compute_average_report_across_runs(test_reports)
    
    print('\n\nOverall Validation Performance:\n', val_overall_report)
    print('\n\nOverall Test Performance:\n', test_overall_report)

<h1 align="center"><span style="font-weight:normal">RECA-PD: A Robust Explainable Cross-Attention Method for Speech-based Parkinson's Disease Classification</h1>
  
<div align="center">
  
[Terry Yi Zhong](https://terryyizhong.github.io/)
</div>

<div align="center">
  
[📘 Introduction](#intro) |
[🛠️ Data Preparation](#preparation) |
[🚀 Training and Evaluation](#training) |
[📖 Citation](#citation) |
[📝 License](#license)
</div>

## <a name="intro"></a> 📘 Introduction

<div align="center"> <img src="docs/TSD-V1.2.drawio.png"  width="720"> </div>

**Abstract.** _Parkinson's Disease (PD) affects over 10 million people globally, with speech impairments often preceding motor symptoms by years, making speech a valuable modality for early, non-invasive detection. While recent deep-learning models achieve high accuracy, they typically lack the explainability required for clinical use. To address this, we propose RECA-PD, a novel, robust, and explainable cross-attention architecture that combines interpretable speech features with self-supervised representations. RECA-PD matches state-of-the-art performance in Parkinson’s disease detection while providing explanations that are more consistent and more clinically meaningful. Additionally, we demonstrate that performance degradation in certain speech tasks (e.g., monologue) can be mitigated by segmenting long recordings. Our findings indicate that performance and explainability are not necessarily mutually exclusive. Future work will enhance the usability of explanations for non-experts and explore severity estimation to increase the real-world clinical relevance._ [📜 Arxiv Link](https://arxiv.org/abs/) [📜 TSD 2025 Link]()

## <a name="preparation"></a> 🛠️ Preparation

- Prepare the **conda environment** to run the experiments:

```
conda create -n ssl-parkinson python=3.10
conda activate ssl-parkinson
pip install -r requirements.txt
```

## <a name="training"></a> 🚀 Training and Evaluation

To train and evaluate our proposed framework, we should follow a pipeline consisting of multiple steps, including data preprocessing, dataset split, feature extraction, as well as the ultimate training and evaluation. As an example, we provide the scripts aimed to address our GITA corpus experiments:

```
bash scripts/runs/dataset_preparation/gita.sh $DATASET_DIR $METADATA_PATH
bash scripts/runs/feature_extraction/gita.sh
bash scripts/runs/experiments/cross_full/gita.sh
```

, where `$DATASET_DIR` and `$METADATA_PATH` refer to the directory containing all the audio waveform samples and the CSV including the corpus subject metadata, respectively. _Please, note that you have to convert the 1st sheet of the .xlsx provided in the GITA dataset to a .csv file._

In order to **evaluate your model** for a specific assessment task across all repetitions and folds, you can run the following command:

```
python scripts/evaluation/overall_performance.py --exps-dir ./exps/gita/cross_full/$TASK/
```

, where `$TASK` corresponds to the name of the target task you want to evaluate. You can always inspect the directory `scripts/evaluation/` to find other interesting scripts.

## <a name="citation"></a> 📖 Citation




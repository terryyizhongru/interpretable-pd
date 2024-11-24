#!/bin/bash

output_dir=./plots/gita/cross_full/phoneme_analysis/high-level/

for sentence_id in JUAN ROSITA LOSLIBROS MICASA LUISA LAURA VISTE TRISTE OMAR; do

  python scripts/interpretability/phoneme_analysis.py \
    --exps-dir ./exps2/gita/cross_full/layer07/SENTENCES/ \
    --wav2vec-dir ./data/gita/speech_features/wav2vec/layer07/ \
    --metadata-path ./splits/gita/task_dataset.csv \
    --study-coarticulation true \
    --output-dir $output_dir/SENTENCES/coarticulations/${sentence_id}/ \
    --task-id SENTENCES \
    --filter-by-id $sentence_id \
    --phoneme-alignment ~/Documents/mfa_alignments/gita/output_sentences_/${sentence_id}.TextGrid

done;

# -- special case because some sample IDs were not properly spelled
python scripts/interpretability/phoneme_analysis.py \
  --exps-dir ./exps2/gita/cross_full/layer07/SENTENCES/ \
  --wav2vec-dir ./data/gita/speech_features/wav2vec/layer07/ \
  --metadata-path ./splits/gita/task_dataset.csv \
  --study-coarticulation true \
  --output-dir $output_dir/SENTENCES/coarticulations/PREOCUPADO/ \
  --task-id SENTENCES \
  --filter-by-id CUPADO \
  --phoneme-alignment ~/Documents/mfa_alignments/gita/output_sentences_/PREOCUPADO.TextGrid

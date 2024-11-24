#!/bin/bash

output_dir=./plots/gita/cross_full/contrastive_analysis/

for sentence_id in JUAN VISTE TRISTE; do

  python scripts/interpretability/contrastive_temporal_cross_attention_analisys.py \
    --exps-dir ./exps2/gita/cross_full/layer07/SENTENCES/ \
    --wav2vec-dir ./data/gita/speech_features/wav2vec/layer07/ \
    --output-dir $output_dir/SENTENCES/word-emphasis/${sentence_id}/ \
    --task-id SENTENCES \
    --filter-by-id $sentence_id \
    --word-level true \
    --phoneme-alignment ~/Documents/mfa_alignments/gita/output_sentences_/${sentence_id}.TextGrid

done;

# -- special case because some sample IDs were not properly spelled
python scripts/interpretability/contrastive_temporal_cross_attention_analisys.py \
  --exps-dir ./exps2/gita/cross_full/layer07/SENTENCES/ \
  --wav2vec-dir ./data/gita/speech_features/wav2vec/layer07/ \
  --output-dir $output_dir/SENTENCES/word-emphasis/PREOCUPADO/ \
  --task-id SENTENCES \
  --filter-by-id CUPADO \
  --word-level true \
  --phoneme-alignment ~/Documents/mfa_alignments/gita/output_sentences_/PREOCUPADO.TextGrid

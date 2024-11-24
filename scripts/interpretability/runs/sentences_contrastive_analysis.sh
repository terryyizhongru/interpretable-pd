#!/bin/bash

exps_dir=./exps2/gita/cross_full_nolinearkeys/layer07
output_dir=./plots/gita/cross_full_nolinearkeys/temporal_analysis/

for sentence_id in MICASA JUAN ROSITA LOSLIBROS LUISA LAURA VISTE TRISTE OMAR; do

  python scripts/interpretability/contrastive_temporal_cross_attention_analisys.py \
    --exps-dir $exps_dir/SENTENCES/ \
    --wav2vec-dir ./data/gita/speech_features/wav2vec/layer07/ \
    --output-dir $output_dir/SENTENCES/phoneme-level/${sentence_id}/ \
    --task-id SENTENCES \
    --filter-by-id $sentence_id \
    --phoneme-alignment ~/Documents/mfa_alignments/gita/output_sentences_/${sentence_id}.TextGrid
done;

# -- special case because some sample IDs were not properly spelled
python scripts/interpretability/contrastive_temporal_cross_attention_analisys.py \
  --exps-dir $exps_dir/SENTENCES/ \
  --wav2vec-dir ./data/gita/speech_features/wav2vec/layer07/ \
  --output-dir $output_dir/SENTENCES/phoneme-level/PREOCUPADO/ \
  --task-id SENTENCES \
  --filter-by-id CUPADO \
  --phoneme-alignment ~/Documents/mfa_alignments/gita/output_sentences_/PREOCUPADO.TextGrid

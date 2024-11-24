#!/bin/bash

output_dir=./plots/gita/cross_full_nolinearkeys/embedding_analysis/

python scripts/interpretability/embedding_cross_attention_analisys.py \
  --exps-dir ./exps2/gita/cross_full_nolinearkeys/layer07/ \
  --output-dir $output_dir/SENTENCES/ALL/ \

for sentence_id in JUAN ROSITA LOSLIBROS MICASA LUISA LAURA VISTE TRISTE OMAR; do

  python scripts/interpretability/embedding_cross_attention_analisys.py \
    --exps-dir ./exps2/gita/cross_full_nolinearkeys/layer07/SENTENCES/ \
    --output-dir $output_dir/SENTENCES/${sentence_id}/ \
    --filter-by-id $sentence_id
done;

# -- special case because some sample IDs were not properly spelled
python scripts/interpretability/embedding_cross_attention_analisys.py \
  --exps-dir ./exps2/gita/cross_full_nolinearkeys/layer07/SENTENCES/ \
  --output-dir $output_dir/SENTENCES/PREOCUPADO/ \
  --filter-by-id CUPADO

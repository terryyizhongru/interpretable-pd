#!/bin/bash

set -e

# dataset=pcgita_splits_10foldnew
dataset=gita
# for task in SUSTAINED-VOWELS WORDS DDK SENTENCES READ-TEXT MONOLOGUE; do
# for seed in 12 21 33 42 52; do
for task in SENTENCES; do
for seed in 12; do
for f in 0; do

  CUDA_VISIBLE_DEVICES=0 python scripts/model_pipeline/pipeline.py \
      --config ./configs/framework-new.yaml \
      --training-dataset ./splits/$dataset/fold_$f/train.csv \
      --validation-dataset ./splits/$dataset/fold_$f/test.csv \
      --test-dataset ./splits/$dataset/fold_$f/test.csv \
      --output-dir ./exps/$dataset/cross_full/${task}/seed${seed}/fold_${f}/ \
      --yaml-overrides device:cuda seed:$seed model:cross_full \
      --save-attention-scores False \
      --filter-tasks $task 

done;

done;
done;

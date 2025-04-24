#!/bin/bash

set -e

dataset=pcgita_splits_10foldnew
# dataset=gita
# for task in SUSTAINED-VOWELS WORDS DDK SENTENCES READ-TEXT MONOLOGUE; do
# for seed in 12 21 33 42 52; do
# for f in 0 1 2 3 4; do
# for f in 0 1 2 3 4 5 6 7 8 9; do


for task in allsent; do
for seed in 12 21 33 42 52; do
for f in 0 1 2 3 4 5 6 7 8 9; do

  CUDA_VISIBLE_DEVICES=0 python scripts/model_pipeline/pipeline.py \
      --config ./configs/framework-new.yaml \
      --training-dataset ./splits/$dataset/fold_$f/train.csv \
      --validation-dataset ./splits/$dataset/fold_$f/test.csv \
      --test-dataset ./splits/$dataset/fold_$f/test.csv \
      --output-dir ./exps/$dataset/cross_new/${task}/seed${seed}/fold_${f}/ \
      --yaml-overrides device:cuda seed:$seed model:cross_full \
      --save-attention-scores False 
      # --filter-tasks $task 

done;

done;
done;

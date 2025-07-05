#!/bin/bash

STAGE='stage2'
BATCH_SIZE=256
DATASET="cifar10_full"
DEVICE="cuda:0"
MAX_LR=1e-5
MIN_LR=1e-7
CHECKPOINT="your trained checkpoint path after stage1"
NSGA_path="./NSGA/cifar10.csv"
GEN_ID=300

python train_stage2.py \
  --stage $STAGE \
  --batch_size $BATCH_SIZE \
  --max_lr $MAX_LR \
  --min_lr $MIN_LR \
  --dataset $DATASET \
  --stage1_checkpoint_path $CHECKPOINT \
  --nsga_path $NSGA_path \
  --gen_id $GEN_ID \
  --device $DEVICE \

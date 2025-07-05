#!/bin/bash
STAGE='stage1'
BATCH_SIZE=256
DATASET="cifar10_full"
DEVICE="cuda:0"
MAX_LR=1e-5
MIN_LR=1e-7
CHECKPOINT="./pretrained_para/rearranged_para.pth"

python train_stage1.py \
  --stage $STAGE \
  --batch_size $BATCH_SIZE \
  --max_lr $MAX_LR \
  --min_lr $MIN_LR \
  --dataset $DATASET \
  --rearranged_checkpoint_path $CHECKPOINT \
  --device $DEVICE \

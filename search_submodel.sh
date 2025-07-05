#!/bin/bash

STAGE='stage2_nsga'
BATCH_SIZE=256
DATASET="cifar10_full"
DEVICE="cuda:0"
CHECKPOINT="your trained checkpoint path after stage1"
NSGA_path="./NSGA/cifar10.csv"

python search_submodel.py \
  --stage $STAGE \
  --batch_size $BATCH_SIZE \
  --dataset $DATASET \
  --stage1_checkpoint_path $CHECKPOINT \
  --nsga_path $NSGA_path \
  --device $DEVICE \

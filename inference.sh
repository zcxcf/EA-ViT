#!/bin/bash

STAGE='inference eval'
BATCH_SIZE=16
DATASET="cifar10_full"
DEVICE="cuda:0"
CHECKPOINT="your trained checkpoint path after stage2"
CONSTRAINT=0.5

python inference.py \
  --stage $STAGE \
  --batch_size $BATCH_SIZE \
  --dataset $DATASET \
  --stage2_checkpoint_path $CHECKPOINT \
  --constrant $CONSTRAINT \
  --device $DEVICE \

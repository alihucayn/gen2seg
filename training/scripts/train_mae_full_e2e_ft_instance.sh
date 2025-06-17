#!/bin/bash

accelerate launch --multi_gpu --config_file training/scripts/multi_gpu.yaml training/train_mae_full.py \
  --pretrained_model_name_or_path "facebook/vit-mae-huge" \
  --max_train_steps 30000 \
  --checkpointing_steps 10000 \
  --train_batch_size 2  \
  --gradient_accumulation_steps 4 \
  --gradient_checkpointing \
  --learning_rate 6e-05 \
  --lr_total_iter_length 40000 \
  --lr_exp_warmup_steps 100 \
  --mixed_precision "no" \
  --output_dir "model-finetuned/mae-huge-instance" \
  --enable_xformers_memory_efficient_attention \
  "$@"
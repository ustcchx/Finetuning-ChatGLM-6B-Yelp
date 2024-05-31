#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --config_file LLaMA-Factory/LLaMA-Factory-main/examples/accelerate/master_config.yaml \
    LLaMA-Factory/LLaMA-Factory-main/src/train_bash.py \
    --stage sft \
    --do_train True \
    --model_name_or_path chatglm3-6b-base-model/models--THUDM--chatglm3-6b-base/snapshots/f91a1de587fdc692073367198e65369669a0b49d \
    --dataset yelp_review_full_train \
    --dataset_dir LLaMA-Factory/LLaMA-Factory-main/data \
    --template default \
    --finetuning_type lora \
    --lora_target query_key_value \
    --lora_dropout 0.08 \
    --lora_rank 8 \
    --output_dir LLaMA-Factory/LLaMA-Factory-main/saves/Chatglm-6B-base/lora/sft_1 \
    --overwrite_output_dir True \
    --cutoff_len 1024 \
    --preprocessing_num_workers 4 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --warmup_steps 20 \
    --save_steps 1000 \
    --eval_steps 500 \
    --evaluation_strategy steps \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --max_samples 160000 \
    --val_size 0.06 \
    --ddp_timeout 1800000 \
    --plot_loss True\
    --fp16

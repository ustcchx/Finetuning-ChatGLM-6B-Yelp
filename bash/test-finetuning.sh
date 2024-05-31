#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
    --config_file LLaMA-Factory/LLaMA-Factory-main/examples/accelerate/master_config_1.yaml \
    LLaMA-Factory/LLaMA-Factory-main/src/train_bash.py \
    --model_name_or_path chatglm3-6b-base-model/models--THUDM--chatglm3-6b-base/snapshots/f91a1de587fdc692073367198e65369669a0b49d \
    --stage sft \
    --do_predict true \
    --finetuning_type lora \
    --adapter_name_or_path LLaMA-Factory/LLaMA-Factory-main/saves/Chatglm-6B-base/lora/sft_1 \
    --dataset yelp_review_full_test \
    --dataset_dir LLaMA-Factory/LLaMA-Factory-main/data \
    --template default \
    --output_dir test-results-finetuning \
    --per_device_eval_batch_size 4 \
    --max_samples 50000 \
    --cutoff_len 1024 \
    --max_new_tokens 32 \
    --predict_with_generate true \
    --fp16 true \
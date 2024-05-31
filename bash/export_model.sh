#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python LLaMA-Factory/LLaMA-Factory-main/src/export_model.py \
    --model_name_or_path chatglm3-6b-base-model/models--THUDM--chatglm3-6b-base/snapshots/f91a1de587fdc692073367198e65369669a0b49d \
    --adapter_name_or_path LLaMA-Factory/LLaMA-Factory-main/saves/Chatglm-6B-base/lora/sft_1/checkpoint-s000 \
    --template default \
    --finetuning_type lora \
    --export_dir chatglm-finetuning \
    --export_size 2 \
    --export_legacy_format False
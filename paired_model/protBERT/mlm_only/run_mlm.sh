#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=prot_bert_mlm

#--config_name 'config3.json' \

eval "$(conda shell.bash hook)"
conda init bash
conda activate lea_env
/home/leab/anaconda3/envs/lea_env/bin/python run_mlm.py \
    --model_name_or_path Rostlab/prot_bert \
    --train_file /ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/data/small_subset_train_larger.txt \
    --validation_file /ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/data/small_subset_val_larger.txt \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluation_strategy 'epoch' \
    --save_strategy 'epoch' \
    --learning_rate 5e-5 \
    --weight_decay 0.1 \
    --num_train_epochs 20 \
    --lr_scheduler_type 'linear' \
    --log_level 'info' \
    --seed 42 \
    --data_seed 42 \
    --bf16 False \
    --project_name prot_bert_mlm_from_transformers \
    --load_best_model_at_end True \
    --metric_for_best_model 'loss' \
    --line_by_line \
    --greater_is_better False \
    --report_to 'wandb' \
    --max_seq_length 30 \
    --overwrite_output_dir \
    --output_dir ./test-mlm

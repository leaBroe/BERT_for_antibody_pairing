#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=paired_model_nsp

#--config_name 'config3.json' \
# --do_eval \


python run_mlm_nsp.py \
    --model_type 'roberta' \
    --tokenizer_name ./ProteinTokenizer \
    --train_file /ibmm_data2/oas_database/paired_lea_tmp/paired_model/src/redo_ch/test.txt \
    --validation_file /ibmm_data2/oas_database/paired_lea_tmp/paired_model/src/redo_ch/val.txt \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --do_train \
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
    --project_name test_paired_model_nsp_mlm \
    --load_best_model_at_end True \
    --metric_for_best_model 'loss' \
    --line_by_line \
    --greater_is_better False \
    --config_name 'config3.json' \
    --report_to 'wandb' \
    --max_seq_length 30 \
    --overwrite_output_dir \
    --output_dir ./test-mlm

#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=run_light_model
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/light_model/outputs/transformers_mlm/light_mlm_old_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/light_model/outputs/transformers_mlm/light_mlm_old_%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate req_ch_py39
/home/leab/anaconda3/envs/req_ch_py39/bin/python /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/run_mlm_old.py \
    --model_name_or_path FacebookAI/roberta-base \
    --train_file /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/training_set_test.txt \
    --validation_file /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/test_set_test.txt \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
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
    --run_name DALM_HEALTHY_bs16_lr5e5 \
    --project_name DALM_healthy \
    --load_best_model_at_end True \
    --metric_for_best_model 'loss' \
    --greater_is_better False \
    --max_seq_length 30 \
    --overwrite_output_dir \
    --output_dir /ibmm_data2/oas_database/paired_lea_tmp/light_model/outputs/tmp/test-mlm


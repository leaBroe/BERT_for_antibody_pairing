#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=run_light_model
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/light_model/outputs/transformers_mlm/run_light_model_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/light_model/outputs/transformers_mlm/run_light_model%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate transformers_mlm
/home/leab/anaconda3/envs/transformers_mlm/bin/python /ibmm_data2/oas_database/paired_lea_tmp/transformers/examples/pytorch/language-modeling/run_mlm.py \
    --model_name_or_path FacebookAI/roberta-base \
    --train_file /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/training_set_test.txt \
    --validation_file /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/test_set_test.txt \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --line_by_line \
    --overwrite_output_dir \
    --output_dir /ibmm_data2/oas_database/paired_lea_tmp/light_model/outputs/tmp/test-mlm


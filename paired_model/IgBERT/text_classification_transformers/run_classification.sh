#!/bin/bash


#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=cl_igbert
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/logs/igbert_classification_transformers_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/logs/igbert_classification_transformers_%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate class_env

/home/leab/anaconda3/envs/class_env/bin/python run_classification.py \
    --model_name_or_path  Exscientia/IgBERT \
    --train_file "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_train_with_unpaired.csv" \
    --validation_file "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_val_with_unpaired.csv" \
    --metric_name accuracy \
    --text_column_names "heavy,light" \
    --text_column_delimiter "," \
    --label_column_name label \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --report_to 'wandb' \
    --project_name classification_paired_model_transformers \
    --num_train_epochs 10 \
    --output_dir /ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/text_classification_transformers/full_seqs_text_classification_transformers
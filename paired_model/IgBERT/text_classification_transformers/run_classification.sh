#!/bin/bash


# --gres=gpu:h100:1
#SBATCH --job-name=cl_igbert
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/logs/igbert_classification_transformers_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/logs/igbert_classification_transformers_%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate lea_env
dataset="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_train_with_unpaired_small.csv"

/home/leab/anaconda3/envs/lea_env/bin/python run_classification.py \
    --model_name_or_path  Exscientia/IgBERT \
    --dataset_name ${dataset} \
    --metric_name accuracy \
    --text_column_name "heavy,light" \
    --text_column_delimiter "," \
    --label_column_name label \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 8 \
    --learning_rate 2e-5 \
    --num_train_epochs 4 \
    --output_dir /test_text_classification_transformers \
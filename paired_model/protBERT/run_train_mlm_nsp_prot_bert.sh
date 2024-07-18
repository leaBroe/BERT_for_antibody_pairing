#!/bin/bash


#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=protbert_nsp
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/logs/train_mlm_nsp_prot_bert_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/logs/train_mlm_nsp_prot_bert_%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate bug_env
/home/leab/anaconda3/envs/bug_env/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/train_mlm_nsp_prot_bert.py


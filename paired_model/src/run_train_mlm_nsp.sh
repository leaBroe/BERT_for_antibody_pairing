#!/bin/bash


#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=nsp_mlm_paired
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/logs/paired_model_nsp_mlm/paired_mlm_nsp_full_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/logs/paired_model_nsp_mlm/paired_mlm_nsp_full_%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate bug_env
/home/leab/anaconda3/envs/bug_env/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/src/train_mlm_nsp.py

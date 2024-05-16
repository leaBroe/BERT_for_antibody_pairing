#!/bin/bash


#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=2e-5_igbert
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/logs/igbert_2e-5%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/logs/igbert_2e-5%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate lea_env
/home/leab/anaconda3/envs/lea_env/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/light_heavy_classification.py

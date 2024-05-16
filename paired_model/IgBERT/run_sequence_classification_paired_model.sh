#!/bin/bash


#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=classification_igbert
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/logs/classification_igbert_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/logs/classification_igbert_%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate class_env
/home/leab/anaconda3/envs/class_env/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/sequence_classification_paired_model.py

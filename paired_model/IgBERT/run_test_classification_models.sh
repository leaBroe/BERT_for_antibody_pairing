#!/bin/bash


#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=class_adaps
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/logs/test_classification_models/test_classification_with_adapters%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/logs/test_classification_models/test_classification_with_adapters%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2
/home/leab/anaconda3/envs/adap_2/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/test_classification_models.py

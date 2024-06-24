#!/bin/bash


#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=bert2bert
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/bert2bert_test_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/bert2bert_test_%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate class_env
/home/leab/anaconda3/envs/class_env/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/src/IgBERT2IgBERT.py

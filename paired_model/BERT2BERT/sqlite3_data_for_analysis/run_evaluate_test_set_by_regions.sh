#!/bin/bash


#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=by_regions
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/by_regions_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/by_regions_%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate OAS_paired_env
/home/leab/anaconda3/envs/OAS_paired_env/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions.py

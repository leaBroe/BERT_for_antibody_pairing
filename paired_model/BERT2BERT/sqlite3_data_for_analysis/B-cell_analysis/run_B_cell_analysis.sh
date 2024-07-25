#!/bin/bash


#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=bcells
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/logs/bcells_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/logs/bcells_%j.e

#env: adap_2

eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2
/home/leab/anaconda3/envs/adap_2/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/B_cell_analysis.py




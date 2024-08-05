#!/bin/bash


#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=h2l_adaps_big_big_10_epochs
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/x_h2l_adaps_big_big_10_epochs_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/x_h2l_adaps_big_big_10_epochs_%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2
/home/leab/anaconda3/envs/adap_2/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/src/heavy2light_with_adapters.py

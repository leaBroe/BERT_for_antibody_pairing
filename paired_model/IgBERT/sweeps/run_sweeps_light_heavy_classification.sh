#!/bin/bash


#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=sweep_class_adaps
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/sweeps/logs/sweep_class_adaps%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/sweeps/logs/sweep_class_adaps%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2
/home/leab/anaconda3/envs/adap_2/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/sweeps/sweeps_light_heavy_classification.py


#!/bin/bash


#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=full_sweep_protbert
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/sweeps/logs/sweep_protbert_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/sweeps/logs/sweep_protbert_%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate bug_env 
/home/leab/anaconda3/envs/bug_env/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/sweeps/sweep_train_mlm_nsp_protbert.py

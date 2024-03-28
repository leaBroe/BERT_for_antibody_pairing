#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=run_light_model
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/light_model/outputs/run_light_model_70_pident%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/light_model/outputs/error_files/run_light_model_70_pident%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate OAS_paired_env
/home/leab/anaconda3/envs/OAS_paired_env/bin/python train_model.py


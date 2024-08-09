#!/bin/bash

#SBATCH --time=1-00:00:00
#SBATCH --mem=48G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=test_gpu
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/light_model/outputs/transformers_mlm/test_gpu_new_env2%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/light_model/outputs/transformers_mlm/test_gpu_new_env2%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate lea_env
/home/leab/anaconda3/envs/lea_env/bin/python /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/test_gpu.py

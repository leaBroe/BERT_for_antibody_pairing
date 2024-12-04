#!/bin/bash


#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=dola
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/logs/bert2gpt_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/logs/bert2gpt_%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2
CUDA_LAUNCH_BLOCKING=1 /home/leab/anaconda3/envs/adap_2/bin/python bert2gpt_with_adapters.py

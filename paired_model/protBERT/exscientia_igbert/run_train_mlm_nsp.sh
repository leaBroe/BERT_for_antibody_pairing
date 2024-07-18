#!/bin/bash


# --gres=gpu:a100:1
#SBATCH --job-name=Exsc_IgBert_nsp_mlm
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/exscientia_igbert/logs/exsc_igbert_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/exscientia_igbert/logs/exsc_igbert_%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate bug_env
/home/leab/anaconda3/envs/bug_env/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/exscientia_igbert/train_mlm_nsp.py


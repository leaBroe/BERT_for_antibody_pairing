#!/bin/bash


#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=adap_protbert_nsp
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/logs/adap_protbert_nsp_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/logs/adap_protbert_nsp_%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2
/home/leab/anaconda3/envs/adap_2/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/nsp_mlm_protbert_with_adapters.py


#!/bin/bash


#SBATCH --gres=gpu:alphafold:1
#SBATCH --job-name=h2l_no_adaps
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/igbert_mlm_small_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/igbert_mlm_small_%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2
/home/leab/anaconda3/envs/adap_2/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/src/igbert_generate_and_eval_seqs.py
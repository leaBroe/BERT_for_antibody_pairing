#!/bin/bash


#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=plabdab_human_no_vac
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/logs/beam_search_decoding_50_epochs%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/logs/beam_search_decoding_50_epochs%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2
/home/leab/anaconda3/envs/adap_2/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/src/heavy2light_with_adapters.py

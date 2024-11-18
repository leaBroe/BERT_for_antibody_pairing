#!/bin/bash


#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=full_eval
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_and_covid/logs/heavy2light_eval_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_and_covid/logs/heavy2light_eval_%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2
/home/leab/anaconda3/envs/adap_2/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/src/full_test_set_evaluation_blosum_perplexity_similarity.py

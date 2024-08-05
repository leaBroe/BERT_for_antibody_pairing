#!/bin/bash


#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=eval_test_set_perplexity_similarity_blosum
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/eval_test_set_perplexity_similarity_blosum_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/eval_test_set_perplexity_similarity_blosum_%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2
/home/leab/anaconda3/envs/adap_2/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/src/evaluate_test_set_perplexity_similarity_blosum.py

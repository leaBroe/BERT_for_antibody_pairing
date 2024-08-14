#!/bin/bash


#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=by_regions
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/h2l_div_beam_search_2_epoch_10_lr_1e-4_wd_0.1/logs/perplexity_by_regions_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/h2l_div_beam_search_2_epoch_10_lr_1e-4_wd_0.1/logs/perplexity_by_regions_%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2
/home/leab/anaconda3/envs/adap_2/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions.py

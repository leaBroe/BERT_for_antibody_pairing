#!/bin/bash


#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=experimenting_beam_search
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/experimenting/beam_search_decoding_100_epochs%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/experimenting/beam_search_decoding_100_epochs%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2
/home/leab/anaconda3/envs/adap_2/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/src/heavy2light_with_adapters.py

#!/bin/bash


#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=seaborn_plotting_mlm_model_only
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/logs/seaborn_plotting_mlm_model_only%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/logs/seaborn_plotting_mlm_model_only%j.e

eval "$(conda shell.bash hook)"
conda init bash
conda activate adap_2
/home/leab/anaconda3/envs/adap_2/bin/python /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/seaborn_plotting_mlm_model_only.py

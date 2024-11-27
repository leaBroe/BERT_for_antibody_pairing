#!/bin/bash

# # # Path to your Anaconda installation
# CONDA_HOME="/home/leab/anaconda3"

# # # Initialize conda for use in the script
# source "$CONDA_HOME/etc/profile.d/conda.sh"

# # Activate the 'abnativ' conda environment
# conda activate abnativ

# #abnativ update

# # Set LD_LIBRARY_PATH to ensure the conda environment's libraries are used
# export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
# #export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib

# # Run the abnativ command
# abnativ score \
#   -nat VKappa \
#   -i /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1/igk_true_sequences_small.fasta \
#   -odir /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1/abnativ_output \
#   -oid igk_true_sequences

# Use conda run to execute the command within the 'abnativ' environment
conda run -n abnativ abnativ score \
  -nat VKappa \
  -align \
  -i /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1/igk_true_sequences_small.fasta \
  -odir /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1/abnativ_output \
  -oid igk_true_sequences
  
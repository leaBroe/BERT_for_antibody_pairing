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

# IMPORTANT: run this in the terminal: export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" before executing this scrpt to avoid the error: libstdc++.so.6: version `GLIBCXX_3.4.26' not found
# IMPORTANT: run abnativ update to download the pretrained models if not already done

# # Use conda run to execute the command within the 'abnativ' environment
# conda run -n abnativ abnativ score \
#   -nat VKappa \
#   -align \
#   -plot \
#   -i /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1/igk_true_sequences_small.fasta \
#   -odir /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1/abnativ_output \
#   -oid small_igk_true_sequences

# kappa light variable seq from their own validation dataset, works
#conda run -n abnativ abnativ score -nat VKappa -i EIVLTQSPATLSLSPGERATLSCRASQSVSIYVVWYQQKPGQAPRLLIYDASNRATGTPARFSGSGSGTDFTLTISSLEPEDGAVYYCQQRQRWPLTFGGGTRVEIK -odir test/test_results_kappa -oid test_single_kappa -align -plot
#conda run -n abnativ abnativ score -nat VKappa -i EIVLTQSPATLSLSPGERATLSCRAS--QSVS------IYVVWYQQKPGQAPRLLIYD--------ASNRATGTPARFSGSGSG--TDFTLTISSLEPEDGAVYYCQQRQR-----------------------WPLTFGGGTRVEIK- -odir test/test_results_kappa_no_align -oid test_single_kappa_no_align -plot

# from own dataset
#conda run -n abnativ abnativ score -nat VKappa -i EIVLTQSPDFQSVTPGERVTITCRASQSIGSSLHWYQQKPDQSPKLLIKYASQSISGVPSRLSGSGSGTDFTLTITSLEAEDAATYYCHQSSSFPLTFG -odir test/test_results_kappa_own -oid test_single_kappa_own -align -plot
# arbitrarily added gaps, but with ths it works -> problem with ANARCI numbering of own light sequences
#conda run -n abnativ abnativ score -nat VKappa -i EIVLTQSPDFQSVTPGERVTITCRAS--QSIG---------------SSLHWYQQKPDQSPKLLIKY--------ASQSISGVPSRLSGSGSGT--DFTLTITSLEAEDAATY-----------------------YCHQSSSFPLTFG -odir test/test_results_kappa_own_no_align -oid test_single_kappa_own_no_align -plot



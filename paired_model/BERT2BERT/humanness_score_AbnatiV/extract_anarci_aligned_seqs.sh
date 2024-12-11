#!/bin/bash

# # used env: abnativ
#CUDA_VISIBLE_DEVICES=6, ANARCI -i /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1/igl_true_sequences.fasta --outfile /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/igl_true_sequences_anarci_aho_full.txt --scheme aho


awk -v prefix="igk_gen_seq_anarci_aho_" '
BEGIN {
    seq_num = 1;  # Initialize sequence counter
}

/^\/\// { 
    # End of a record
    if (seq != "") {
        while (length(seq) < 149) seq = seq "-";  # Pad to 149 characters
        printf ">%s%d\n%s\n", prefix, seq_num, seq;  # Output in FASTA format
        seq_num++;  # Increment sequence counter
    }
    seq = ""; next;  # Reset sequence for the next record
}

/^[A-Z]/ { 
    seq = seq $3;  # Collect sequence from the second column
}

END { 
    if (seq != "") {
        while (length(seq) < 149) seq = seq "-";  # Pad to 149 characters
        printf ">%s%d\n%s\n", prefix, seq_num, seq;  # Output in FASTA format
    }
}' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/igl_gen_sequences_anarci_aho_full.txt > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/extr_igl_gen_sequences_anarci_aho_full.fasta


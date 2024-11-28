#!/bin/bash

awk -v prefix="igk_true_seq_anarci_aho_" '
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
}' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/igk_true_sequences_anarci_aho.txt > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/igk_true_sequences_small_extracted_anarci.fasta


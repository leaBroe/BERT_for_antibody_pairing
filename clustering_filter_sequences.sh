#!/bin/bash

# Path to the full sequences FASTA file
FULL_SEQUENCES_FASTA="/ibmm_data2/oas_database/paired_lea_tmp/linclust_full_seq_all_paired/all_human_paired_full_aa_seq_no_duplicates.fasta"
# Path to the output FASTA file with filtered sequences
OUTPUT_FASTA="/ibmm_data2/oas_database/paired_lea_tmp/filtered_fasta_file/all_human_paired_full_aa_seq_filtered_4_awk.fasta"
# Path to the centroids IDs file
CENTROIDS_IDS_FILE="/ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/centroids_ids_cdr3_aa_100.txt"  

# Prepare an AWK command script to filter sequences based on IDs
AWK_SCRIPT=$(awk 'BEGIN {while (getline < "'$CENTROIDS_IDS_FILE'") ids[$1]} /^>/ {p = substr($1,2) in ids} p' "$FULL_SEQUENCES_FASTA")

# Use AWK with the prepared script to filter the sequences
echo "$AWK_SCRIPT" > "$OUTPUT_FASTA"

echo "Filtering complete. Check $OUTPUT_FASTA for the filtered sequences."


#!/bin/bash

# List of CSV files to process
ROOT_DIR="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/peak_investigation/data"
#CSV_FILES=("$ROOT_DIR/global_alignment_PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1_lower_60.csv" "$ROOT_DIR/global_alignment_PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1_betw_60_80.csv", "$ROOT_DIR/global_alignment_PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1_upper_80.csv")

CSV_FILES=("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/peak_investigation/data/global_alignment_PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1_betw_60_80.csv")

# Python function to convert amino acid to DNA
convert_aa_to_dna() {
    python3 -c "
import sys

# Codon table
codon_table = {
    'A': 'GCT', 'C': 'TGT', 'D': 'GAT', 'E': 'GAA', 'F': 'TTT', 'G': 'GGT', 'H': 'CAT', 'I': 'ATT',
    'K': 'AAA', 'L': 'TTA', 'M': 'ATG', 'N': 'AAT', 'P': 'CCT', 'Q': 'CAA', 'R': 'CGT', 'S': 'TCT',
    'T': 'ACT', 'V': 'GTT', 'W': 'TGG', 'Y': 'TAT', '*': 'TAA', 'Z': 'NNN', 'X': 'NNN', 'B': 'NNN'
}

# Input sequence
aa_sequence = sys.argv[1]

# Convert to DNA
try:
    dna_sequence = ''.join(codon_table[aa] for aa in aa_sequence)
    print(dna_sequence)
except KeyError:
    print('ERROR: Invalid amino acid found.', file=sys.stderr)
    sys.exit(1)
" "$1"
}

# Loop through each file
for INPUT_FILE in "${CSV_FILES[@]}"; do
    # Extract the base name of the file (without extension) to use in output file names
    BASE_NAME=$(basename "$INPUT_FILE" .csv)

    # Define output FASTA file names
    TRUE_SEQ_FASTA="${BASE_NAME}_true_sequences.fasta"
    GEN_SEQ_FASTA="${BASE_NAME}_generated_sequences.fasta"

    # Ensure the output files are empty
    > "$TRUE_SEQ_FASTA"
    > "$GEN_SEQ_FASTA"

    # Read the CSV file line by line
    tail -n +2 "$INPUT_FILE" | while IFS=, read -r col1 sequence_alignment_aa_light generated_sequence_light BLOSUM_score similarity perplexity calculated_blosum calculated_similarity; do
        # Remove quotes from fields (if any)
        col1=$(echo "$col1" | tr -d '"')
        sequence_alignment_aa_light=$(echo "$sequence_alignment_aa_light" | tr -d '"')
        generated_sequence_light=$(echo "$generated_sequence_light" | tr -d '"')

        # Convert amino acid sequences to DNA
        true_dna_seq=$(convert_aa_to_dna "$sequence_alignment_aa_light")
        gen_dna_seq=$(convert_aa_to_dna "$generated_sequence_light")

        # Write to true sequences FASTA
        echo ">true_seq_$col1" >> "$TRUE_SEQ_FASTA"
        echo "$true_dna_seq" >> "$TRUE_SEQ_FASTA"

        # Write to generated sequences FASTA
        echo ">gen_seq_$col1" >> "$GEN_SEQ_FASTA"
        echo "$gen_dna_seq" >> "$GEN_SEQ_FASTA"
    done
done
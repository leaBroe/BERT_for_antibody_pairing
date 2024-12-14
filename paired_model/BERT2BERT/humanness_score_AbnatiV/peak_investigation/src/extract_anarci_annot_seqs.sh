#!/bin/bash

#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=extract
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/peak_investigation/logs/extract_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/peak_investigation/logs/extract_%j.e

# Used environment: abnativ

# Input and output directories
input_dir="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/peak_investigation/anarci_numbered_files/"
output_dir="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/peak_investigation/final_fasta_files_for_abnativ/"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Process all .txt files in the input directory
for input_file in "$input_dir"/*.txt; do
    # Check if there are any .txt files in the directory
    if [[ ! -e $input_file ]]; then
        echo "No .txt files found in $input_dir"
        exit 1
    fi

    # Extract the base filename (without path and extension) for the output
    base_name=$(basename "$input_file" .txt)

    # Define the output file path
    output_file="${output_dir}/${base_name}.fasta"

    echo "Processing $input_file -> $output_file"

    # Run the AWK command for the current input file
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
        seq = seq $3;  # Collect sequence from the third column
    }
    END { 
        if (seq != "") {
            while (length(seq) < 149) seq = seq "-";  # Pad to 149 characters
            printf ">%s%d\n%s\n", prefix, seq_num, seq;  # Output in FASTA format
        }
    }' "$input_file" > "$output_file"

    # Check if AWK command succeeded
    if [[ $? -eq 0 ]]; then
        echo "Successfully processed $input_file -> $output_file"
    else
        echo "Error processing $input_file"
    fi
done
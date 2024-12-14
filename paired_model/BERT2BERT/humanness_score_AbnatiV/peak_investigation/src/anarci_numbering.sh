#!/bin/bash
# used env: abnativ

# Directory containing input FASTA files
input_dir="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/peak_investigation/fasta_for_abnativ"

# Output directory for ANARCI results
output_dir="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/peak_investigation/anarci_numbered_fasta_files"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Loop through all .fasta files in the input directory
for input_file in "$input_dir"/*.fasta; do
    # Check if there are any .fasta files in the directory
    if [[ ! -e $input_file ]]; then
        echo "No .fasta files found in $input_dir"
        exit 1
    fi

    # Extract the base name of the file (without path and extension)
    base_name=$(basename "$input_file" .fasta)

    # Define the output file path
    output_file="$output_dir/${base_name}_anarci_aho_full.txt"

    # Run the ANARCI command
    echo "Processing $input_file..."
    CUDA_VISIBLE_DEVICES=4 ANARCI -i "$input_file" --outfile "$output_file" --scheme aho

    # Check if the command was successful
    if [[ $? -eq 0 ]]; then
        echo "Successfully processed $input_file. Output saved to $output_file."
    else
        echo "Error processing $input_file."
    fi
done
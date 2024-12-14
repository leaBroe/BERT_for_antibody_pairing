#!/bin/bash

#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=abnativ
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/peak_investigation/logs/abnativ_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/peak_investigation/logs/abnativ_%j.e


# Conda environment
conda_env="abnativ"

# Input directory containing FASTA files
input_dir="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/peak_investigation/final_fasta_files_for_abnativ"  # Replace with the path to your input directory

# Output directory
output_dir="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/peak_investigation/abnativ_results"  # Replace with the path to your desired output directory

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

# Loop through all .fasta files in the input directory
for input_file in "$input_dir"/*.fasta; do
    # Check if there are any .fasta files in the directory
    if [[ ! -e $input_file ]]; then
        echo "No .fasta files found in $input_dir"
        exit 1
    fi

    # Extract the base filename (without extension) for the output identifiers
    base_name=$(basename "$input_file" .fasta)

    # Define the output identifier for this file
    output_identifier="${base_name}"

    echo "Processing $input_file with output identifier $output_identifier"

    # Run the command using conda
    conda run -n "$conda_env" abnativ score \
        -nat VLambda \
        -i "$input_file" \
        -odir "$output_dir" \
        -oid "$output_identifier"

    # Check if the command succeeded
    if [[ $? -eq 0 ]]; then
        echo "Successfully processed $input_file -> $output_dir/${output_identifier}"
    else
        echo "Error processing $input_file"
    fi
done
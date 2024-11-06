#!/bin/bash

input_file="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/conc_oas_healthy_human_plabdab_sep_seqs_only.txt"
output_file="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/conc_oas_healthy_human_plabdab_sep_seqs_only.fasta"

# Initialize the counter
counter=1

# Read the input file line by line
while IFS= read -r line
do
  # Write the identifier and sequence to the output file
  echo ">${counter}" >> "${output_file}"
  echo "${line}" >> "${output_file}"
  
  # Increment the counter
  counter=$((counter + 1))
done < "${input_file}"



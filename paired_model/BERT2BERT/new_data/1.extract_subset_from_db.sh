#!/bin/bash

#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=extr_subset
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/logs/extr_subset_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/logs/extr_subset_%j.e


# Define the database file
DATABASE="/ibmm_data2/oas_database/OAS_paired.db"

# Define the SQL query
SQL_QUERY="SELECT * FROM all_human_paired WHERE Disease = 'None' AND Vaccine = 'None';"

# Execute the SQL query and output the results
sqlite3 "$DATABASE" "$SQL_QUERY"

# Save the output to a file
sqlite3 "$DATABASE" "$SQL_QUERY" > paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/extracted_subset_full.csv

#!/bin/bash

#SBATCH --gres=gpu:h100:1
#SBATCH --job-name=extr_subset
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/logs/extr_subset_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/logs/extr_subset_%j.e


# Define the database file
DATABASE="/ibmm_data2/oas_database/OAS_paired.db"

OUTPUT_FILE="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/extracted_subset_full.csv"

# Define the SQL query
SQL_QUERY="SELECT * FROM all_human_paired WHERE Disease = 'None' AND Vaccine = 'None';"

# Run SQLite commands
sqlite3 $DATABASE <<EOF
.headers on
.mode csv
.output $OUTPUT_FILE
$SQL_QUERY
.output stdout
.quit
EOF

echo "Data extracted to $OUTPUT_FILE"

# extract the following columns from the output file
# Species, Disease, BType, Isotype_light, sequence_alignment_aa_light, sequence_alignment_light, sequence_alignment_aa_heavy, sequence_alignment_heavy, sequence_alignment_aa_full, sequence_alignment_heavy_sep_light

SQL_QUERY_2="SELECT rowid, Species, Disease, BType, Isotype_light, sequence_alignment_aa_light, sequence_alignment_light, sequence_alignment_aa_heavy, sequence_alignment_heavy, sequence_alignment_aa_full, sequence_alignment_heavy_sep_light FROM all_human_paired WHERE Disease = 'None' AND Vaccine = 'None';"

OUTPUT_FILE_REL_COLS="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/extracted_subset_relevant_columns.csv"

# Run SQLite commands
sqlite3 $DATABASE <<EOF
.headers on
.mode csv
.output $OUTPUT_FILE_REL_COLS
$SQL_QUERY_2
.output stdout
.quit
EOF

echo "Data extracted to $OUTPUT_FILE_REL_COLS"


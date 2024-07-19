#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=extract_columns_from_slite_db_analysis
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/extract_columns_from_slite_db_analysis_%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/extract_columns_from_slite_db_analysis_%j.e


# Path to the database
DATABASE_PATH="/ibmm_data2/oas_database/OAS_paired.db"

# Output CSV file
OUTPUT_FILE="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/full_paired_data_for_analysis.csv"

# Columns to extract
COLUMNS="cdr1_aa_light, cdr1_end_light, cdr1_light, cdr1_start_light, \
cdr2_aa_light, cdr2_end_light, cdr2_light, cdr2_start_light, \
cdr3_aa_light, cdr3_end_light, cdr3_light, cdr3_start_light, \
fwr1_aa_light, fwr1_end_light, fwr1_light, fwr1_start_light, \
fwr2_aa_light, fwr2_end_light, fwr2_light, fwr2_start_light, \
fwr3_aa_light, fwr3_end_light, fwr3_light, fwr3_start_light, \
fwr4_aa_light, fwr4_end_light, fwr4_light, fwr4_start_light, \
sequence_alignment_aa_light, sequence_alignment_light, sequence_alignment_heavy_sep_light"

# Run SQLite commands
sqlite3 $DATABASE_PATH <<EOF
.mode csv
.output $OUTPUT_FILE
SELECT $COLUMNS FROM all_human_paired;
.output stdout
.quit
EOF

echo "Data extracted to $OUTPUT_FILE"


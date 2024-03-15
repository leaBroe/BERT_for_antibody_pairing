#!/bin/bash

# Path to your SQLite database file
DATABASE_PATH="/ibmm_data2/oas_database/OAS_2.db"

# Output file path
OUTPUT_FILE="/ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/Bcells_subset_human_unpaired_light_cdr3_light_seq_2.txt"

# Execute the SQLite command
sqlite3 "$DATABASE_PATH" <<EOF
.mode csv
.output "$OUTPUT_FILE"
SELECT cdr3_aa, sequence_alignment_aa FROM Bcells_subset_human_unpaired_light;
.output stdout
.quit
EOF

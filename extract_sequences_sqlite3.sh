#!/bin/bash

# Define paths and table names
CENTROIDS_FILE="/ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/centroids_ids_cdrl3_aa_70_human_unpaired.txt"
DB_PATH="/ibmm_data2/oas_database/OAS_2.db"
TABLE_NAME="Bcells_subset_human_unpaired_light"
TEMP_TABLE_NAME="temp_rowids"

# Step 1: Create the temporary table and ensure it's empty
sqlite3 $DB_PATH <<SQL
DROP TABLE IF EXISTS $TEMP_TABLE_NAME;
CREATE TEMP TABLE $TEMP_TABLE_NAME (id INTEGER);
SQL

# Step 2: Import the centroids IDs into the temporary table
sqlite3 $DB_PATH <<SQL
.import $CENTROIDS_FILE $TEMP_TABLE_NAME
SQL

# Step 3: Execute the SELECT query and output the results
sqlite3 $DB_PATH <<SQL > /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/matched_rows_cdrl3_unpaired_sqlite3_1.txt
SELECT t.rowid, t.cdr3_aa, t.sequence_alignment_aa FROM $TABLE_NAME t JOIN $TEMP_TABLE_NAME temp ON t.rowid = temp.id;
SQL

echo "Query executed. Results saved to output.txt."

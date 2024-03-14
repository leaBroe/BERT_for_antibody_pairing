#!/bin/bash

# Define paths to your centroids text file and SQLite database
CENTROIDS_TXT="/path/to/your/centroids.txt"
SQLITE_DB="/path/to/your/database.sqlite"
OUTPUT_FILE="/path/to/your/output_sequences.txt"

# Prepare the output file
echo "" > "$OUTPUT_FILE"

# Read each line in the centroids text file
while IFS= read -r cdrh3_sequence
do
    # Query the SQLite database for matching full sequences
    sqlite3 "$SQLITE_DB" "SELECT sequence_id_heavy_light, sequence_alignment_aa_heavy, sequence_alignment_aa_light, sequence_alignment_aa_full, cdr3_aa_heavy FROM all_human_paired WHERE cdr3_aa_heavy = '$cdrh3_sequence';" >> "$OUTPUT_FILE"
done < "$CENTROIDS_TXT"

echo "Extraction complete. Full sequences saved to $OUTPUT_FILE."

#!/bin/bash

# /ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/all_human_paired_cdr3_aa.fasta

# Check for correct usage
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 DB"
    exit 1
fi

DB="$1"
OUTPUT_DIR="./linclust_mmseq_output" # Define the output directory
mkdir -p "$OUTPUT_DIR" # Ensure the output directory exists

# Log file creation in the output directory
log_file="$OUTPUT_DIR/${DB}_mmseqs_linclust.log"
echo "Log saved to: $log_file"
db_rows=$(grep -o ">" "$DB".fasta | wc -l)
echo "Number of sequences in $DB.fasta: $db_rows" > "$log_file"
echo "" >> "$log_file"

echo "#####################################################################################################################################" >> "$log_file"

# Process with various pident values
for pident in 70 80 90 100
do
    min_seq_id=$(echo "scale=2; $pident / 100" | bc)
    mmseqs createdb "$DB".fasta "$DB"
    mmseqs linclust "$DB" "$OUTPUT_DIR/${DB}_${pident}_clu" tmp --min-seq-id "$min_seq_id"
    mmseqs createsubdb "$OUTPUT_DIR/${DB}_${pident}_clu" "$DB" "$OUTPUT_DIR/${DB}_${pident}_clu_rep"
    mmseqs convert2fasta "$OUTPUT_DIR/${DB}_${pident}_clu_rep" "$OUTPUT_DIR/${DB}_${pident}_clu_rep.fasta"
    mmseqs createtsv "$DB" "$DB" "$OUTPUT_DIR/${DB}_${pident}_clu" "$OUTPUT_DIR/${DB}_${pident}_clu.tsv"

    awk '/^>/ { printf("\n%s,", substr($0, 2)); next; } { printf("%s", $0);} END { printf("\n"); }' "$OUTPUT_DIR/${DB}_${pident}_clu_rep.fasta" > "$OUTPUT_DIR/${DB}_${pident}_clu_rep_idseq"
    awk -F ',' '!seen[$2]++' "$OUTPUT_DIR/${DB}_${pident}_clu_rep_idseq" > "$OUTPUT_DIR/${DB}_${pident}_clu_rep_idseq_noduplicates"

    clu_rows=$(wc -l < "$OUTPUT_DIR/${DB}_${pident}_clu.tsv")
    rep_rows=$(wc -l < "$OUTPUT_DIR/${DB}_${pident}_clu_rep_idseq")
    unique_sequences=$(cut -d ',' -f 2 "$OUTPUT_DIR/${DB}_${pident}_clu_rep_idseq" | sort | uniq | grep -c "")
    rep_rows_noduplicates=$(wc -l < "$OUTPUT_DIR/${DB}_${pident}_clu_rep_idseq_noduplicates")

    echo "${pident}% CLUSTERING" >> "$log_file"
    echo "Number of rows in ${DB}_${pident}_clu.tsv: $clu_rows" >> "$log_file"
    echo "Number of rows in ${DB}_${pident}_clu_rep_idseq: $rep_rows" >> "$log_file"
    echo "Number of unique centroids sequences in the second column: $unique_sequences" >> "$log_file"
    echo "Number of rows in ${DB}_${pident}_clu_rep_idseq_noduplicates after removing duplicates: $rep_rows_noduplicates" >> "$log_file"
    echo "" >> "$log_file"
    rm -r tmp
done

echo "#####################################################################################################################################" >> "$log_file"

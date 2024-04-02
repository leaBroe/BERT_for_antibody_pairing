#!/bin/bash

#SBATCH --time=2-00:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=create_fasta_csv
#SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/logs/paired_create_fasta%j.o
#SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/logs/paired_create_fasta%j.e

# create a new column in the database that contains the full sequence alignment (heavy + light)
sqlite3 /ibmm_data2/oas_database/OAS_paired.db "ALTER TABLE all_human_paired ADD COLUMN sequence_alignment_aa_full;"
sqlite3 /ibmm_data2/oas_database/OAS_paired.db "UPDATE all_human_paired SET sequence_alignment_aa_full = sequence_alignment_aa_heavy || sequence_alignment_aa_light;"

## extract rowid and cdrh3 in csv and fasta format (for the clustering)
sqlite3 -header -csv /ibmm_data2/oas_database/OAS_paired.db "SELECT ROWID, cdr3_aa_heavy, sequence_alignment_aa_heavy, sequence_alignment_aa_light, sequence_alignment_aa_full FROM all_human_paired;" | tee >(awk -F, '{print ">"$1"\n"$2}' > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/data_oas/paired_rowid_cdrh3.fasta) > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/data_oas/paired_rowid_cdrh3_full.csv

# SELECT ROWID, sequence_alignment_aa_heavy, sequence_alignment_aa_light, sequence_alignment_aa_full, cdr3_aa_heavy FROM all_human_paired
# sqlite3 -header -csv /ibmm_data2/oas_database/OAS_heavy.db "SELECT ROWID,cdr3_aa,sequence_alignment_aa FROM Bcells_subset_human_unpaired_heavy;" | awk -F, '{print ">"$1"\n"$2}' > /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/data_from_oas/heavy_unpaired_cdrh3.fasta
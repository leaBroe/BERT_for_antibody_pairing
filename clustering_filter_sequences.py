def read_fasta_ids(fasta_file):
    ids = set()
    with open(fasta_file, 'r') as f:
        for line in f:
            if line.startswith('>'):
                id = line.strip().lstrip('>')
                ids.add(id)
    return ids

def filter_sequences_by_ids(source_fasta, target_ids, output_fasta):
    with open(source_fasta, 'r') as source, open(output_fasta, 'w') as output:
        write_sequence = False
        for line in source:
            if line.startswith('>'):
                sequence_id = line.strip().lstrip('>')
                if sequence_id in target_ids:
                    write_sequence = True
                    output.write(line)
                else:
                    write_sequence = False
            elif write_sequence:
                output.write(line)

# File paths
centroids_fasta = "/ibmm_data2/oas_database/paired_lea_tmp/linclust_mmseq2/all_human_paired_cdr3_aa_100_clu_rep.fasta"
full_sequences_fasta = "/ibmm_data2/oas_database/paired_lea_tmp/linclust_full_seq_all_paired/all_human_paired_full_aa_seq_no_duplicates.fasta"
output_fasta = "/ibmm_data2/oas_database/paired_lea_tmp/filtered_fasta_file/all_human_paired_full_aa_seq_filtered.fasta"  

# Step 1: Read IDs from the first FASTA file
centroid_ids = read_fasta_ids(centroids_fasta)

print(len(centroid_ids))

# Step 2 and 3: Filter sequences from the second FASTA file and save to a new file
#filter_sequences_by_ids(full_sequences_fasta, centroid_ids, output_fasta)

#print("Filtered sequences have been saved.")

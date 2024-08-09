
def read_ids_from_file(file_path):
    """Read IDs from a text file, one per line."""
    with open(file_path, 'r') as file:
        ids = {line.strip() for line in file}
    return ids

def filter_sequences_by_ids(source_fasta, target_ids, output_fasta):
    """Filter sequences by IDs and save them to a new FASTA file."""
    with open(source_fasta, 'r') as source, open(output_fasta, 'w') as output:
        write_sequence = False
        for line in source:
            if line.startswith('>'):
                sequence_id = line.strip().lstrip('>').split()[0]
                if sequence_id in target_ids:
                    write_sequence = True
                    output.write(line)
                else:
                    write_sequence = False
            elif write_sequence:
                output.write(line)

# File paths
centroids_ids_file = "/ibmm_data2/oas_database/paired_lea_tmp/txt_files_oas_db/centroids_ids_cdr3_aa_100.txt"
full_sequences_fasta = "/ibmm_data2/oas_database/paired_lea_tmp/linclust_full_seq_all_paired/all_human_paired_full_aa_seq_no_duplicates.fasta"
output_fasta = "/ibmm_data2/oas_database/paired_lea_tmp/filtered_fasta_file/all_human_paired_full_aa_seq_filtered_2.fasta"  

# Step 1: Read IDs from the centroids IDs file
centroid_ids = read_ids_from_file(centroids_ids_file)
#print(len(centroid_ids))

# Step 2: Filter sequences from the full sequences FASTA file and save to a new file
filter_sequences_by_ids(full_sequences_fasta, centroid_ids, output_fasta)

print("Filtered sequences have been saved to the output FASTA file.")


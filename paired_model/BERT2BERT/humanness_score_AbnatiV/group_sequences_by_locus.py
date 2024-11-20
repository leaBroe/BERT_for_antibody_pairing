import csv

# Input and output file paths
tsv_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1/true_sequences_dna.tsv'  # Replace with your actual file path
output_fasta_path = 'igk_true_sequences.fasta'

# Open the output FASTA file for writing
with open(output_fasta_path, 'w') as fasta_file:
    # Open the TSV file for reading
    with open(tsv_file_path, 'r', newline='') as tsv_file:
        # Create a CSV reader with tab delimiter
        reader = csv.DictReader(tsv_file, delimiter='\t')
        
        # Iterate over each row in the file
        for line_number, row in enumerate(reader, 2):  # Start from line 2 due to header
            # Get the 'locus' value, normalize it
            locus = row.get('locus', '').strip().upper()
            if locus == 'IGK':
                # Extract the required fields
                sequence_id = row.get('sequence_id')
                sequence_alignment_aa = row.get('sequence_alignment_aa')
                # Check if fields are present
                if sequence_id and sequence_alignment_aa:
                    # Write to FASTA file
                    fasta_file.write(f">{sequence_id}\n{sequence_alignment_aa}\n")
                else:
                    print(f"Missing 'sequence_id' or 'sequence_alignment_aa' on line {line_number}")
            else:
                # Skip records that are not 'IGK'
                continue

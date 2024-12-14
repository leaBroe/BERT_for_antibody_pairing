import csv

data_dir = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/peak_investigation/data_for_and_from_pyir/"

# List of input TSV file paths
tsv_file_paths = [
    f'{data_dir}lower_60_generated_sequences_dna.tsv',
    f'{data_dir}lower_60_true_sequences_dna.tsv',
    f'{data_dir}betw_60_80_generated_sequences_dna.tsv',
    f'{data_dir}betw_60_80_true_sequences_dna.tsv',
    f'{data_dir}upper_80_generated_sequences_dna.tsv',
    f'{data_dir}upper_80_true_sequences_dna.tsv'
]

locus = 'IGK'
locus_prefix = locus.lower()

# Loop through each file in the list
for file_number, tsv_file_path in enumerate(tsv_file_paths, start=1):
    tsv_file_id = tsv_file_path.split('/')[-1].split('.')[0]
    # Define the output FASTA file path dynamically
    output_fasta_path = f"{locus_prefix}_{tsv_file_id}.fasta"

    print(f"Processing file: {tsv_file_path}")
    print(f"Output FASTA will be saved to: {output_fasta_path}")

    try:
        # Open the output FASTA file for writing
        with open(output_fasta_path, 'w') as fasta_file:
            # Open the TSV file for reading
            with open(tsv_file_path, 'r', newline='') as tsv_file:
                # Create a CSV reader with tab delimiter
                reader = csv.DictReader(tsv_file, delimiter='\t')

                # Iterate over each row in the file
                for line_number, row in enumerate(reader, start=2):  # Start from line 2 due to header
                    # Get the 'locus' value, normalize it
                    locus = row.get('locus', '').strip().upper()
                    if locus == locus:
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
    except FileNotFoundError:
        print(f"Error: File not found: {tsv_file_path}")
    except Exception as e:
        print(f"An error occurred while processing {tsv_file_path}: {e}")
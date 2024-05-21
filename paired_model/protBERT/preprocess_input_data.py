def add_space_between_amino_acids(file_path, output_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    processed_lines = []
    for line in lines:
        # Split the line by the [SEP] token
        parts = line.strip().split('[SEP]')
        # Add spaces between amino acids for each part
        spaced_parts = [' '.join(list(part.strip())) for part in parts]
        # Join the parts back together with the [SEP] token
        processed_line = ' [SEP] '.join(spaced_parts)
        processed_lines.append(processed_line)

    # Write the processed lines to the output file
    with open(output_path, 'w') as file:
        for processed_line in processed_lines:
            file.write(processed_line + '\n')

# Usage
input_file = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_val_no_ids_small.txt'
output_file = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_val_no_ids_small_SPACE_separated.txt'
add_space_between_amino_acids(input_file, output_file)

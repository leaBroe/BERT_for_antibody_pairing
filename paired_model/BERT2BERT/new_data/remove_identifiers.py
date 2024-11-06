def remove_identifiers(input_fasta, output_fasta):
    with open(input_fasta, 'r') as infile, open(output_fasta, 'w') as outfile:
        for line in infile:
            if not line.startswith('>'):
                outfile.write(line)

# Example usage
input_fasta = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/train_test_val_datasets/human_healthy_no_vac_allocated_sep_test.txt'
output_fasta = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/train_test_val_datasets/human_healthy_no_vac_allocated_sep_test_no_identifiers.txt'
remove_identifiers(input_fasta, output_fasta)
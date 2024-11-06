def remove_identifiers(input_fasta, output_fasta):
    with open(input_fasta, 'r') as infile, open(output_fasta, 'w') as outfile:
        for line in infile:
            if not line.startswith('>'):
                outfile.write(line)

# do it for train val and test sets in a loop
for set_name in ['train', 'val', 'test']:
    input_fasta = f'/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/train_val_test_datasets/plabdab_human_healthy_no_vac_allocated_{set_name}.txt'
    output_fasta = f'/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/train_val_test_datasets/plabdab_human_healthy_no_vac_allocated_{set_name}_no_identifiers.txt'
    remove_identifiers(input_fasta, output_fasta)

# input_fasta = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/extracted_subset_relevant_columns_no_dupl_sep.fasta"
# output_fasta = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/extracted_subset_relevant_columns_no_dupl_sep_no_identifiers.fasta"

#remove_identifiers(input_fasta, output_fasta)
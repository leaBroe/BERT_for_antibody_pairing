import pandas as pd

# Read the CSV file
file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/heavy_model_vdj_gene_analysis/heavy_vdj_genes_paired_oas_db_test_set_extraction_no_dupl.txt'
df = pd.read_csv(file_path, na_filter=False)

def add_spaces_to_aa(sequence):
    parts = sequence.split('[SEP]')
    heavy = ' '.join(parts[0])
    light = ' '.join(parts[1])
    return f"{heavy} [SEP] {light}"

# Apply the function to the last column
df['sequence_alignment_heavy_sep_light'] = df['sequence_alignment_heavy_sep_light'].apply(add_spaces_to_aa)

# Save the modified DataFrame to a new CSV file
output_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/heavy_model_vdj_gene_analysis/spaces_heavy_vdj_genes_paired_oas_db_test_set_extraction_no_dupl.txt'
df.to_csv(output_file_path, index=False)

print(f"Processed file saved to {output_file_path}")

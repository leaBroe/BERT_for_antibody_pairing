import pandas as pd

# Load the CSV file
#input_csv_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/heavy_model_vdj_gene_analysis/relevant_cols_spaces_heavy_vdj_genes_paired_oas_db_test_set_extraction_no_dupl.txt'  
#input_csv_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/extracted_seqs_light_model_test_set_no_dupl.txt"
input_csv_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/extracted_seqs_heavy_model_test_set_80000.txt"
df = pd.read_csv(input_csv_path)

# Extract the required columns and create new columns with fewer values
# df['v_call_fewer_heavy'] = df['v_call_heavy'].str.split('-').str[0]
df['d_call_fewer'] = df['d_call'].str.split('-').str[0]

df['v_call_fewer'] = df['v_call'].str.split('-').str[0]
#df['j_call_fewer'] = df['j_call'].str.split('-').str[0]

# For j_call_fewer_heavy, remove everything after the '*' including '*'
df['j_call_fewer'] = df['j_call'].str.split('-').str[0].str.split('*').str[0]

# Add another column where you put everything before the '*' in the column v_call_heavy
df['v_call_fewer_star'] = df['v_call'].str.split('*').str[0]

# # Select the columns to save
# output_df = df[['v_call_fewer_heavy', 'd_call_fewer_heavy', 'j_call_fewer_heavy', 
#                 'v_call_fewer_heavy_star', 'v_call_heavy', 'd_call_heavy', 'j_call_heavy', 
#                 'sequence_alignment_heavy_sep_light']]

# Select the columns to save
output_df = df[['v_call_fewer', 'j_call_fewer', 'd_call_fewer',
                'v_call_fewer_star', 'v_call', 'd_call', 'j_call', 
                'sequence_alignment_aa']]

# Save the result to a new CSV file
output_csv_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/umap_tsne_pca_heavy_light_models/fewer_genes_extracted_seqs_heavy_model_test_set_80000.txt"
output_df.to_csv(output_csv_path, index=False)

print("New CSV file saved successfully!")
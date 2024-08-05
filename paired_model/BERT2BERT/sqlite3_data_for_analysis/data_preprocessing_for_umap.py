import pandas as pd

# Load the first CSV file
file1 = pd.read_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/full_test_data/full_test_data_extraction_with_header.txt')
# Load the second CSV file
file2 = pd.read_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/extracted_data_from_PyIR/umap_extraction_from_pyir_full_data.csv')

# Merge the dataframes on the sequence_alignment_aa_light/sequence_alignment_aa columns
merged_df = pd.merge(file1, file2, on='sequence_alignment_aa', how='inner')

# Select the desired columns
final_df = merged_df[['sequence_alignment_heavy_sep_light', 'locus']]

# Save the result to a new CSV file
final_df.to_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_for_umap_pca/full_data_heavyseplight_locus.csv', index=False)

print("Merged CSV file has been created successfully.")


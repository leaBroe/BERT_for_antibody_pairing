import pandas as pd

# Read the files
file_one_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_for_umap_pca/full_data_heavyseplight_locus_no_dupl_spaces_rm.csv'
file_two_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/kappa_lambda_analysis/umap_k_l_subtypes/full_data_heavyseplight_locus_v_family_no_dupl_spaces_rm2.csv'

# Assuming the files are CSV formatted
df1 = pd.read_csv(file_one_path)
df2 = pd.read_csv(file_two_path)

# Extract the first two columns
df1_columns = df1.iloc[:, :2]
df2_columns = df2.iloc[:, :2]

# Find the mismatched lines
mismatched_lines = df1_columns[df1_columns != df2_columns].dropna()

# Output the result
if mismatched_lines.empty:
    print("The first two columns in both files are exactly the same.")
else:
    print("The first two columns in both files are not the same.")
    print("Mismatched lines:")
    print(mismatched_lines)


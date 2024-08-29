import pandas as pd

# Load the CSV file
df = pd.read_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_diseases_subgroups_analysis/full_test_data_extraction_species_diseases_no_dupl.txt')

# Filter rows where BType is either 'Memory-B-Cells' or 'Naive-B-Cells'
filtered_df = df[df['BType'].isin(['Memory-B-Cells', 'Naive-B-Cells'])]

# Save the filtered data to a new CSV file
filtered_df.to_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_diseases_subgroups_analysis/filtered_mem_naive_full_test_data_extraction_species_diseases_no_dupl.csv', index=False)
import pandas as pd

df1 = pd.read_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/plabdab_paired_sequences_full_seqs_no_dupl.csv')
df2 = pd.read_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/extracted_subset_relevant_columns_no_dupl.csv')

# # Sample data for testing
# data1 = {
#     'sep_full_sequence': ['AAA', 'BBB', 'CCC', 'DDD', 'EEE']
# }
# data2 = {
#     'sequence_alignment_heavy_sep_light': ['CCC', 'DDD', 'FFF', 'GGG', 'AAA']
# }

# # Convert dictionaries to DataFrames
# df1 = pd.DataFrame(data1)
# df2 = pd.DataFrame(data2)

# Extract the relevant columns
col1 = df1['sep_full_sequence'].dropna()
col2 = df2['sequence_alignment_heavy_sep_light'].dropna()

# Check if each value in col1 exists anywhere in col2
matches_in_col1 = col1.isin(col2)
matches_in_col2 = col2.isin(col1)

# Count the number of matches
count_matches_in_col1 = matches_in_col1.sum()
count_matches_in_col2 = matches_in_col2.sum()

# Calculate the percentage of values from each column found in the other
percentage_matches_in_col1 = (count_matches_in_col1 / len(col1)) * 100
percentage_matches_in_col2 = (count_matches_in_col2 / len(col2)) * 100

print(f"Entries in col1 found anywhere in col2: {count_matches_in_col1} ({percentage_matches_in_col1}%)")
print(f"Entries in col2 found anywhere in col1: {count_matches_in_col2} ({percentage_matches_in_col2}%)")

# concatenate the two dataframes 
df = pd.concat([df1, df2], axis=1)

# add a new column to the dataframe with a unique ID (numbering ascending from 1)
df['unique_id'] = range(1, len(df) + 1)
# put unique_id as the first column
df = df[['unique_id'] + [col for col in df.columns if col != 'unique_id']]

# Save the updated DataFrame to a new CSV file
df.to_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/plabdab_paired_sequences_full_seqs_no_dupl_with_human_healthy_no_vac_from_oas.csv', index=False)

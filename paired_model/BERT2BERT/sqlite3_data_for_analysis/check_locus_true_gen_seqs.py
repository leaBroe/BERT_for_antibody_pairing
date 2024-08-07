import pandas as pd

# # Load data into pandas DataFrame
# df = pd.read_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/full_test_set_true_gen_seqs_all_relevant_cols.csv')

# # Extract the numeric part and type from sequence_id for sorting
# df['sequence_id_num'] = df['sequence_id'].str.extract('(\d+)').astype(int)
# df['sequence_type'] = df['sequence_id'].str.extract('(True|Generated)')

# # Sort the DataFrame by sequence_id_num and sequence_type
# df = df.sort_values(by=['sequence_id_num', 'sequence_type'], ascending=[True, True])

# # Reset the index
# df.reset_index(drop=True, inplace=True)

# # save the results to a CSV file
# df.to_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/full_test_set_true_gen_seqs_all_relevant_cols_sorted.csv', index=False)

# df = pd.read_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/full_test_set_true_gen_seqs_all_relevant_cols_sorted.csv')

# # extract sequence_id sequence_alignment_aa and locus from df
# df_relevant_cols = df[['sequence_id', 'sequence_alignment_aa', 'locus']]

# # save the results to a CSV file
# df_relevant_cols.to_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/full_test_set_true_gen_seqs_relevant_cols_locus.csv', index=False)


# Load the CSV data into a DataFrame
data = pd.read_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/full_test_set_true_gen_seqs_relevant_cols_locus.csv')

# Check if the locus values are the same for each pair
# Assuming the pairs are in sequence in the file (Generated followed by True)
results = []
for i in range(0, len(data), 2):
    generated_seq = data.iloc[i]
    true_seq = data.iloc[i+1]
    same_locus = generated_seq['locus'] == true_seq['locus']
    results.append({
        'generated_sequence_id': generated_seq['sequence_id'],
        'true_sequence_id': true_seq['sequence_id'],
        'same_locus': same_locus
    })

# Convert results to DataFrame for better visualization
results_df = pd.DataFrame(results)

# Calculate the average of 'True' in the 'same_locus' column
average_true = results_df['same_locus'].mean()

# Print the results DataFrame and the average
print(results_df)
print("Average of 'True' in same_locus column:", average_true)


import pandas as pd

# # Read the first file
# file1_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/full_eval_heavy2light_with_adapters125463.o'

# # Read the second file
# file2_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/Btypes_full_paired_test_data_no_dupl.csv'

# with open(file1_path, 'r') as file:
#     lines = file.readlines()

# # Parse the first file
# data1 = []
# seq_pair = {}
# for line in lines:
#     line = line.strip()
#     if line.startswith('Sequence pair'):
#         if seq_pair:
#             data1.append(seq_pair)
#         seq_pair = {}
#         seq_pair['Sequence pair'] = line.split(' ')[2][:-1]
#     elif line.startswith('True Sequence:'):
#         seq_pair['True Sequence'] = line.split(' ')[2:]
#     elif line.startswith('Generated Sequence:'):
#         seq_pair['Generated Sequence'] = line.split(' ')[2:]
#     elif line.startswith('BLOSUM Score:'):
#         seq_pair['BLOSUM Score'] = float(line.split(' ')[2])
#     elif line.startswith('Similarity Percentage:'):
#         seq_pair['Similarity Percentage'] = float(line.split(' ')[2][:-1])
#     elif line.startswith('Perplexity:'):
#         seq_pair['Perplexity'] = float(line.split(' ')[1])

# if seq_pair:
#     data1.append(seq_pair)

# # Convert to DataFrame
# df1 = pd.DataFrame(data1)
# df1['True Sequence'] = df1['True Sequence'].apply(lambda x: ''.join(x))
# df1['Generated Sequence'] = df1['Generated Sequence'].apply(lambda x: ''.join(x))

# # Rename 'True Sequence' to 'sequence_alignment_aa_light'
# df1 = df1.rename(columns={'True Sequence': 'sequence_alignment_aa_light'})

# print(df1.head())
# print(len(df1))

# # Read the second file
# df2 = pd.read_csv(file2_path)

# # Remove spaces from sequences in df2
# df2['sequence_alignment_aa_light'] = df2['sequence_alignment_aa_light'].str.replace(' ', '')

# # Ensure there are no duplicates in the key columns
# df2 = df2.drop_duplicates(subset=['sequence_alignment_heavy_sep_light'])

# print(df2.head())
# print(len(df2))

# # Perform the join operation
# merged_df = pd.merge(df1, df2, on='sequence_alignment_aa_light', how='inner')

# # print all column names of the merged DataFrame
# print(merged_df.columns)

# # Select final columns and format the DataFrame
# final_df = merged_df[['BType', 'sequence_alignment_aa_light', 'sequence_alignment_heavy_sep_light', 'Generated Sequence', 'BLOSUM Score', 'Similarity Percentage', 'Perplexity']]

# # Save the final DataFrame
# output_csv_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/B_cell_analysis_output.csv'
# final_df.to_csv(output_csv_path, index=False)

# # Display the final DataFrame (for verification)
# final_df.head()

# Read the output file into a DataFrame
output_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/B_cell_analysis_output_no_dupl.csv'
df = pd.read_csv(output_file_path)

# # Group by BType and calculate the mean for Perplexity, BLOSUM Score, and Similarity Percentage, and count the number of sequences
# grouped_df = df.groupby('BType').agg({
#     'Perplexity': 'mean',
#     'BLOSUM Score': 'mean',
#     'Similarity Percentage': 'mean',
#     'sequence_alignment_aa_light': 'count'
# }).reset_index()

print(df.groupby('BType')[['Perplexity', 'BLOSUM Score', 'Similarity Percentage']].mean())

# # Rename columns for clarity
# grouped_df.columns = ['BType', 'Average Perplexity', 'Average BLOSUM Score', 'Average Similarity Percentage', 'Number of Sequences']

# # Format the averages to 2 decimal places
# grouped_df['Average Perplexity'] = grouped_df['Average Perplexity'].map('{:.2f}'.format)
# grouped_df['Average BLOSUM Score'] = grouped_df['Average BLOSUM Score'].map('{:.2f}'.format)
# grouped_df['Average Similarity Percentage'] = grouped_df['Average Similarity Percentage'].map('{:.2f}'.format)

# # Save the resulting DataFrame to a CSV file
# output_csv_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/grouped_statistics_by_btype.csv'
# grouped_df.to_csv(output_csv_path, index=False)

# print(f"Grouped statistics saved to {output_csv_path}")


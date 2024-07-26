import pandas as pd

# File paths
file1_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/full_eval_heavy2light_with_adapters125463.o'
file4_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/kappa_lambda_analysis/kappa_lambda_subtypes_extraction.csv'
additional_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/Btypes_full_paired_test_data_no_dupl.csv'

# Parse the first file
with open(file1_path, 'r') as file:
    lines = file.readlines()

data1 = []
seq_pair = {}
for line in lines:
    line = line.strip()
    if line.startswith('Sequence pair'):
        if seq_pair:
            data1.append(seq_pair)
        seq_pair = {}
        seq_pair['Sequence pair'] = line.split(' ')[2][:-1]
    elif line.startswith('True Sequence:'):
        seq_pair['True Sequence'] = ''.join(line.split(' ')[2:])
    elif line.startswith('Generated Sequence:'):
        seq_pair['Generated Sequence'] = ''.join(line.split(' ')[2:])
    elif line.startswith('BLOSUM Score:'):
        seq_pair['BLOSUM Score'] = float(line.split(' ')[2])
    elif line.startswith('Similarity Percentage:'):
        seq_pair['Similarity Percentage'] = float(line.split(' ')[2][:-1])
    elif line.startswith('Perplexity:'):
        seq_pair['Perplexity'] = float(line.split(' ')[1])

if seq_pair:
    data1.append(seq_pair)

# Convert to DataFrame
df1 = pd.DataFrame(data1)

print(len(df1))

# Read the fourth file
df4 = pd.read_csv(file4_path)
df4['sequence_alignment_aa'] = df4['sequence_alignment_aa'].str.replace(' ', '')

print(len(df4))

# Read the additional file
df_additional = pd.read_csv(additional_file_path)

# Extract the light sequence after the [SEP] token and remove spaces
df_additional['Light Sequence'] = df_additional['sequence_alignment_heavy_sep_light'].apply(lambda x: x.split('[SEP]')[-1].replace(' ', ''))

print(len(df_additional))

# Merge df1 and df4 on True Sequence and sequence_alignment_aa
merged_df = pd.merge(df1, df_additional, left_on='True Sequence', right_on='Light Sequence', how='inner')

print(len(merged_df))

merged_df.drop_duplicates(subset=['sequence_alignment_heavy_sep_light'])

print(len(merged_df))   

# # Merge with df_additional on True Sequence and Light Sequence
# final_merged_df = pd.merge(merged_df, df_additional, left_on='True Sequence', right_on='Light Sequence', how='inner')

# final_merged_df.drop_duplicates(subset=['sequence_alignment_heavy_sep_light'])

# print(len(final_merged_df))

# # Select final columns and rename for clarity
# final_df = final_merged_df[['True Sequence', 'Generated Sequence', 'BLOSUM Score', 'Similarity Percentage', 'Perplexity', 'v_family', 'sequence_alignment_heavy_sep_light']]
# final_df = final_df.rename(columns={
#     'True Sequence': 'True Sequence',
#     'Generated Sequence': 'Generated Sequence',
#     'BLOSUM Score': 'BLOSUM Score',
#     'Similarity Percentage': 'Similarity Percentage',
#     'Perplexity': 'Perplexity',
#     'v_family': 'Subtype',
#     'sequence_alignment_heavy_sep_light': 'sequence_alignment_heavy_sep_light'
# })

# # Save the final DataFrame
# output_csv_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/kappa_lambda_analysis/kappa_lambda_subtypes_analysis_output.csv'
# final_df.to_csv(output_csv_path, index=False)

# # Display the final DataFrame (for verification)
# final_df.head()


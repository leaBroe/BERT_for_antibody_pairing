import pandas as pd

# Read the first file
file1_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/full_eval_heavy2light_with_adapters125463.o'

# Read the second file
file2_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/Btypes_full_paired_test_data_no_dupl.csv'

# Read the new file
file3_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/kappa_lambda_analysis/full_data_heavyseplight_locus_no_dupl_spaces_rm.csv'

with open(file1_path, 'r') as file:
    lines = file.readlines()

# Parse the first file
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
        seq_pair['True Sequence'] = line.split(' ')[2:]
    elif line.startswith('Generated Sequence:'):
        seq_pair['Generated Sequence'] = line.split(' ')[2:]
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
df1['True Sequence'] = df1['True Sequence'].apply(lambda x: ''.join(x))
df1['Generated Sequence'] = df1['Generated Sequence'].apply(lambda x: ''.join(x))

# Rename 'True Sequence' to 'sequence_alignment_aa_light'
df1 = df1.rename(columns={'True Sequence': 'sequence_alignment_aa_light'})

# Read the second file
df2 = pd.read_csv(file2_path)

# Remove spaces from sequences in df2
df2['sequence_alignment_aa_light'] = df2['sequence_alignment_aa_light'].str.replace(' ', '')

# Ensure there are no duplicates in the key columns
df1 = df1.drop_duplicates(subset=['sequence_alignment_aa_light'])
df2 = df2.drop_duplicates(subset=['sequence_alignment_aa_light'])

# Merge the two DataFrames based on 'sequence_alignment_aa_light'
merged_df1 = pd.merge(df1, df2, on='sequence_alignment_aa_light', how='inner')

# Select final columns and format the DataFrame
final_df1 = merged_df1[['BType', 'sequence_alignment_aa_light', 'Generated Sequence', 'BLOSUM Score', 'Similarity Percentage', 'Perplexity']]

# Read the third file
df3 = pd.read_csv(file3_path)

# Split the sequences and remove spaces
df3['Light Sequence'] = df3['sequence_alignment_heavy_sep_light'].apply(lambda x: x.split('[SEP]')[-1].replace(' ', ''))

# Merge the DataFrames based on 'sequence_alignment_aa_light' and 'Light Sequence'
merged_df2 = pd.merge(final_df1, df3, left_on='sequence_alignment_aa_light', right_on='Light Sequence', how='inner')

# Group by 'locus' and compute the required statistics
grouped_df = merged_df2.groupby('locus').agg({
    'BLOSUM Score': 'mean',
    'Similarity Percentage': 'mean',
    'Perplexity': 'mean',
    'sequence_alignment_aa_light': 'count'
}).reset_index()

print(merged_df2.groupby('locus')[['BLOSUM Score', 'Similarity Percentage', 'Perplexity']].median())

# Rename columns for clarity
grouped_df.columns = ['locus', 'Average BLOSUM Score', 'Average Similarity Percentage', 'Average Perplexity', 'Number of Sequences']

# Save the final DataFrame
output_csv_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/kappa_lambda_analysis/kappa_lambda_analysis_output.csv'
grouped_df.to_csv(output_csv_path, index=False)

# Display the final DataFrame (for verification)
grouped_df.head()


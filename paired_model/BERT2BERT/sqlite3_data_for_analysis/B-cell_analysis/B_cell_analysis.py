import pandas as pd

# Read the first file
file1_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/full_eval_heavy2light_with_adapters125463.o'
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

# Read the second file
file2_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/Btypes_full_paired_test_data_no_dupl.csv'
df2 = pd.read_csv(file2_path)

# Keep relevant columns and remove spaces from sequences
df2 = df2[['BType', 'sequence_alignment_aa_light', 'sequence_alignment_light',
           'sequence_alignment_aa_heavy', 'sequence_alignment_heavy', 
           'sequence_alignment_heavy_sep_light']]
df2['sequence_alignment_aa_light'] = df2['sequence_alignment_aa_light'].str.replace(' ', '')
df2['sequence_alignment_aa_heavy'] = df2['sequence_alignment_aa_heavy'].str.replace(' ', '')

# Function to match sequences between the two dataframes
def match_sequences(row, df2):
    true_seq = row['True Sequence']
    match = df2[(df2['sequence_alignment_aa_light'] == true_seq) | 
                (df2['sequence_alignment_aa_heavy'] == true_seq)]
    if not match.empty:
        return match.iloc[0]['BType']
    return None

# Add BType to df1
df1['BType'] = df1.apply(lambda row: match_sequences(row, df2), axis=1)

# Drop rows without matching BType
df1 = df1.dropna(subset=['BType'])

# Reorder and select final columns
final_df = df1[['BType', 'True Sequence', 'Generated Sequence', 'BLOSUM Score', 'Similarity Percentage', 'Perplexity']]
final_df.head()

# Save the final DataFrame

final_df.to_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/B_cell_analysis_output.csv', index=False)


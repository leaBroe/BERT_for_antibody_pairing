import pandas as pd

# Read the CSV file
df = pd.read_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/plabdab_paired_sequences.csv')

# Concatenate heavy_sequence and light_sequence to create full_sequence
df['full_sequence'] = df['heavy_sequence'] + df['light_sequence']

# Concatenate heavy_sequence and light_sequence with a [SEP] token in between
df['sep_full_sequence'] = df['heavy_sequence'] + '[SEP]' + df['light_sequence']

# Save the updated DataFrame to a new CSV file
df.to_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/plabdab_paired_sequences_full_seqs.csv', index=False)
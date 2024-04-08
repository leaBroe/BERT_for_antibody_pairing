"""
This code processes a TSV dataset containing cluster and sequence IDs. 
It extract the clusters from the clustered 50% pident data (*cluster.tsv), sorts them by size, and allocates sequences to training, validation, and test sets based on specified percentages.
The script logs information about the dataset, cluster count, and sequence count.
It then populates train_sequences, val_sequences, and test_sequences lists while maintaining entire clusters in each set. Bigger cluster are kept for the training set.
The log includes counts of sequences allocated to each set (iga and igg) and the intended sizes.
"""
import random
import subprocess
from collections import defaultdict
import argparse
import logging

# Set up command line argument parsing
parser = argparse.ArgumentParser(description='')
parser.add_argument("--tsv_dataset", help="Path to the TSV dataset file", type=str)
parser.add_argument("--rep_fasta_file", help="Path to the representative FASTA file", type=str)
parser.add_argument("--prefix", help="Prefix for output files", type=str, default='CDRH3')

args = parser.parse_args()

# Extract arguments
main_dataset = args.tsv_dataset
rep_fasta_file = args.rep_fasta_file
data_prefix = args.prefix

# Configure logging
logging.basicConfig(filename='train_val_test_split.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

logging.info(f'DATASET: {main_dataset}')

# Function to retrieve sequences using awk
def retrieve_fasta_sequences(id_file, fasta_file, output_file):
    command = f"""awk 'NR==FNR {{ ids[$1]=1; next }} /^>/ {{ f=substr($0,2) in ids }} f' {id_file} {fasta_file} > {output_file}"""
    subprocess.run(command, shell=True, check=True)


# Load the TSV dataset and create a dictionary of clusters
cluster_data = defaultdict(list)
with open(main_dataset, 'r') as tsv_file:
    for line in tsv_file:
        cluster_id, sequence_id = line.strip().split('\t')
        cluster_data[cluster_id].append(sequence_id)

logging.info("Total clusters (# centroids): %d", len(cluster_data))

# Sort clusters by size in descending order
seed = 42  # Set your seed for reproducibility
random.seed(seed)
sorted_clusters = sorted(cluster_data.items(), key=lambda x: len(x[1]), reverse=True)

# Set percentages for train, validation, and test sets
train_percentage = 0.8
val_percentage = 0.1
test_percentage = 0.1

# Count total number of sequences
line_count = sum(len(v) for v in cluster_data.values())
logging.info(f'Total sequences: {line_count}')

# Calculate number of sequences for each set
n_train_sequences = line_count * train_percentage
n_val_sequences = line_count * val_percentage
n_test_sequences = line_count * test_percentage

# Allocate sequences to each set, keeping entire clusters together
train_sequences, val_sequences, test_sequences = [], [], []
for cluster_id, sequences in sorted_clusters:
    if len(train_sequences) + len(sequences) <= n_train_sequences:
        train_sequences.extend(sequences)
    elif len(val_sequences) + len(sequences) <= n_val_sequences:
        val_sequences.extend(sequences)
    else:
        test_sequences.extend(sequences)

# Logging the distribution
logging.info('TRAIN: Number of sequences %d', len(train_sequences))
logging.info('VAL: Number of sequences %d', len(val_sequences))
logging.info('TEST: Number of sequences %d', len(test_sequences))

# Function to write IDs to file
def write_to_file(filename, data):
    with open(filename, 'w') as f:
        for d in data:
            f.write(f'{d}\n')

# Prepare file names
train_ids = f'{data_prefix}_ids_train.txt'
val_ids = f'{data_prefix}_ids_val.txt'
test_ids = f'{data_prefix}_ids_test.txt'
train_file = f'{data_prefix}_train.txt'
val_file = f'{data_prefix}_val.txt'
test_file = f'{data_prefix}_test.txt'

# Write IDs to files
write_to_file(train_ids, train_sequences)
write_to_file(val_ids, val_sequences)
write_to_file(test_ids, test_sequences)

# Retrieve the sequences for each set
retrieve_fasta_sequences(train_ids, rep_fasta_file, train_file)
retrieve_fasta_sequences(val_ids, rep_fasta_file, val_file)
retrieve_fasta_sequences(test_ids, rep_fasta_file, test_file)

logging.info('Sequence retrieval completed.')



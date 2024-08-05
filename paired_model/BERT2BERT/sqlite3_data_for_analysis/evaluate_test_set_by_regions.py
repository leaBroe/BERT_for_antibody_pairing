# env: OAS_paired_env
from Bio import pairwise2
from Bio.Align import substitution_matrices
from crowelab_pyir import PyIR
import pandas as pd
import re

# Codon table for reverse translation (simplified, using common codons)
codon_table = {
    'A': 'GCT', 'C': 'TGT', 'D': 'GAT', 'E': 'GAA', 'F': 'TTT', 'G': 'GGT', 'H': 'CAT', 'I': 'ATT',
    'K': 'AAA', 'L': 'TTA', 'M': 'ATG', 'N': 'AAT', 'P': 'CCT', 'Q': 'CAA', 'R': 'CGT', 'S': 'TCT',
    'T': 'ACT', 'V': 'GTT', 'W': 'TGG', 'Y': 'TAT', '*': 'TAA'
}

# Function to reverse translate protein sequence to DNA
def protein_to_dna(protein_seq):
    return "".join([codon_table[aa] for aa in protein_seq])

def extract_sequences(file_path):
    data = []
    with open(file_path, 'r') as file:
        content = file.read()

    # Regular expression pattern to match the sequence pairs
    pattern = re.compile(
        r"Sequence pair \d+:\s*True Sequence: ([A-Z ]+)\s*Generated Sequence: ([A-Z ]+)",
        re.MULTILINE
    )

    # Find all matches in the file content
    matches = pattern.findall(content)

    # Extract the true and generated sequences, remove spaces
    for match in matches:
        true_sequence = match[0].replace(" ", "")
        generated_sequence = match[1].replace(" ", "")
        data.append({
            "true_sequence": true_sequence,
            "generated_sequence": generated_sequence
        })

    return data

# Example usage
file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/full_eval_heavy2light_with_adapters125463.o' 
data = extract_sequences(file_path)

# Step 1 & 2: Extract sequences and convert to DNA
fasta_entries = []
for i, pair in enumerate(data):
    true_seq_dna = protein_to_dna(pair['true_sequence'].replace(" ", ""))
    generated_seq_dna = protein_to_dna(pair['generated_sequence'].replace(" ", ""))
    
    # Create FASTA entries
    fasta_entries.append((f">True_Seq_{i+1}", true_seq_dna))
    fasta_entries.append((f">Generated_Seq_{i+1}", generated_seq_dna))

# Step 3: Write to FASTA file
fasta_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/sequences.fasta"
with open(fasta_file, "w") as f:
    for header, sequence in fasta_entries:
        f.write(f"{header}\n{sequence}\n")

# Step 4: Use PyIR to identify regions (example placeholder code)
# Assuming PyIR can parse the FASTA file and annotate the regions
# This is a simplified example, replace with actual PyIR usage
regions_results = []
for i, (true_header, generated_header) in enumerate(fasta_entries):
    # Replace this with actual PyIR parsing logic
    # PyIR would return something like:
    true_regions = {
        "FWR1": "GCTTATCAGGTCACCA",  # Example DNA sequence
        "CDR1": "GTTGCT",  # Example DNA sequence
        # ... other regions
    }
    generated_regions = {
        "FWR1": "GCTTATCAGGTCACCC",  # Example DNA sequence
        "CDR1": "GTTGCC",  # Example DNA sequence
        # ... other regions
    }
    regions_results.append((true_regions, generated_regions))

# Step 5: Calculate similarity and BLOSUM scores
blosum62 = substitution_matrices.load("BLOSUM62")
results = []

def calculate_blosum_score(true_seq, generated_seq, matrix):
    score = 0
    matches = 0
    min_length = min(len(true_seq), len(generated_seq))

    for i in range(min_length):
        pair = (true_seq[i], generated_seq[i])
        if pair in matrix:
            score += matrix[pair]
        elif (pair[1], pair[0]) in matrix:
            score += matrix[(pair[1], pair[0])]
        if true_seq[i] == generated_seq[i]:
            matches += 1

    similarity_percentage = (matches / min_length) * 100
    return score, min_length, matches, similarity_percentage

for true_regions, generated_regions in regions_results:
    for region in true_regions.keys():
        score, min_length, matches, similarity_percentage = calculate_blosum_score(
            true_regions[region], generated_regions[region], blosum62
        )
        results.append({
            "Region": region,
            "True Sequence": true_regions[region],
            "Generated Sequence": generated_regions[region],
            "BLOSUM Score": score,
            "Similarity Percentage": similarity_percentage
        })

# Convert results to DataFrame and display
results_df = pd.DataFrame(results)
print(results_df)


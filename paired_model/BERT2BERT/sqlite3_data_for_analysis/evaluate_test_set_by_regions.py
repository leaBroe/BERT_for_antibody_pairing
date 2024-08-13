# env: OAS_paired_env
from Bio import pairwise2
from Bio.Align import substitution_matrices
from crowelab_pyir import PyIR
import pandas as pd
import re


# Amino Acid Table:
# | Amino Acid                    | Three Letter Code | One Letter Code |
# |-------------------------------|-------------------|-----------------|
# | Alanine                       | Ala               | A               |
# | Arginine                      | Arg               | R               |
# | Asparagine                    | Asn               | N               |
# | Aspartic Acid                 | Asp               | D               |
# | Cysteine                      | Cys               | C               |
# | Glutamic Acid                 | Glu               | E               |
# | Glutamine                     | Gln               | Q               |
# | Glycine                       | Gly               | G               |
# | Histidine                     | His               | H               |
# | Isoleucine                    | Ile               | I               |
# | Leucine                       | Leu               | L               |
# | Lysine                        | Lys               | K               |
# | Methionine                    | Met               | M               |
# | Phenylalanine                 | Phe               | F               |
# | Proline                       | Pro               | P               |
# | Serine                        | Ser               | S               |
# | Threonine                     | Thr               | T               |
# | Tryptophan                    | Trp               | W               |
# | Tyrosine                      | Tyr               | Y               |
# | Valine                        | Val               | V               |


# Codon table for reverse translation (simplified, using common codons)
codon_table = {
    'A': 'GCT', 'C': 'TGT', 'D': 'GAT', 'E': 'GAA', 'F': 'TTT', 'G': 'GGT', 'H': 'CAT', 'I': 'ATT',
    'K': 'AAA', 'L': 'TTA', 'M': 'ATG', 'N': 'AAT', 'P': 'CCT', 'Q': 'CAA', 'R': 'CGT', 'S': 'TCT',
    'T': 'ACT', 'V': 'GTT', 'W': 'TGG', 'Y': 'TAT', '*': 'TAA'
}

# 1. Extract sequences from the log file and convert them to DNA because PyIR requires DNA sequences and cannot handle protein sequences.
# Here I am using a simplified codon table to reverse translate the protein sequences to DNA sequences, because there are multiple possible codons for some amino acids.
# This is not ideal, but there is no direct way to convert protein sequences to DNA sequences without ambiguity.

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

# input: log file of the form:
#
# Sequence pair 67209:
# True Sequence: D I Q V T Q S P S S L S A S I G D R V T I T C Q A S Q D I S D N L N W Y Q Q K P G K V P K L L I Y D A S N L Q T G V P S R F S G S G S G T Y F S V T I S S L Q P E D I A T Y Y C Q S Y G K F R P R T F G Q G T K L E I K
# Generated Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A S Q S I S S Y L N W Y Q Q K P G K A P K L L I Y A A S S L Q S G V P S R F S G S G S G T D F T L T I S S L Q P E D F A T Y Y C Q Q S Y S T P R T F G Q G T K V E I K
# BLOSUM Score: 355.0
# Similarity Percentage: 70.09345794392523%
# Perplexity: 2.4618451595306396

# #file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/full_eval_heavy2light_with_adapters125463.o' 
# file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/full_eval_full_diverse_beam_search_2_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1_127798.o"
# data = extract_sequences(file_path)

# # Step 1 & 2: Extract sequences and convert to DNA
# fasta_entries = []
# for i, pair in enumerate(data):
#     true_seq_dna = protein_to_dna(pair['true_sequence'].replace(" ", ""))
#     generated_seq_dna = protein_to_dna(pair['generated_sequence'].replace(" ", ""))
    
#     # Create FASTA entries
#     fasta_entries.append((f">True_Seq_{i+1}", true_seq_dna))
#     fasta_entries.append((f">Generated_Seq_{i+1}", generated_seq_dna))

# # Step 3: Write to FASTA file
# # output file:
# fasta_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/h2l_div_beam_search_2_epoch_10_lr_1e-4_wd_0.1/true_gen_sequences_in_DNA.fasta"
# with open(fasta_file, "w") as f:
#     for header, sequence in fasta_entries:
#         f.write(f"{header}\n{sequence}\n")

#fasta_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/sequences.fasta"
fasta_file="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/h2l_div_beam_search_2_epoch_10_lr_1e-4_wd_0.1/true_gen_sequences_in_DNA.fasta"

# Step 4: Use PyIR to identify regions 
pyirfile = PyIR(query=fasta_file, args=['--outfmt', 'tsv'])
pyir_result = pyirfile.run()

# # Step 5: Calculate similarity and BLOSUM scores
# blosum62 = substitution_matrices.load("BLOSUM62")
# results = []


# def calculate_blosum_score(true_seq, generated_seq, matrix):
#     if not true_seq or not generated_seq:
#         raise ValueError("True sequence or generated sequence is empty.")
    
#     score = 0
#     matches = 0
#     min_length = min(len(true_seq), len(generated_seq))

#     for i in range(min_length):
#         pair = (true_seq[i], generated_seq[i])
#         if pair in matrix:
#             score += matrix[pair]
#         elif (pair[1], pair[0]) in matrix:
#             score += matrix[(pair[1], pair[0])]
#         if true_seq[i] == generated_seq[i]:
#             matches += 1

#     similarity_percentage = (matches / min_length) * 100 if min_length > 0 else 0
#     return score, min_length, matches, similarity_percentage


# # Read the CSV file into a DataFrame
# df = pd.read_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/full_test_set_true_gen_seqs_relevant_cols.csv')

# # Define regions to process
# regions = ['fwr1_aa', 'cdr1_aa', 'fwr2_aa', 'cdr2_aa', 'fwr3_aa', 'cdr3_aa', 'fwr4_aa']

# # List to store results
# results = []

# # Iterate over the DataFrame in pairs (True, Generated)
# for i in range(0, len(df), 2):
#     true_seq_row = df.iloc[i]
#     generated_seq_row = df.iloc[i + 1]

#     result_entry = {'sequence_id': true_seq_row['sequence_id']}
    
#     for region in regions:
#         true_seq = true_seq_row[region]
#         generated_seq = generated_seq_row[region]

#         # Handle missing values by setting them to empty strings
#         if pd.isna(true_seq):
#             true_seq = ""
#         if pd.isna(generated_seq):
#             generated_seq = ""
        
#         # Skip if both sequences are empty
#         if not true_seq and not generated_seq:
#             print(f"Both sequences are empty for region {region} in sequence pair {true_seq_row['sequence_id']}. Skipping.")
#             continue
        
#         try:
#             blosum_score, min_length, matches, similarity_percentage = calculate_blosum_score(true_seq, generated_seq, blosum62)
#         except ValueError as e:
#             print(f"Error processing sequences {true_seq_row['sequence_id']} in region {region}: {e}")
#             continue
        
#         result_entry[f'{region}_blosum_score'] = blosum_score
#         result_entry[f'{region}_min_length'] = min_length
#         result_entry[f'{region}_matches'] = matches
#         result_entry[f'{region}_similarity_percentage'] = similarity_percentage
    
#     results.append(result_entry)

# # Convert results to DataFrame for analysis or export
# results_df = pd.DataFrame(results)

# # save the results to a CSV file
# results_df.to_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/region_similarity_scores.csv', index=False)


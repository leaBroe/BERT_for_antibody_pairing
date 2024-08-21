# env: OAS_paired_env
# env: adap_2
from Bio import pairwise2
from Bio.Align import substitution_matrices
#from crowelab_pyir import PyIR
import pandas as pd
import re
from transformers import EncoderDecoderModel, AutoTokenizer, GenerationConfig
import torch
from adapters import init
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from math import exp
from crowelab_pyir import PyIR



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


def initialize_model_and_tokenizer(model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name):
    """
    Initialize the model, tokenizer, and generation configuration.
    
    Args:
        model_path (str): Path to the model.
        tokenizer_path (str): Path to the tokenizer.
        adapter_path (str): Path to the adapter.
        generation_config_path (str): Path to the generation configuration.
        device (torch.device): Device to run the model on.
    
    Returns:
        model (EncoderDecoderModel): Initialized model.
        tokenizer (AutoTokenizer): Initialized tokenizer.
        generation_config (GenerationConfig): Generation configuration.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = EncoderDecoderModel.from_pretrained(model_path)
    model.to(device)
    init(model)
    print(f"model is on device: {model.device}")
    model.to(device)
    model.load_adapter(adapter_path)
    model.set_active_adapters(adapter_name)
    generation_config = GenerationConfig.from_pretrained(generation_config_path)
    
    return model, tokenizer, generation_config


# # heavy2light 10 epochs diverse beam search beam = 2
# run_name="full_diverse_beam_search_2_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1"
# model_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_diverse_beam_search_2_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1"
# tokenizer_path = f"{model_path}/checkpoint-84010"
# adapter_path = f"{model_path}/final_adapter"
# generation_config_path = model_path
# adapter_name = "heavy2light_adapter"

# Paths and device configuration
run_name = "full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1"
model_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1"
tokenizer_path = f"{model_path}/checkpoint-504060"
adapter_path = f"{model_path}/final_adapter"
generation_config_path = model_path
adapter_name = "heavy2light_adapter"


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model, tokenizer, generation_config = initialize_model_and_tokenizer(model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name)


# Codon table for reverse translation (simplified, using common codons)
# codon_table = {
#     'A': 'GCT', 'C': 'TGT', 'D': 'GAT', 'E': 'GAA', 'F': 'TTT', 'G': 'GGT', 'H': 'CAT', 'I': 'ATT',
#     'K': 'AAA', 'L': 'TTA', 'M': 'ATG', 'N': 'AAT', 'P': 'CCT', 'Q': 'CAA', 'R': 'CGT', 'S': 'TCT',
#     'T': 'ACT', 'V': 'GTT', 'W': 'TGG', 'Y': 'TAT', '*': 'TAA'
# }

# # 1. Extract sequences from the log file and convert them to DNA because PyIR requires DNA sequences and cannot handle protein sequences.
# # Here I am using a simplified codon table to reverse translate the protein sequences to DNA sequences, because there are multiple possible codons for some amino acids.
# # This is not ideal, but there is no direct way to convert protein sequences to DNA sequences without ambiguity.

# # Function to reverse translate protein sequence to DNA
# def protein_to_dna(protein_seq):
#     return "".join([codon_table[aa] for aa in protein_seq])


# def extract_sequences(file_path):
#     data = []
#     with open(file_path, 'r') as file:
#         content = file.read()

#     # Regular expression pattern to match the sequence pairs
#     pattern = re.compile(
#         r"Sequence pair \d+:\s*True Sequence: ([A-Z ]+)\s*Generated Sequence: ([A-Z ]+)",
#         re.MULTILINE
#     )

#     # Find all matches in the file content
#     matches = pattern.findall(content)

#     # Extract the true and generated sequences, remove spaces
#     for match in matches:
#         true_sequence = match[0].replace(" ", "")
#         generated_sequence = match[1].replace(" ", "")
#         data.append({
#             "true_sequence": true_sequence,
#             "generated_sequence": generated_sequence
#         })

#     return data

# # input: log file of the form:
# #
# # Sequence pair 67209:
# # True Sequence: D I Q V T Q S P S S L S A S I G D R V T I T C Q A S Q D I S D N L N W Y Q Q K P G K V P K L L I Y D A S N L Q T G V P S R F S G S G S G T Y F S V T I S S L Q P E D I A T Y Y C Q S Y G K F R P R T F G Q G T K L E I K
# # Generated Sequence: D I Q M T Q S P S S L S A S V G D R V T I T C R A S Q S I S S Y L N W Y Q Q K P G K A P K L L I Y A A S S L Q S G V P S R F S G S G S G T D F T L T I S S L Q P E D F A T Y Y C Q Q S Y S T P R T F G Q G T K V E I K
# # BLOSUM Score: 355.0
# # Similarity Percentage: 70.09345794392523%
# # Perplexity: 2.4618451595306396

# #file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/full_eval_heavy2light_with_adapters125463.o' 
# file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/full_eval_full_diverse_beam_search_2_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1_127798.o"
# data = extract_sequences(file_path)

# # save the extracted sequences to a CSV file
# df = pd.DataFrame(data)
# df.to_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/h2l_div_beam_search_2_epoch_10_lr_1e-4_wd_0.1/full_test_set_true_gen_aa_seqs.csv', index=False)


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

# #fasta_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/sequences.fasta"
# fasta_file="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/h2l_div_beam_search_2_epoch_10_lr_1e-4_wd_0.1/true_gen_sequences_in_DNA.fasta"

# # Step 4: Use PyIR to identify regions 
# pyirfile = PyIR(query=fasta_file, args=['--outfmt', 'tsv'])
# pyir_result = pyirfile.run()

# # Step 5: Calculate similarity and BLOSUM scores
# blosum62 = substitution_matrices.load("BLOSUM62")
# results = []


# Read the CSV file into a DataFrame
#df = pd.read_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/full_test_set_true_gen_seqs_relevant_cols.csv')
df = pd.read_csv("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/h2l_div_beam_search_2_epoch_10_lr_1e-4_wd_0.1/full_test_set_true_gen_seqs_all_relevant_cols.csv")

# Initialize the model and tokenizer
model, tokenizer, generation_config = initialize_model_and_tokenizer(model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name)

# Input sequence (heavy chain)
input_sequence = "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVGVIWYDGSKKYYSDSVKGRFTISRDSPNNMLYLQMNSLRAEDTAVYFCARDDDGSNQYGIFEYWGQGTVVTVSS"

# Tokenize the input sequence
inputs = tokenizer(input_sequence, padding="max_length", truncation=True, max_length=512, return_tensors="pt").to(device)

# Generate the sequence using the model
generated_seq = model.generate(
    input_ids=inputs.input_ids, 
    attention_mask=inputs.attention_mask, 
    max_length=150, 
    output_scores=True, 
    return_dict_in_generate=True, 
    generation_config=generation_config
)

# Decode the generated sequence to get the text
sequence = generated_seq["sequences"][0]
generated_text = tokenizer.decode(sequence, skip_special_tokens=True)

# Remove spaces
generated_text = generated_text.replace(" ", "")
print(f"Generated Sequence without spaces: {generated_text}")

# Function to convert amino acid sequence to DNA
def amino_acid_to_dna(aa_sequence):
    codon_table = {
        'A': 'GCT', 'C': 'TGT', 'D': 'GAT', 'E': 'GAA', 'F': 'TTT', 'G': 'GGT', 'H': 'CAT', 'I': 'ATT',
        'K': 'AAA', 'L': 'TTA', 'M': 'ATG', 'N': 'AAT', 'P': 'CCT', 'Q': 'CAA', 'R': 'CGT', 'S': 'TCT',
        'T': 'ACT', 'V': 'GTT', 'W': 'TGG', 'Y': 'TAT', '*': 'TAA'
    }
    dna_sequence = ''.join(codon_table[aa] for aa in aa_sequence)
    return dna_sequence

# Convert the generated sequence to DNA
dna_sequence = amino_acid_to_dna(generated_text)
print(f"Converted DNA Sequence: {dna_sequence}")

# Write the DNA sequence to a FASTA file
fasta_filename = 'example.fasta'

with open(fasta_filename, 'w') as fasta_file:
    fasta_file.write(f">Generated_Sequence\n{dna_sequence}\n")


# Use the FASTA file as the query for PyIR
pyirfile = PyIR(query=fasta_filename, args=['--outfmt', 'dict'])
pyir_result = pyirfile.run()

print(f"PyIR Result: {pyir_result}")
print(f"PyIR Result Type: {type(pyir_result)}")

sequence_regions = {
    "fwr1": pyir_result['Generated_Sequence']['fwr1_aa'],
    "cdr1": pyir_result['Generated_Sequence']['cdr1_aa'],
    "fwr2": pyir_result['Generated_Sequence']['fwr2_aa'],
    "cdr2": pyir_result['Generated_Sequence']['cdr2_aa'],
    "fwr3": pyir_result['Generated_Sequence']['fwr3_aa'],
    "cdr3": pyir_result['Generated_Sequence']['cdr3_aa'],
    "fwr4": pyir_result['Generated_Sequence']['fwr4_aa']
}

# Calculate the lengths of each region
region_lengths = {region: len(seq) for region, seq in sequence_regions.items()}

# Decode the generated sequence to get the text
sequence = generated_seq["sequences"][0]
generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
print(f"Generated Text: {generated_text}")

# Extract logits
logits = generated_seq.scores  

# Logits shape should be (sequence_length, vocab_size) after concatenation
logits_tensor = torch.stack(logits, dim=1)
print(f"Logits shape: {logits_tensor.shape}") # Shape: [2, sequence_length, vocab_size] -> 2 sequences are generated because num beams = 2 in the generation config
# Access the logits for the best sequence (usually the first one in the beam search results)
best_logits = logits_tensor[0]  # Shape: [sequence_length, vocab_size]

# Convert the logits for the best sequence to probabilities
probs = F.softmax(best_logits, dim=-1)

print(f"Probabilities shape: {probs.shape}")

# Example: Print probabilities for the first token in the sequence
print(f"Probabilities for the first token: {probs[0]}")

# # Define the sequence regions
# sequence_regions = {
#     "fwr1": "DIQMTQSPSSLSASVGDRVTFTCRSS",
#     "cdr1": "QNIGIY",
#     "fwr2": "LNWYQQKPGRAPTVLIY",
#     "cdr2": "TAS",
#     "fwr3": "SLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYFC",
#     "cdr3": "QQSYSLPYT",
#     "fwr4": "FGQGARLQIK"
# }

# Calculate the lengths of each region
#region_lengths = {region: len(seq) for region, seq in sequence_regions.items()}

# To keep track of the start index of each region
start_idx = 1 # 1 bc of cls token (not 0)

# Initialize cumulative length
cumulative_length = 0 # 1 bc of cls token (not 0)

# Iterate over each region to calculate perplexity
perplexities = {}

for region, length in region_lengths.items():
    # Update the cumulative length including the current region
    cumulative_length += length
    
    if region == "fwr4": # since PyIr is not fully recognizing the fwr4 region, we have to manually set the end index to the end of the sequence
        end_idx = len(probs) - 1  # Exclude the last token (sep token)
        region_length = end_idx - start_idx
    else:
        end_idx = start_idx + length
        region_length = length
        
    
    # Extract the probabilities for the current region
    region_probs = probs[start_idx:end_idx]
    
    # Get the indices of the most likely predictions for the current region
    correct_indices = region_probs.argmax(dim=-1)
    
    # Extract the correct probabilities
    correct_probs = region_probs[range(region_length), correct_indices]
    
    # Calculate the perplexity for the current region using cumulative length
    n = cumulative_length  # Include the lengths of all preceding regions
    perplexity = torch.prod(correct_probs ** (-1 / n)).item()
    
    # Store the perplexity for the current region
    perplexities[region] = perplexity
    
    # Update the start index for the next region
    start_idx = end_idx

# Print perplexities for each region
for region, perplexity in perplexities.items():
    print(f"Perplexity for {region}: {perplexity}")

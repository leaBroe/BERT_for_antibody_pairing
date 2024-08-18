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


# heavy2light 10 epochs diverse beam search beam = 2
run_name="full_diverse_beam_search_2_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1"
model_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_diverse_beam_search_2_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1"
tokenizer_path = f"{model_path}/checkpoint-84010"
adapter_path = f"{model_path}/final_adapter"
generation_config_path = model_path
adapter_name = "heavy2light_adapter"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model, tokenizer, generation_config = initialize_model_and_tokenizer(model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name)


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

#file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/full_eval_heavy2light_with_adapters125463.o' 
file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/full_eval_full_diverse_beam_search_2_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1_127798.o"
data = extract_sequences(file_path)

# save the extracted sequences to a CSV file
df = pd.DataFrame(data)
df.to_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/h2l_div_beam_search_2_epoch_10_lr_1e-4_wd_0.1/full_test_set_true_gen_aa_seqs.csv', index=False)


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


# Read the CSV file into a DataFrame
#df = pd.read_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/full_test_set_true_gen_seqs_relevant_cols.csv')
df = pd.read_csv("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/h2l_div_beam_search_2_epoch_10_lr_1e-4_wd_0.1/full_test_set_true_gen_seqs_all_relevant_cols.csv")


# Define regions to process
regions = ['fwr1_aa']
#regions = ['fwr1_aa', 'cdr1_aa', 'fwr2_aa', 'cdr2_aa', 'fwr3_aa', 'cdr3_aa', 'fwr4_aa']


def calculate_perplexity(model, tokenizer, generated_seq, true_seq, device):
    """
    Calculate the perplexity of a generated sequence against the true sequence.
    
    Args:
        model (torch.nn.Module): The trained model.
        tokenizer (AutoTokenizer): The tokenizer used for encoding sequences.
        generated_seq (str): The generated sequence.
        true_seq (str): The true sequence.
        device (torch.device): The device to run the calculations on.
    
    Returns:
        float: The perplexity score.
    """
    inputs = tokenizer(generated_seq, padding=True, truncation=True, return_tensors="pt").to(device)
    targets = tokenizer(true_seq, padding=True, truncation=True, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(input_ids=inputs.input_ids, decoder_input_ids=targets.input_ids)
    
    logits = outputs.logits
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = targets.input_ids[:, 1:].contiguous()
    
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
    target_mask = (shift_labels != tokenizer.pad_token_id).float()
    loss = loss.view(shift_labels.size()) * target_mask
    
    log_likelihood = loss.sum(dim=1)
    perplexity = torch.exp(log_likelihood / target_mask.sum(dim=1)).cpu().detach().numpy()
    
    return perplexity[0]

perplexity_results = []

for i in range(0, len(df), 2):
    true_seq_row = df.iloc[i]
    generated_seq_row = df.iloc[i + 1]

    perplexity_entry = {'sequence_id': true_seq_row['sequence_id']}
    
    for region in regions:
        true_seq = true_seq_row[region]
        generated_seq = generated_seq_row[region]

        # Remove empty rows: Skip the region if either sequence is NaN or empty
        if pd.isnull(true_seq) or pd.isnull(generated_seq) or not true_seq.strip() or not generated_seq.strip():
            print(f"Empty sequence for region {region} in sequence pair {true_seq_row['sequence_id']}. Skipping.")
            continue
        
        try:
            perplexity = calculate_perplexity(model, tokenizer, generated_seq, true_seq, device)
        except Exception as e:
            print(f"Error processing sequences {true_seq_row['sequence_id']} in region {region}: {e}")
            continue
        
        perplexity_entry[f'{region}_perplexity'] = perplexity
    
    if len(perplexity_entry) > 1:  # Only append if there's at least one region's perplexity calculated
        perplexity_results.append(perplexity_entry)


# Convert results to DataFrame for analysis or export
perplexity_df = pd.DataFrame(perplexity_results)

# save the results to a CSV file
perplexity_df.to_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/h2l_div_beam_search_2_epoch_10_lr_1e-4_wd_0.1/perplexity_by_region_test.csv', index=False)

# save the mean perplexities of each region to a CSV file
mean_perplexity_df = pd.DataFrame([{'region': region, 'mean_perplexity': perplexity_df[f'{region}_perplexity'].mean()} for region in regions])
mean_perplexity_df.to_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/h2l_div_beam_search_2_epoch_10_lr_1e-4_wd_0.1/mean_perplexity_by_region_test.csv', index=False)

# save the median perplexities of each region to a CSV file
median_perplexity_df = pd.DataFrame([{'region': region, 'median_perplexity': perplexity_df[f'{region}_perplexity'].median()} for region in regions])
median_perplexity_df.to_csv('/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/h2l_div_beam_search_2_epoch_10_lr_1e-4_wd_0.1/median_perplexity_by_region_test.csv', index=False)


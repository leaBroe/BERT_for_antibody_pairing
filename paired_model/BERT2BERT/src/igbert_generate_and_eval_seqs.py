from transformers import BertTokenizer, BertForMaskedLM
import torch
import torch.nn.functional as F
from Bio import pairwise2
from Bio.Align import substitution_matrices

# Load the tokenizer and the MLM model
tokenizer = BertTokenizer.from_pretrained("Exscientia/IgBert", do_lower_case=False)
model = BertForMaskedLM.from_pretrained("Exscientia/IgBert")

# Heavy and light chain sequences
sequences_heavy = [
    "VQLAQSGSELRKPGASVKVSCDTSGHSFTSNAIHWVRQAPGQGLEWMGWINTDTGTPTYAQGFTGRFVFSLDTSARTAYLQISSLKADDTAVFYCARERDYSDYFFDYWGQGTLVTVSS",
    "QVQLVESGGGVVQPGRSLRLSCAASGFTFSNYAMYWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRTEDTAVYYCASGSDYGDYLLVYWGQGTLVTVSS"
]

sequences_light = [
    "EVVMTQSPASLSVSPGERATLSCRARASLGISTDLAWYQQRPGQAPRLLIYGASTRATGIPARFSGSGSGTEFTLTISSLQSEDSAVYYCQQYSNWPLTFGGGTKVEIK",
    "ALTQPASVSGSPGQSITISCTGTSSDVGGYNYVSWYQQHPGKAPKLMIYDVSKRPSGVSNRFSGSKSGNTASLTISGLQSEDEADYYCNSLTSISTWVFGGGTKLTVL"
]

# Format the sequences with spaces between each amino acid
def format_sequence_with_spaces(sequence):
    return ' '.join(sequence)

formatted_sequences_heavy = [format_sequence_with_spaces(seq) for seq in sequences_heavy]
formatted_sequences_light = [format_sequence_with_spaces(seq) for seq in sequences_light]

# Prepare masked sequences: mask the entire light chain
paired_sequences = []
for sequence_heavy, sequence_light in zip(formatted_sequences_heavy, formatted_sequences_light):
    masked_light = ' '.join(['[MASK]'] * len(sequence_light.split()))
    paired_sequences.append(sequence_heavy + ' [SEP] ' + masked_light)

# Tokenize the sequences
tokens = tokenizer.batch_encode_plus(
    paired_sequences,
    add_special_tokens=True,
    pad_to_max_length=True,
    return_tensors="pt"
)

# Generate predictions for masked tokens
outputs = model(
    input_ids=tokens['input_ids'],
    attention_mask=tokens['attention_mask']
)

# Get the predictions for the masked positions
predicted_ids = torch.argmax(outputs.logits, dim=-1)

# Initialize variables for storing metrics
predicted_light_sequences = []
perplexities = []
global_similarities = []
global_blosum_scores = []
hard_similarities = []
hard_blosum_scores = []

# Load the BLOSUM62 matrix
substitution_matrix = substitution_matrices.load('BLOSUM62')

# Function to calculate perplexity
def calculate_perplexity(logits, labels, mask):
    logits = logits[mask]
    labels = labels[mask]
    log_likelihood = F.cross_entropy(logits, labels, reduction='sum')
    perplexity = torch.exp(log_likelihood / mask.sum())
    return perplexity.item()


# # Function to calculate pseudo-perplexity
# def calculate_pseudo_perplexity(model, tokenizer, sequence):
#     token_ids = tokenizer.encode(sequence, return_tensors='pt')
#     input_length = token_ids.size(1)
#     log_likelihood = 0.0
#
#     for i in range(input_length):
#         # Create a copy of the token IDs
#         masked_token_ids = token_ids.clone()
#         # Mask a token that we will try to predict back
#         masked_token_ids[0, i] = tokenizer.mask_token_id
#
#         with torch.no_grad():
#             output = model(masked_token_ids)
#             logit_prob = torch.nn.functional.log_softmax(output.logits, dim=-1)
#
#         # Accumulate the log likelihood for the true token
#         log_likelihood += logit_prob[0, i, token_ids[0, i]]
#
#     # Calculate the average log likelihood per token
#     avg_log_likelihood = log_likelihood / input_length
#
#     # Compute and return the pseudo-perplexity
#     pseudo_perplexity = torch.exp(-avg_log_likelihood)
#     return pseudo_perplexity.item()

# Function to calculate global alignment similarity
def calculate_global_similarity(seq1, seq2):
    alignments = pairwise2.align.globalxx(seq1, seq2)
    top_alignment = alignments[0]
    return top_alignment[2] / len(seq2) * 100  # Similarity score as a fraction of the true sequence length

# Function to calculate global alignment BLOSUM score
def calculate_global_blosum_score(seq1, seq2):
    alignment = pairwise2.align.globalds(seq1, seq2, substitution_matrix, -10, -0.5)[0]
    return alignment[2]

# Load sequences from the input text file
# Load small test data
input_file = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_test_no_ids_space_separated_SMALL.txt'

# load FULL test data
#input_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_test_no_ids_space_separated.txt"

with open(input_file, "r") as file:
    lines = file.readlines()

# Process and evaluate each sequence
for line in lines:
    # Strip any surrounding whitespace and split the heavy and light sequences
    line = line.strip()
    if "[SEP]" not in line:
        print(f"Skipping invalid line: {line}")
        continue

    # Tokenize the line
    tokens = tokenizer.batch_encode_plus([line], add_special_tokens=True, return_tensors="pt")

    # Generate predictions for masked tokens
    outputs = model(input_ids=tokens['input_ids'], attention_mask=tokens['attention_mask'])
    predicted_ids = torch.argmax(outputs.logits, dim=-1)

    # Extract the token IDs after the [SEP] token and decode them
    sep_token_id = tokenizer.convert_tokens_to_ids('[SEP]')
    sep_indices = (tokens['input_ids'][0] == sep_token_id).nonzero(as_tuple=True)[0]
    if sep_indices.numel() > 0:
        sep_index = sep_indices[0].item()  # Take the first occurrence of [SEP]

        # Extract the light chain part
        predicted_light_ids = predicted_ids[0, sep_index + 1:].tolist()
        predicted_light_tokens = tokenizer.convert_ids_to_tokens(predicted_light_ids)

        # Join the tokens to form the light sequence
        predicted_light_sequence = ''.join(predicted_light_tokens).replace('##', '').replace(' ', '')

        # Extract the true light sequence from the line
        true_light_sequence = line.split("[SEP]")[1].replace(' ', '')

        # Calculate perplexity
        mask = (tokens['input_ids'][0] != tokenizer.pad_token_id)
        perplexity = calculate_perplexity(outputs.logits[0], tokens['input_ids'][0], mask)

        # Calculate global alignment similarity
        global_similarity = calculate_global_similarity(predicted_light_sequence, true_light_sequence)

        # Calculate global alignment BLOSUM score
        global_blosum_score = calculate_global_blosum_score(predicted_light_sequence, true_light_sequence)

        # Calculate hard BLOSUM score and hard similarity
        hard_blosum_score, hard_similarity = calculate_hard_blosum_and_similarity(true_light_sequence, predicted_light_sequence, substitution_matrix)

        # Output the results
        print(f"Predicted light sequence: {predicted_light_sequence}")
        print(f"  Perplexity: {perplexity}")
        print(f"  Global Alignment Similarity: {global_similarity:.4f}")
        print(f"  Global Alignment BLOSUM score: {global_blosum_score}")
        print(f"  Hard Similarity: {hard_similarity:.2f}%")
        print(f"  Hard BLOSUM score: {hard_blosum_score}")
    else:
        print(f"Skipping line due to missing [SEP]: {line}")


        
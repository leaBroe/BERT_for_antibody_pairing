from transformers import BertTokenizer, BertForMaskedLM
import torch
import torch.nn.functional as F
from Bio import pairwise2
from Bio.Align import substitution_matrices

# Load the tokenizer and the MLM model
tokenizer = BertTokenizer.from_pretrained("Exscientia/IgBert", do_lower_case=False)
model = BertForMaskedLM.from_pretrained("Exscientia/IgBert")

# # Heavy and light chain sequences
# sequences_heavy = [
#     "VQLAQSGSELRKPGASVKVSCDTSGHSFTSNAIHWVRQAPGQGLEWMGWINTDTGTPTYAQGFTGRFVFSLDTSARTAYLQISSLKADDTAVFYCARERDYSDYFFDYWGQGTLVTVSS",
#     "QVQLVESGGGVVQPGRSLRLSCAASGFTFSNYAMYWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRTEDTAVYYCASGSDYGDYLLVYWGQGTLVTVSS"
# ]

# sequences_light = [
#     "EVVMTQSPASLSVSPGERATLSCRARASLGISTDLAWYQQRPGQAPRLLIYGASTRATGIPARFSGSGSGTEFTLTISSLQSEDSAVYYCQQYSNWPLTFGGGTKVEIK",
#     "ALTQPASVSGSPGQSITISCTGTSSDVGGYNYVSWYQQHPGKAPKLMIYDVSKRPSGVSNRFSGSKSGNTASLTISGLQSEDEADYYCNSLTSISTWVFGGGTKLTVL"
# ]

# # Format the sequences with spaces between each amino acid
# def format_sequence_with_spaces(sequence):
#     return ' '.join(sequence)

# formatted_sequences_heavy = [format_sequence_with_spaces(seq) for seq in sequences_heavy]
# formatted_sequences_light = [format_sequence_with_spaces(seq) for seq in sequences_light]

# # Prepare masked sequences: mask the entire light chain
# paired_sequences = []
# for sequence_heavy, sequence_light in zip(formatted_sequences_heavy, formatted_sequences_light):
#     masked_light = ' '.join(['[MASK]'] * len(sequence_light.split()))
#     paired_sequences.append(sequence_heavy + ' [SEP] ' + masked_light)

# # Tokenize the sequences
# tokens = tokenizer.batch_encode_plus(
#     paired_sequences,
#     add_special_tokens=True,
#     pad_to_max_length=True,
#     return_tensors="pt"
# )

# # Generate predictions for masked tokens
# outputs = model(
#     input_ids=tokens['input_ids'],
#     attention_mask=tokens['attention_mask']
# )

# # Get the predictions for the masked positions
# predicted_ids = torch.argmax(outputs.logits, dim=-1)

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


# Function to filter out invalid characters (non-amino acids) from sequences
def filter_sequence(sequence):
    valid_amino_acids = set("LAGVESIKRDTPNQFYMHCWXUBZO")  # vocab of IgBERT
    return ''.join([char for char in sequence if char in valid_amino_acids])

# Function to calculate global alignment similarity
def calculate_global_similarity(seq1, seq2):
    seq1 = filter_sequence(seq1)
    seq2 = filter_sequence(seq2)
    alignments = pairwise2.align.globalxx(seq1, seq2)
    top_alignment = alignments[0]
    return top_alignment[2] / len(seq2) * 100  # Similarity score as a fraction of the true sequence length

# Function to calculate global alignment BLOSUM score
def calculate_global_blosum_score(seq1, seq2):
    seq1 = filter_sequence(seq1)
    seq2 = filter_sequence(seq2)
    alignment = pairwise2.align.globalds(seq1, seq2, substitution_matrix, -10, -0.5)[0]
    return alignment[2]

# Function to calculate hard BLOSUM score and hard similarity
def calculate_hard_blosum_and_similarity(true_seq, generated_seq, matrix):
    true_seq = filter_sequence(true_seq)
    generated_seq = filter_sequence(generated_seq)
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
    return score, similarity_percentage

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


# Load sequences from the input text file
# Load small test data
input_file = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_test_no_ids_space_separated_SMALL.txt'

# load FULL test data
#input_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_test_no_ids_space_separated.txt"

#input_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_test_no_ids_space_separated_2000_lines.txt"

with open(input_file, "r") as file:
    lines = file.readlines()

# Initialize accumulators for averaging metrics
total_perplexity = 0.0
total_global_similarity = 0.0
total_global_blosum_score = 0.0
total_hard_similarity = 0.0
total_hard_blosum_score = 0.0
num_sequences = 0

# Process and evaluate each sequence
for line in lines:
    # Strip any surrounding whitespace and split the heavy and light sequences
    line = line.strip()
    if "[SEP]" not in line:
        print(f"Skipping invalid line: {line}")
        continue

    # Split heavy and light sequences
    heavy_seq, light_seq = line.split("[SEP]")
    
    # Tokenize the sequences
    heavy_tokens = tokenizer(heavy_seq, add_special_tokens=False, return_tensors="pt")
    light_tokens = tokenizer(light_seq, add_special_tokens=False, return_tensors="pt")
    
    # Create masked input where the entire light chain is masked
    masked_light_tokens = torch.full_like(light_tokens['input_ids'], tokenizer.mask_token_id)
    masked_input = torch.cat([heavy_tokens['input_ids'], torch.tensor([[tokenizer.sep_token_id]]), masked_light_tokens], dim=1)
    
    # Create attention mask
    attention_mask = torch.cat([heavy_tokens['attention_mask'], torch.tensor([[1]]), light_tokens['attention_mask']], dim=1)
    
    # Pass the masked input to the model
    outputs = model(input_ids=masked_input, attention_mask=attention_mask)
    
    # Get the predictions for the masked positions
    predicted_ids = torch.argmax(outputs.logits, dim=-1)
    
    # Extract the predicted light chain
    predicted_light_ids = predicted_ids[0, len(heavy_tokens['input_ids'][0]) + 1:].tolist()
    predicted_light_tokens = tokenizer.convert_ids_to_tokens(predicted_light_ids)
    predicted_light_sequence = ''.join(predicted_light_tokens).replace('##', '').replace(' ', '')
    
    # Remove spaces from true light sequence
    true_light_sequence = light_seq.replace(' ', '')
    
    # Calculate perplexity using only the logits and labels for the light chain
    light_chain_logits = outputs.logits[0, len(heavy_tokens['input_ids'][0]) + 1:, :]
    mask = (masked_light_tokens[0] == tokenizer.mask_token_id)
    perplexity = calculate_perplexity(light_chain_logits, light_tokens['input_ids'][0], mask)
    total_perplexity += perplexity

    # Calculate global alignment similarity
    global_similarity = calculate_global_similarity(predicted_light_sequence, true_light_sequence)
    total_global_similarity += global_similarity

    # Calculate global alignment BLOSUM score
    global_blosum_score = calculate_global_blosum_score(predicted_light_sequence, true_light_sequence)
    total_global_blosum_score += global_blosum_score

    # Calculate hard BLOSUM score and hard similarity
    hard_blosum_score, hard_similarity = calculate_hard_blosum_and_similarity(true_light_sequence, predicted_light_sequence, substitution_matrix)
    total_hard_blosum_score += hard_blosum_score
    total_hard_similarity += hard_similarity

    num_sequences += 1

    # Output the results for the current sequence
    print("\n")
    print(f"True light sequence: {true_light_sequence}")
    print(f"Predicted light sequence: {predicted_light_sequence}")
    print(f"  Perplexity: {perplexity}")
    print(f"  Global Alignment Similarity: {global_similarity:.4f}")
    print(f"  Global Alignment BLOSUM score: {global_blosum_score}")
    print(f"  'Hard' Similarity: {hard_similarity:.2f}%")
    print(f"  'Hard' BLOSUM score: {hard_blosum_score}")

# Calculate and print average metrics if there are valid sequences
if num_sequences > 0:
    avg_perplexity = total_perplexity / num_sequences
    avg_global_similarity = total_global_similarity / num_sequences
    avg_global_blosum_score = total_global_blosum_score / num_sequences
    avg_hard_similarity = total_hard_similarity / num_sequences
    avg_hard_blosum_score = total_hard_blosum_score / num_sequences

    print("\n--- Averages Across All Sequences ---")
    print(f"Average Perplexity: {avg_perplexity}")
    print(f"Average Global Alignment Similarity: {avg_global_similarity:.4f}")
    print(f"Average Global Alignment BLOSUM Score: {avg_global_blosum_score}")
    print(f"Average 'Hard' Similarity: {avg_hard_similarity:.2f}%")
    print(f"Average 'Hard' BLOSUM Score: {avg_hard_blosum_score}")
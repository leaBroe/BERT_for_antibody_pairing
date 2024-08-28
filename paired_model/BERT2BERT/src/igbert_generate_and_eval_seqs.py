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

# Function to calculate hard BLOSUM score and hard similarity
def calculate_hard_blosum_and_similarity(true_seq, generated_seq, matrix):
    score = 0
    matches = 0
    generated_seq = generated_seq.replace(" ", "")
    true_seq = true_seq.replace(" ", "")
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

# Calculate metrics for each sequence
sep_token_id = tokenizer.convert_tokens_to_ids('[SEP]')
for i in range(predicted_ids.size(0)):
    sep_indices = (tokens['input_ids'][i] == sep_token_id).nonzero(as_tuple=True)[0]
    if sep_indices.numel() > 0:
        sep_index = sep_indices[0].item()  # Take the first occurrence of [SEP]

        # Extract the token IDs after the [SEP] token and decode them
        predicted_light_ids = predicted_ids[i, sep_index + 1:].tolist()
        predicted_light_tokens = tokenizer.convert_ids_to_tokens(predicted_light_ids)

        # Join the tokens to form the light sequence
        predicted_light_sequence = ''.join(predicted_light_tokens).replace('##', '').replace(' ', '')
        predicted_light_sequences.append(predicted_light_sequence)

        # Calculate perplexity
        mask = (tokens['input_ids'][i] == tokenizer.convert_tokens_to_ids('[MASK]')).nonzero(as_tuple=True)[0]
        perplexity = calculate_perplexity(outputs.logits[i], tokens['input_ids'][i], mask)
        perplexities.append(perplexity)

        # Calculate global alignment similarity
        global_similarity = calculate_global_similarity(predicted_light_sequence, sequences_light[i])
        global_similarities.append(global_similarity)

        # Calculate global alignment BLOSUM score
        global_blosum_score = calculate_global_blosum_score(predicted_light_sequence, sequences_light[i])
        global_blosum_scores.append(global_blosum_score)

        # Calculate hard BLOSUM score and hard similarity
        hard_blosum_score, hard_similarity = calculate_hard_blosum_and_similarity(sequences_light[i], predicted_light_sequence, substitution_matrix)
        hard_blosum_scores.append(hard_blosum_score)
        hard_similarities.append(hard_similarity)
    else:
        predicted_light_sequences.append("")
        perplexities.append(None)
        global_similarities.append(None)
        global_blosum_scores.append(None)
        hard_similarities.append(None)
        hard_blosum_scores.append(None)

# Output the predicted light sequences and metrics
for i, (pred_seq, pp, global_sim, global_blosum, hard_sim, hard_blosum) in enumerate(zip(predicted_light_sequences, perplexities, global_similarities, global_blosum_scores, hard_similarities, hard_blosum_scores)):
    print(f"Predicted light sequence {i + 1}: {pred_seq}")
    print(f"  Perplexity: {pp}")
    print(f"  Global Alignment Similarity: {global_sim:.4f} %")
    print(f"  Global Alignment BLOSUM score: {global_blosum}")
    print(f"  Hard Similarity: {hard_sim:.2f}%")
    print(f"  Hard BLOSUM score: {hard_blosum}")


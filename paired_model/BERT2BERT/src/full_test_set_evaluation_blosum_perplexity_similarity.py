from transformers import EncoderDecoderModel, AutoTokenizer, GenerationConfig
import torch
import pandas as pd
from adapters import init
from Bio.Align import substitution_matrices
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

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
    init(model)
    model.to(device)
    model.load_adapter(adapter_path)
    model.set_active_adapters(adapter_name)
    generation_config = GenerationConfig.from_pretrained(generation_config_path)
    
    return model, tokenizer, generation_config


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

#################################### heavy2light with adapters ################################################
# model heavy2light run name: save_adapter_FULL_data_temperature_0.5_tests_max_length_150_early_stopping_true_heavy2light_with_adapters_batch_size_64_epochs_40_lr_0.0001_weight_decay_0.1
model_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/save_adapter_FULL_data_temperature_0.5"
tokenizer_path = f"{model_path}/checkpoint-336040"
adapter_path = f"{model_path}/final_adapter"
generation_config_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/save_adapter_FULL_data_temperature_0.5"
adapter_name = "heavy2light_adapter"

model, tokenizer, generation_config = initialize_model_and_tokenizer(model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name)

# Load test data
test_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_test_no_ids_space_separated_SMALL.txt'

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(line.strip())
    sequences = []
    for entry in data:
        split_entry = entry.split(' [SEP] ')
        if len(split_entry) == 2:
            sequences.append(split_entry)
        else:
            print(f"Skipping invalid entry: {entry}")
    df = pd.DataFrame(sequences, columns=['heavy', 'light'])
    return df

test_df = load_data(test_file_path)
heavy_sequences = test_df["heavy"]
true_light_sequences = test_df["light"]

# convert true_light_sequences to list
true_light_sequences = true_light_sequences.tolist()

# Generate sequences
generated_light_seqs = []

for i in range(len(heavy_sequences)):
    inputs = tokenizer(heavy_sequences[i], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    generated_seq = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=150, output_scores=True, return_dict_in_generate=True, generation_config=generation_config)
    sequence = generated_seq["sequences"][0]
    generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
    generated_light_seqs.append(generated_text)


# Calculate BLOSUM scores
def calculate_blosum_score(true_seq, generated_seq, matrix):
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
    return score, min_length, matches, similarity_percentage


blosum62 = substitution_matrices.load("BLOSUM62")
scores = []
similarities = []

for i in range(len(generated_light_seqs)):
    score, min_length, matches, similarity_percentage = calculate_blosum_score(true_light_sequences[i], generated_light_seqs[i], blosum62)
    scores.append(score)
    similarities.append(similarity_percentage)
    print(f"\nSequence pair {i+1}:")
    print(f"True Sequence: {true_light_sequences[i]}")
    print(f"Generated Sequence: {generated_light_seqs[i]}")
    print(f"BLOSUM Score: {score}")
    print(f"Minimum Length: {min_length}")
    print(f"Matches: {matches}")
    print(f"Similarity Percentage: {similarity_percentage}%")

average_blosum_score = sum(scores) / len(scores)
average_similarity_percentage = sum(similarities) / len(similarities)
print(f"\nAverage BLOSUM Score: {average_blosum_score}")
print(f"Average Similarity Percentage: {average_similarity_percentage}%")

# Calculate perplexity
inputs = tokenizer(generated_light_seqs, padding=True, truncation=True, return_tensors="pt")
targets = tokenizer(true_light_sequences, padding=True, truncation=True, return_tensors="pt")

outputs = model(input_ids=inputs.input_ids, decoder_input_ids=targets.input_ids).to(device)
logits = outputs.logits
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = targets.input_ids[:, 1:].contiguous()
loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
target_mask = (shift_labels != tokenizer.pad_token_id).float()
loss = loss.view(shift_labels.size()) * target_mask
log_likelihood = loss.sum(dim=1)
perplexity = torch.exp(log_likelihood / target_mask.sum(dim=1)).cpu().detach().numpy()

for seq, ppl in zip(generated_light_seqs, perplexity):
    print(f"Generated Sequence: {seq}")
    print(f"Perplexity: {ppl}")

mean_perplexity = np.mean(perplexity)
print(f"Mean Perplexity: {mean_perplexity}")






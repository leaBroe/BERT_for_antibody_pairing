# used env: adap_2
from transformers import EncoderDecoderModel, AutoTokenizer, GenerationConfig
import torch
import pandas as pd
from adapters import init
from Bio.Align import substitution_matrices
import numpy as np
from tqdm import tqdm
from Bio import pairwise2


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
    model.to(device)
    init(model)
    print(f"model is on device: {model.device}")
    #model.to(device)
    model.load_adapter(adapter_path)
    model.set_active_adapters(adapter_name)
    generation_config = GenerationConfig.from_pretrained(generation_config_path)
    
    return model, tokenizer, generation_config



#################################### heavy2light with adapters ################################################
# model heavy2light run name: save_adapter_FULL_data_temperature_0.5_tests_max_length_150_early_stopping_true_heavy2light_with_adapters_batch_size_64_epochs_40_lr_0.0001_weight_decay_0.1
# run_name = "save_adapter_FULL_data_temperature_0.5_tests_max_length_150_early_stopping_true_heavy2light_with_adapters_batch_size_64_epochs_40_lr_0.0001_weight_decay_0.1"
# model_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/save_adapter_FULL_data_temperature_0.5"
# tokenizer_path = f"{model_path}/checkpoint-336040"
# adapter_path = f"{model_path}/final_adapter"
# generation_config_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/save_adapter_FULL_data_temperature_0.5"
# adapter_name = "heavy2light_adapter"


# heavy2light 500 epochs run name: FULL_data_small_small_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_500_lr_0.0005_wd_0.05
# use adapter at epoch 356 -> lowest eval loss
# run_name = "FULL_data_small_small_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_500_lr_0.0005_wd_0.05"
# model_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/FULL_data_small_small_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_500_lr_0.0005_wd_0.05"
# tokenizer_path = f"{model_path}/checkpoint-2990756"
# adapter_path = f"{tokenizer_path}/heavy2light_adapter"
# generation_config_path = model_path
# adapter_name = "heavy2light_adapter"

# # heavy2light run name: /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/FULL_data_small_small_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.01
# run_name = "FULL_data_small_small_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.01"
# model_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/FULL_data_small_small_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.01"
# tokenizer_path = f"{model_path}/checkpoint-420050"
# adapter_path = f"{model_path}/final_adapter"
# generation_config_path = model_path
# adapter_name = "heavy2light_adapter"

# # heavy2light BIG/BIG
# run_name = "FULL_data_big_big_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_1e-05_wd_0.5"
# model_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/FULL_data_big_big_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_1e-05_wd_0.5"
# tokenizer_path = f"{model_path}/checkpoint-84010"
# adapter_path = f"{model_path}/final_adapter"
# generation_config_path = model_path
# adapter_name = "heavy2light_adapter"

# # heavy2light 10 epochs diverse beam search beam = 5
# run_name="full_Diverse_beam_search_5_decoding_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1"
# model_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_Diverse_beam_search_5_decoding_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1"
# tokenizer_path = f"{model_path}/checkpoint-84010"
# adapter_path = f"{model_path}/final_adapter"
# generation_config_path = model_path
# adapter_name = "heavy2light_adapter"

# # heavy2light 10 epochs diverse beam search beam = 2
# run_name="full_diverse_beam_search_2_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1"
# model_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_diverse_beam_search_2_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1"
# tokenizer_path = f"{model_path}/checkpoint-84010"
# adapter_path = f"{model_path}/final_adapter"
# generation_config_path = model_path
# adapter_name = "heavy2light_adapter"

# # # heavy2light 50 epochs diverse beam search beam = 5
# run_name="full_Diverse_beam_search_5_decoding_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1"
# model_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_Diverse_beam_search_5_decoding_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1"
# tokenizer_path = f"{model_path}/checkpoint-420050"
# adapter_path = f"{model_path}/final_adapter"
# generation_config_path = model_path
# adapter_name = "heavy2light_adapter"

# # heavy2light contrastive search 40 epochs
# run_name="full_contrastive_search_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_40_lr_0.0001_wd_0.1"
# model_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_contrastive_search_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_40_lr_0.0001_wd_0.1"
# tokenizer_path = f"{model_path}/checkpoint-336040"
# adapter_path = f"{model_path}/final_adapter"
# generation_config_path = model_path
# adapter_name = "heavy2light_adapter"

# # IgBERT2IgBERT 
# run_name="FULL_data_cross_attention_with_adapters_batch_size_64_epochs_20_lr_0.0005_weight_decay_0.05"
# model_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/bert2bert-translation_heavy-to-light_model_checkpoints/FULL_data_cross_attention_with_adapters_batch_size_64_epochs_20_lr_0.0005_weight_decay_0.05"
# tokenizer_path = f"{model_path}/checkpoint-168020"
# adapter_path = f"{model_path}/checkpoint-168020/seq2seq_adapter"
# generation_config_path = model_path
# adapter_name = "seq2seq_adapter"

# # IgBERT2IgBERT 20 epochs weird drop in loss
# run_name="FULL_data_cross_attention_with_adapters_batch_size_64_epochs_20_lr_0.0001_weight_decay_0.1"
# model_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/bert2bert-translation_heavy-to-light_model_checkpoints/FULL_data_cross_attention_with_adapters_batch_size_64_epochs_20_lr_0.0001_weight_decay_0.1"
# tokenizer_path = f"{model_path}/checkpoint-168020"
# adapter_path = f"{model_path}/checkpoint-168020/seq2seq_adapter"
# generation_config_path = model_path
# adapter_name = "seq2seq_adapter"

# # heavy2light 60 epochs diverse beam search beam = 5
# run_name="full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1"
# model_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1"
# tokenizer_path=f"{model_path}/checkpoint-504060"
# adapter_path=f"{model_path}/final_adapter"
# generation_config_path=model_path
# adapter_name="heavy2light_adapter"

# # heavy2light 100 epochs diverse beam search beam = 5
# run_name="full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_100_lr_0.001_wd_0.1"
# model_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_100_lr_0.001_wd_0.1"
# tokenizer_path=f"{model_path}/checkpoint-840100"
# adapter_path=f"{model_path}/final_adapter"
# generation_config_path=model_path
# adapter_name="heavy2light_adapter"

################## NEW DATA ##################
# # PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1
# run_name="PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1"
# model_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1"
# tokenizer_path=f"{model_path}/checkpoint-367750"
# adapter_path=f"{model_path}/final_adapter"
# generation_config_path=model_path
# adapter_name="heavy2light_adapter"

# # PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1
# run_name="PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1"
# model_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1"
# tokenizer_path=f"{model_path}/checkpoint-441300"
# adapter_path=f"{model_path}/final_adapter"
# generation_config_path=model_path
# adapter_name="heavy2light_adapter"

# # PLAbDab_all_human_paired_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.0001_wd_0.1
# run_name="PLAbDab_all_human_paired_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.0001_wd_0.1"
# model_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/PLAbDab_all_human_paired_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.0001_wd_0.1"
# tokenizer_path=f"{model_path}/checkpoint-806340"
# adapter_path=f"{model_path}/final_adapter"
# generation_config_path=model_path
# adapter_name="heavy2light_adapter"

# PLAbDab_healthy_human_covid_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.0001_wd_0.1
run_name="PLAbDab_healthy_human_covid_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.0001_wd_0.1"
model_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/PLAbDab_healthy_human_covid_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.0001_wd_0.1"
tokenizer_path = f"{model_path}/checkpoint-969240"
adapter_path = f"{model_path}/final_adapter"
generation_config_path = model_path
adapter_name = "heavy2light_adapter"

model, tokenizer, generation_config = initialize_model_and_tokenizer(model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name)

# Load small test data
#test_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_test_no_ids_space_separated_SMALL.txt'

# load FULL test data
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_test_no_ids_space_separated.txt"

# NEW DATA
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/train_val_test_datasets/plabdab_human_healthy_no_vac_allocated_test_no_identifiers_spaces.txt"

# test data all human paired + plabdab
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_all_disease_plabdab/train_val_test_datasets/human_healthy_all_diseases_plabdab_test_no_identifiers_spaces.txt"

# human_healthy_and_covid
test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_and_covid/train_test_val_datasets/human_healthy_covid_allocated__test_no_identifiers_spaces.txt"

# small test data
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_and_covid/train_test_val_datasets/human_healthy_covid_allocated__test_no_identifiers_spaces_small.txt"


print(f"Fully evaluating model with run name: {run_name}")

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
    return score, similarity_percentage

# # Lists to store the results
# scores = []
# similarities = []
# perplexities = []

# for i in tqdm(range(len(heavy_sequences)), desc="Processing sequences"):
#     # Generate sequence
#     inputs = tokenizer(heavy_sequences[i], padding="max_length", truncation=True, max_length=512, return_tensors="pt").to(device)
#     model.to(device)
#     print(f"model is on device {model.device}")
#     generated_seq = model.generate(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, max_length=150, output_scores=True, return_dict_in_generate=True, generation_config=generation_config)
#     sequence = generated_seq["sequences"][0]
#     generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
    
#     # Calculate BLOSUM score and similarity
#     score, similarity_percentage = calculate_blosum_score(true_light_sequences[i], generated_text, blosum62)
#     scores.append(score)
#     similarities.append(similarity_percentage)
    
#     # Calculate perplexity
#     inputs = tokenizer(generated_text, padding=True, truncation=True, return_tensors="pt").to(device)
#     targets = tokenizer(true_light_sequences[i], padding=True, truncation=True, return_tensors="pt").to(device)
    
#     with torch.no_grad():
#         outputs = model(input_ids=inputs.input_ids, decoder_input_ids=targets.input_ids)
    
#     logits = outputs.logits
#     shift_logits = logits[:, :-1, :].contiguous()
#     shift_labels = targets.input_ids[:, 1:].contiguous()
    
#     loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
#     loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    
#     target_mask = (shift_labels != tokenizer.pad_token_id).float()
#     loss = loss.view(shift_labels.size()) * target_mask
    
#     log_likelihood = loss.sum(dim=1)
#     perplexity = torch.exp(log_likelihood / target_mask.sum(dim=1)).cpu().detach().numpy()
    
#     perplexities.append(perplexity[0])
    
#     # Print results for each sequence pair
#     print(f"\nSequence pair {i+1}:")
#     print(f"True Sequence: {true_light_sequences[i]}")
#     print(f"Generated Sequence: {generated_text}")
#     print(f"BLOSUM Score: {score}")
#     print(f"Similarity Percentage: {similarity_percentage}%")
#     print(f"Perplexity: {perplexity[0]}")

# # Calculate and print average scores and perplexity
# average_blosum_score = np.mean(scores)
# average_similarity_percentage = np.mean(similarities)
# mean_perplexity = np.mean(perplexities)

# print(f"\nAverage BLOSUM Score: {average_blosum_score}")
# print(f"Average Similarity Percentage: {average_similarity_percentage}%")
# print(f"Mean Perplexity: {mean_perplexity}")


# Use the BLOSUM62 matrix
blosum62 = substitution_matrices.load("BLOSUM62")


def calculate_blosum_score_with_global_alignment(seq1, seq2, blosum_matrix):
    # Clean sequences to remove invalid characters
    seq1 = seq1.replace(' ', '')
    seq2 = seq2.replace(' ', '')
    
    # Perform global alignment
    alignments = pairwise2.align.globalds(seq1, seq2, blosum_matrix, -10, -1)
    best_alignment = alignments[0]
    
    # Extract aligned sequences and calculate similarity
    aligned_seq1, aligned_seq2, score, start, end = best_alignment
    matches = sum(1 for a, b in zip(aligned_seq1, aligned_seq2) if a == b and a != '-')
    similarity_percentage = (matches / max(len(seq1), len(seq2))) * 100
    
    return score, similarity_percentage

# # test global alignment function
# true_seq="DIELTQSPAIMSASLGEKVTMSCRASSSVNFIYWYQQKSDASPKLWVYYTSHLPPGVPARFSGSGSGNSYSLTISSMEGEDAATYYCQQFTSSPFTFGSGTKLEIK"
# gen_seq="DIVMTQSPSSLAVSAGEKVTMSCKSSQSLLNSRTRKNYLAWYQQKPGQSPKLLIYWASTRESGVPDRFTGSGSGTDFTLTISSVQAEDLAVYYCKQSYNLRTFGGGTKLEIK"
# score, similarity_percentage = calculate_blosum_score_with_global_alignment(true_seq, gen_seq, blosum62)
# print(f"score: {score}, similarity_percentage: {similarity_percentage}")


# Updated loop for sequence generation and BLOSUM score calculation
scores = []
similarities = []
perplexities = []

for i in tqdm(range(len(heavy_sequences)), desc="Processing sequences"):
    # Generate sequence
    inputs = tokenizer(
        heavy_sequences[i],
        padding="max_length",
        truncation=True,
        max_length=512,
        return_tensors="pt"
    ).to(device)
    model.to(device)
    print(f"model is on device {model.device}")
    generated_seq = model.generate(
        input_ids=inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_length=150,
        output_scores=True,
        return_dict_in_generate=True,
        generation_config=generation_config
    )
    sequence = generated_seq["sequences"][0]
    generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
    
    # Calculate BLOSUM score and similarity using global alignment
    score, similarity_percentage = calculate_blosum_score_with_global_alignment(
        true_light_sequences[i], generated_text, blosum62
    )
    scores.append(score)
    similarities.append(similarity_percentage)
    
    # Calculate perplexity
    inputs = tokenizer(generated_text, padding=True, truncation=True, return_tensors="pt").to(device)
    targets = tokenizer(true_light_sequences[i], padding=True, truncation=True, return_tensors="pt").to(device)
    
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
    
    perplexities.append(perplexity[0])
    
    # Print results for each sequence pair
    print(f"\nSequence pair {i+1}:")
    print(f"True Sequence: {true_light_sequences[i]}")
    print(f"Generated Sequence: {generated_text}")
    print(f"BLOSUM Score: {score}")
    print(f"Similarity Percentage: {similarity_percentage}%")
    print(f"Perplexity: {perplexity[0]}")



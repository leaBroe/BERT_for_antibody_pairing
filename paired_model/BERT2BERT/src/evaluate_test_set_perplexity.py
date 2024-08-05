from transformers import EncoderDecoderModel, PreTrainedTokenizerFast, AutoTokenizer
import torch
import numpy as np
from adapters import init


# Load the model and tokenizer
# "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/2ndr_run_FULL_data_heavy2light_with_adapters_batch_size_64_epochs_20_lr_0.0001_weight_decay_0.1"
# /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/FULL_data_temperature_0.5_tests_max_length_150_early_stopping_true_heavy2light_with_adapters_batch_size_64_epochs_30_lr_0.0001_weight_decay_0.1

#################################### heavy2light with adapters ################################################

# # model heavy2light run name: save_adapter_FULL_data_temperature_0.5_tests_max_length_150_early_stopping_true_heavy2light_with_adapters_batch_size_64_epochs_40_lr_0.0001_weight_decay_0.1
# adapter_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/save_adapter_FULL_data_temperature_0.5/final_adapter"

# # tokenizer and model not pretrained (models before finetuning)
# #tokenizer = AutoTokenizer.from_pretrained('/ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-56556520')
# #model = EncoderDecoderModel.from_encoder_decoder_pretrained("/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-117674391", "/ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-56556520", add_cross_attention=True)

# # pretrained tokenizer and model
# tokenizer_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/save_adapter_FULL_data_temperature_0.5/checkpoint-336040"
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
# model_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/save_adapter_FULL_data_temperature_0.5"
# model = EncoderDecoderModel.from_pretrained(model_path)
# init(model)
# model.load_adapter(adapter_path)
# model.set_active_adapters("heavy2light_adapter")

# # heavy2light with adapters output file path
# file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/HEAVY2LIGHT_114312.o" 

#################################### IgBERT2IgBERT with adapters ################################################

# #run name: FULL_data_cross_attention_with_adapters_batch_size_64_epochs_20_lr_0.0001_weight_decay_0.1

# # model heavy2light run name: save_adapter_FULL_data_temperature_0.5_tests_max_length_150_early_stopping_true_heavy2light_with_adapters_batch_size_64_epochs_40_lr_0.0001_weight_decay_0.1
# adapter_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/bert2bert-translation_heavy-to-light_model_checkpoints/FULL_data_cross_attention_with_adapters_batch_size_64_epochs_20_lr_0.0001_weight_decay_0.1/checkpoint-168020/seq2seq_adapter"

# # pretrained tokenizer and model
# tokenizer_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/bert2bert-translation_heavy-to-light_model_checkpoints/FULL_data_cross_attention_with_adapters_batch_size_64_epochs_20_lr_0.0001_weight_decay_0.1/checkpoint-168020"
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
# model_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/bert2bert-translation_heavy-to-light_model_checkpoints/FULL_data_cross_attention_with_adapters_batch_size_64_epochs_20_lr_0.0001_weight_decay_0.1"
# model = EncoderDecoderModel.from_pretrained(model_path)
# init(model)
# model.load_adapter(adapter_path)
# model.set_active_adapters("seq2seq_adapter")

# IgBERT2IgBERT output file path
# file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/b2b_adaps_114271.o"


# #################################### heavy2light without adapters ################################################
# # run name: FULL_data_heavy2light_without_adapters_batch_size_64_epochs_20_lr_0.0001_weight_decay_0.1
# # used checkpoint at epoch 13 bc there the loss was the smallest

# # pretrained tokenizer and model
# tokenizer_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/FULL_data_heavy2light_without_adapters_batch_size_64_epochs_20_lr_0.0001_weight_decay_0.1/checkpoint-109213"
# tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
# model_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/FULL_data_heavy2light_without_adapters_batch_size_64_epochs_20_lr_0.0001_weight_decay_0.1/checkpoint-109213"
# model = EncoderDecoderModel.from_pretrained(model_path)

# # heavy2light without adapters output file path
# file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/h2l_no_adaps_114294.o"

#################################### IgBERT2IgBERT without adapters ################################################
#run name: FULL_data_cross_attention_without_adapters_batch_size_32_epochs_20_lr_0.0001_weight_decay_0.1

# pretrained tokenizer and model
tokenizer_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/bert2bert-translation_heavy-to-light_model_checkpoints/FULL_data_cross_attention_without_adapters_batch_size_32_epochs_20_lr_0.0001_weight_decay_0.1/checkpoint-336040"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/bert2bert-translation_heavy-to-light_model_checkpoints/FULL_data_cross_attention_without_adapters_batch_size_32_epochs_20_lr_0.0001_weight_decay_0.1"
model = EncoderDecoderModel.from_pretrained(model_path)

# IgBERT2IgBERT without adapters output file path
file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/b2b_no_adaps_114272.o"

# input sequences: generated light sequences
# target sequences: true light sequences

def extract_sequences_from_file(file_path):
    decoded_sequences = []
    true_sequences = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            if line.startswith("decoded light sequence:"):
                decoded_sequences.append(line.split(":")[1].strip())
            elif line.startswith("true light sequence:"):
                true_sequences.append(line.split(":")[1].strip())
    return decoded_sequences, true_sequences




input_sequences, target_sequences = extract_sequences_from_file(file_path)

# Tokenize input and target sequences
inputs = tokenizer(input_sequences, padding=True, truncation=True, return_tensors="pt")
targets = tokenizer(target_sequences, padding=True, truncation=True, return_tensors="pt")

# Generate predictions (logits)
outputs = model(input_ids=inputs.input_ids, decoder_input_ids=targets.input_ids)
logits = outputs.logits

# Compute log-likelihoods
shift_logits = logits[:, :-1, :].contiguous()
shift_labels = targets.input_ids[:, 1:].contiguous()
loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

# Mask padding tokens
target_mask = (shift_labels != tokenizer.pad_token_id).float()
loss = loss.view(shift_labels.size()) * target_mask
log_likelihood = loss.sum(dim=1)

# Calculate perplexity
perplexity = torch.exp(log_likelihood / target_mask.sum(dim=1))

# Convert to numpy for easier handling
perplexity = perplexity.cpu().detach().numpy()

# Print perplexity for each sequence
for seq, ppl in zip(input_sequences, perplexity):
    print(f"Input Sequence: {seq}")
    print(f"Perplexity: {ppl}")

# Compute mean perplexity for all sequences
mean_perplexity = np.mean(perplexity)
print(f"Mean Perplexity: {mean_perplexity}")


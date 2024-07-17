from transformers import EncoderDecoderModel, PreTrainedTokenizerFast, AutoTokenizer
import torch
import numpy as np
from adapters import init


# Load the model and tokenizer
# "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/2ndr_run_FULL_data_heavy2light_with_adapters_batch_size_64_epochs_20_lr_0.0001_weight_decay_0.1"
# /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/FULL_data_temperature_0.5_tests_max_length_150_early_stopping_true_heavy2light_with_adapters_batch_size_64_epochs_30_lr_0.0001_weight_decay_0.1

# model heavy2light run name: save_adapter_FULL_data_temperature_0.5_tests_max_length_150_early_stopping_true_heavy2light_with_adapters_batch_size_64_epochs_40_lr_0.0001_weight_decay_0.1
adapter_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/save_adapter_FULL_data_temperature_0.5/final_adapter"

# tokenizer and model not pretrained (models before finetuning)
#tokenizer = AutoTokenizer.from_pretrained('/ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-56556520')
#model = EncoderDecoderModel.from_encoder_decoder_pretrained("/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-117674391", "/ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-56556520", add_cross_attention=True)

# pretrained tokenizer and model
tokenizer_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/save_adapter_FULL_data_temperature_0.5/checkpoint-336040"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
model_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/save_adapter_FULL_data_temperature_0.5"
model = EncoderDecoderModel.from_pretrained(model_path)
init(model)
model.load_adapter(adapter_path)
model.set_active_adapters("heavy2light_adapter")

#model_name = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/FULL_data_temperature_0.5_tests_max_length_150_early_stopping_true_heavy2light_with_adapters_batch_size_64_epochs_30_lr_0.0001_weight_decay_0.1"
#tokenizer = AutoTokenizer.from_pretrained("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/FULL_data_temperature_0.5_tests_max_length_150_early_stopping_true_heavy2light_with_adapters_batch_size_64_epochs_30_lr_0.0001_weight_decay_0.1/checkpoint-252030")
#model = EncoderDecoderModel.from_pretrained(model_name)

# input sequences: generated light sequences
# target sequences: true light sequences

# Prepare your input data
input_sequences = ["D I Q M T Q S P S T L S A S V G D R V T I T C R A N Q N I N N W L A W Y Q Q K P G K A P K L L I Y K T S S L E S G V P L R F S D T G S E T E F T F I I S N L Q P D D F A T Y Y C Q H Y N S Y P W A F G Q G T K V E I K"]
target_sequences = ["E I V M T Q S P G T L S L S P G E T A T L S C R A S Q S V S S H F A W Y Q Q T P G Q A P R L V I Y A T S T R A A G V P A R F S G S G S G T E F T L T I S S L Q S E D F A V Y Y C H Q Y N N W P F N F G G G T K V E I K"]

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


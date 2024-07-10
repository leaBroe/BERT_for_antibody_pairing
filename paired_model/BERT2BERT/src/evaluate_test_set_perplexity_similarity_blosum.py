from transformers import EncoderDecoderModel, PreTrainedTokenizerFast, AutoTokenizer
import torch
import numpy as np


# Load the model and tokenizer
model_name = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/2ndr_run_FULL_data_heavy2light_with_adapters_batch_size_64_epochs_20_lr_0.0001_weight_decay_0.1"
tokenizer = AutoTokenizer.from_pretrained("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/2ndr_run_FULL_data_heavy2light_with_adapters_batch_size_64_epochs_20_lr_0.0001_weight_decay_0.1/checkpoint-168020")
model = EncoderDecoderModel.from_pretrained(model_name)

# Prepare your input data
input_sequences = ["Q S A L T Q P V S V S G S P G Q S I A I S C T G T S S D V G G Y N S V S W F Q Q H P G K A P K L M I Y D V S N R P S G V S N R F S G S K S G N T A S L T I S G L Q A E D E A D Y Y C S S Y T S S S T R L F G G G T K L T V L"]
target_sequences = ["Q S A L T Q P V S V S G S P G Q S I A I S C T G T S S D V G G Y N S V S W F Q Q H P G K A P K L M I Y D V S N R P S G V S N R F S G S K S G N T A S L T I S G L Q A E D E A D Y Y C S S Y T S S S T R L F G G G T K L T V L"]

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


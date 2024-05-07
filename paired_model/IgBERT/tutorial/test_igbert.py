from transformers import BertModel, BertTokenizer
import torch


tokenizer = BertTokenizer.from_pretrained("Exscientia/IgBert", do_lower_case=False)
model = BertModel.from_pretrained("Exscientia/IgBert", add_pooling_layer=False)

# The tokeniser expects input of the form ["V Q ... S S [SEP] E V ... I K", ...]
# load paired sequences from txt file as list
with open("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/data/small_training_set.txt", "r") as f:
    paired_sequences = f.read().splitlines()

tokens = tokenizer.batch_encode_plus(
    paired_sequences, 
    add_special_tokens=True, 
    padding="longest",
    return_tensors="pt",
    return_special_tokens_mask=True
)

# print the first 10 tokens of the first sequence

print(tokens["input_ids"][0][:10])

output = model(
    input_ids=tokens['input_ids'], 
    attention_mask=tokens['attention_mask']
)

residue_embeddings = output.last_hidden_state

print(residue_embeddings.shape)

# mask special tokens before summing over embeddings
residue_embeddings[tokens["special_tokens_mask"] == 1] = 0
sequence_embeddings_sum = residue_embeddings.sum(1)

# average embedding by dividing sum by sequence lengths
sequence_lengths = torch.sum(tokens["special_tokens_mask"] == 0, dim=1)
sequence_embeddings = sequence_embeddings_sum / sequence_lengths.unsqueeze(1)

print(sequence_embeddings.shape)

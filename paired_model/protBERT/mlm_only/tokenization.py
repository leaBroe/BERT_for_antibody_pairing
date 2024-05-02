
# Since BERT's default tokenizer is designed for natural language and expects words or subwords as inputs, 
# using it directly for amino acid sequences is not optimal. 
# Instead, use character-level tokenization, where each amino acid is treated as a separate token. 
# This requires mapping each amino acid to a unique ID, similar to how subwords are tokenized in NLP.

import torch
from torch.utils.data import Dataset, DataLoader
import random

def load_sequences(file_path):
    with open(file_path, 'r') as file:
        sequences = file.read().splitlines()
    return sequences

#training_sequences = load_sequences('/ibmm_data2/oas_database/paired_lea_tmp/light_model/data/training_set_light_seq_100_pident.txt')
#test_sequences = load_sequences('/ibmm_data2/oas_database/paired_lea_tmp/light_model/data/test_set_light_seq_100_pident.txt')

training_sequences = load_sequences('/ibmm_data2/oas_database/paired_lea_tmp/light_model/data/training_set_test.txt')
test_sequences = load_sequences('/ibmm_data2/oas_database/paired_lea_tmp/light_model/data/test_set_test.txt')


amino_acids = 'LAGVESIKRDTPNQFYMHCWXUBZO'
special_tokens = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}

# Create a dictionary mapping each amino acid and special token to a unique integer
aa_to_id = {aa: i+5 for i, aa in enumerate(amino_acids)}
aa_to_id.update(special_tokens)


def tokenize_and_mask_sequences(sequences, aa_to_id, max_len, mask_probability=0.15):
    tokenized_sequences = []
    masked_labels = []
    attention_masks = []  # Add this line

    for seq in sequences:
        token_ids = [aa_to_id['[CLS]']]
        labels = [-100]
        attention_mask = [1]  # `[CLS]` is attended to

        for aa in seq:
            if len(token_ids) == max_len - 1:  # Account for [SEP] token
                break
            token_id = aa_to_id.get(aa, aa_to_id['[UNK]'])
            if random.random() < mask_probability and aa not in special_tokens:
                token_ids.append(aa_to_id['[MASK]'])
                labels.append(token_id)
                attention_mask.append(1)  # Masked tokens are attended to
            else:
                token_ids.append(token_id)
                labels.append(-100)
                attention_mask.append(1)  # Real tokens are attended to

        token_ids.append(aa_to_id['[SEP]'])
        labels.append(-100)
        attention_mask.append(1)  # `[SEP]` is attended to

        # Pad the sequences to max_len
        padded_length = max_len - len(token_ids)
        token_ids.extend([aa_to_id['[PAD]']] * padded_length)
        labels.extend([-100] * padded_length)
        attention_mask.extend([0] * padded_length)  # Padding tokens are not attended to

        tokenized_sequences.append(token_ids)
        masked_labels.append(labels)
        attention_masks.append(attention_mask)  # Add this line

    return tokenized_sequences, masked_labels, attention_masks  # Modify this line



class AminoAcidDataset(torch.utils.data.Dataset):
    def __init__(self, sequences, masks, labels=None):
        self.sequences = sequences
        self.masks = masks
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        item = {
            "input_ids": torch.tensor(self.sequences[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.masks[idx], dtype=torch.long)
        }
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item




max_len = 160

#tokenized_training_sequences = tokenize_and_mask_sequences(training_sequences, aa_to_id, max_len=128)
#tokenized_test_sequences = tokenize_and_mask_sequences(test_sequences, aa_to_id, max_len=128)

tokenized_training_sequences, training_labels, training_masks = tokenize_and_mask_sequences(training_sequences, aa_to_id, max_len=max_len)
tokenized_test_sequences, test_labels, test_masks = tokenize_and_mask_sequences(test_sequences, aa_to_id, max_len=max_len)


# # Find the minimum and maximum sequence lengths
# min_length = min(len(seq) for seq in tokenized_training_sequences)
# max_length = max(len(seq) for seq in tokenized_training_sequences)

# print(f"Minimum sequence length: {min_length}")
# print(f"Maximum sequence length: {max_length}")

# # Ensure all sequences are of the expected max length
# assert all(len(seq) == max_len for seq in tokenized_training_sequences), "Not all sequences are of the expected maximum length."


# Create dataset instances
#train_dataset = AminoAcidDataset(tokenized_training_sequences)
#test_dataset = AminoAcidDataset(tokenized_test_sequences)

# Adjusted to include attention masks
#train_dataset = AminoAcidDataset(tokenized_training_sequences, training_masks)
test_dataset = AminoAcidDataset(tokenized_test_sequences, test_masks, test_labels)
train_dataset = AminoAcidDataset(tokenized_training_sequences, training_masks, training_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


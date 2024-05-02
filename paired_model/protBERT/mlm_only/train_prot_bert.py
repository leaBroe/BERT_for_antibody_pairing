import torch
import os
import wandb
from transformers import BertForMaskedLM, BertTokenizer
from torch.optim import AdamW  
from torch.utils.data import Dataset, DataLoader
from tokenization import load_sequences

import torch
import os
import random
import wandb
from transformers import BertForMaskedLM, BertTokenizer
from torch.optim import AdamW  
from torch.utils.data import DataLoader, TensorDataset

# Initialize WandB
wandb.init(project="prot_bert_mlm_only_tok", name="mlm_small_dataset")

# Check if CUDA (GPU support) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device for training.")

# Initialize the tokenizer from Rostlab's ProtBERT
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)

# Function to manually mask tokens and prepare labels for MLM
def mask_tokens(inputs, tokenizer, mlm_probability=0.15):
    """Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original."""
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `mlm_probability`)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in labels.tolist()]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.1)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the original tokens
    return inputs, labels

# Function to encode sequences using the tokenizer
def encode_sequences(sequences, tokenizer, max_len=128):
    encodings = tokenizer(sequences, max_length=max_len, truncation=True, padding="max_length", return_tensors="pt")
    inputs, labels = mask_tokens(encodings['input_ids'], tokenizer)
    return {'input_ids': inputs, 'attention_mask': encodings['attention_mask'], 'labels': labels}

# small subset of the training and validation sets
training_sequences = load_sequences("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/data/small_training_set.txt")
test_sequences = load_sequences("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/data/small_val_set.txt")

# Tokenize and encode the sequences with masking and labels
train_encodings = encode_sequences(training_sequences, tokenizer)
test_encodings = encode_sequences(test_sequences, tokenizer)

# Convert encodings to TensorDataset
train_dataset = TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_encodings['labels'])
test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], test_encodings['labels'])

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Load the model
model = BertForMaskedLM.from_pretrained('Rostlab/prot_bert_bfd').to(device)

# Set up optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Configuration for WandB
config = wandb.config
config.learning_rate = 5e-5
config.epochs = 3
config.batch_size = 8
config.train_data_size = len(train_dataset)
config.eval_data_size = len(test_dataset)
config.architecture = "BertForMaskedLM"

# Training loop
def train():
    model.train()
    for epoch in range(config.epochs):
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = (t.to(device) for t in batch)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            wandb.log({"Batch Loss": loss.item()}, step=batch_idx + epoch * len(train_loader))

        avg_loss = total_loss / len(train_loader)
        wandb.log({"Epoch Loss": avg_loss}, step=epoch)

# Evaluation function
def evaluate(model, dataloader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = (t.to(device) for t in batch)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    wandb.log({"Test Loss": avg_loss})
    return avg_loss

# Run training and evaluation
train()
test_loss = evaluate(model, test_loader)
print(f"Test Loss: {test_loss:.3f}")

# Finish WandB run
wandb.finish()

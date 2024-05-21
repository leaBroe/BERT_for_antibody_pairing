# ran it with conda env bug_env
import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from transformers import BertConfig, BertForPreTraining, BertLMHeadModel, BertTokenizer, logging
from transformers.data.datasets.language_modeling import TextDatasetForNextSentencePrediction
from transformers import pipeline, Trainer, TrainingArguments
from transformers import AutoConfig, AutoModel
from transformers import LineByLineTextDataset
from transformers import DataCollatorForLanguageModeling
from transformers import AutoTokenizer
import evaluate
# from nlp import load_dataset
# from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import seqeval
import json
import math
import logging
from datetime import datetime
import pytz     # for time zone
import gzip
import csv
import pickle
import re
import copy
from torch.optim import AdamW
from transformers import get_scheduler

#BUCKET_NAME = 'clinical_bert_bucket'

#instantiate the tokenizer
#tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
#config = AutoConfig.from_pretrained(pretrained_model_name_or_path="ProteinTokenizer/config.json")
# tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="ProteinTokenizer", do_lower_case=False)

# # Add new tokens (this updates the tokenizer's vocabulary as well)
# new_tokens = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
# tokenizer.add_tokens(new_tokens)

# # Check that the new tokens were added to the tokenizer
# print(tokenizer.additional_special_tokens)

# # Save the updated tokenizer to a directory
# tokenizer.save_pretrained('UpdatedProteinTokenizer')

# config = BertConfig.from_json_file("ProteinTokenizer/config.json")
# config.vocab_size = len(tokenizer)

# # Save the updated configuration to the same directory as the tokenizer
# config.save_pretrained('UpdatedProteinTokenizer')

os.environ['CUDA_VISIBLE_DEVICES'] ='0'

# print used device cpu or cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

# Load the updated tokenizer and configuration
tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
config = AutoConfig.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False, vocab_size=len(tokenizer), force_download=True)

#input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device) # This line.

# print tokenizer and config
print(f'tokenizer: {tokenizer}')
print(f'config: {config}')


PST = pytz.timezone('Europe/Zurich')

#instantiate the model
print("start loading model=",datetime.now(PST))
#model = BertLMHeadModel.from_pretrained("bert-base-uncased")
#model = BertForPreTraining.from_pretrained("bert-base-uncased")

#config.type_vocab_size = 2

# Load and configure your model
model = BertForPreTraining(config=config)

# Move model to the appropriate device (GPU or CPU)
model.to(device)  # 'device' is determined by torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Now you can print this to confirm
print(f"Model is on device: {next(model.parameters()).device}")  # It will show cuda:0 if on GPU


# Load your model's configuration and check the vocab_size parameter. It must match the total number of tokens in your tokenizer’s vocabulary.
print(f"Model's vocab_size: {model.config.vocab_size}")
print(f"Tokenizer's vocab_size: {tokenizer.vocab_size}")

print("Model's vocab size from embeddings:", model.bert.embeddings.word_embeddings.num_embeddings)


os.environ["WANDB_PROJECT"] = "paired_model_nsp_mlm_protbert"

# define run name
run_name = "small_dataset_10_epochs_own_training_loop_SPACES"
os.environ["WANDB_RUN_NAME"] = run_name

output_dir = run_name

# define the arguments for the trainer
training_args = TrainingArguments(
    output_dir="./small_dataset_10_epochs_own_training_loop_SPACES",          # output directory
    num_train_epochs=10,              # total # of training epochs
    per_device_train_batch_size=16,  # batch size per device during training (try 16 if needed)
    per_device_eval_batch_size=16,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='nsp_mlm_paired_protbert_test',     # directory for storing logs
    do_train=True,
    eval_strategy="epoch",
    logging_steps=5
)

def check_input_ids_validity(dataset, tokenizer):
    vocab_size = tokenizer.vocab_size
    for example in dataset:
        # Extracting input_ids from each example
        input_ids = example['input_ids'] if isinstance(example, dict) else example.input_ids
        max_id = max(input_ids)
        if max_id >= vocab_size:
            raise ValueError(f"An input_id ({max_id}) exceeds the tokenizer's vocabulary size ({vocab_size}).")
    print(f"All input_ids are within the vocabulary size.")

# Log input_ids right before training to ensure they are all valid
def log_input_ids(data_loader):
    for batch in data_loader:
        input_ids = batch['input_ids']
        if torch.any(input_ids >= tokenizer.vocab_size):
            print("Invalid input_ids detected:", input_ids)
        assert torch.all(input_ids < tokenizer.vocab_size), f"Found input_ids >= vocab size: {input_ids.max().item()}"


# small train dataset: /ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_train_for_nsp_small.txt
# small val dataset: /ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_val_for_nsp_small.txt

# small dataset with input heavyseq\nlightseq (no [SEP])
#small_train_dataset_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_train_for_nsp_small.txt"
#small_val_dataset_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_val_for_nsp_small.txt"

# small dataset with input heavyseq[SEP]lightseq with each AA SPACE SEPARATED!!
small_train_dataset_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_train_no_ids_small_SPACE_separated.txt"
small_val_dataset_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_val_no_ids_small_SPACE_separated.txt"

full_train_dataset_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_train_for_nsp.txt"
full_val_dataset_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_val_for_nsp.txt"

# Prepare the train_dataset
print("start building train_dataset=", datetime.now(PST))
train_dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer,
    file_path=small_train_dataset_path,
    block_size=128
)

# Check train_dataset input_ids
print("Checking train_dataset input_ids...")
check_input_ids_validity(train_dataset, tokenizer)

# Prepare the eval_dataset
print("start building eval_dataset=", datetime.now(PST))
eval_dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer,
    file_path=small_val_dataset_path,
    block_size=128
)

# Check eval_dataset input_ids
print("Checking eval_dataset input_ids...")
check_input_ids_validity(eval_dataset, tokenizer)

# Initialize the Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Function to create a DataLoader with debugging
def create_data_loader_with_debugging(dataset, batch_size, data_collator):
    data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
    for batch in data_loader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        next_sentence_label = batch['next_sentence_label']
        
        # Print the first batch to debug
        print("First batch before data collator:")
        for i, example in enumerate(dataset):
            if i == batch_size:
                break
            print(f"Example {i}: {tokenizer.decode(example['input_ids'])}")
        
        print("First batch after data collator:")
        print(f"Input IDs: {input_ids}")
        print(f"Attention Mask: {attention_mask}")
        print(f"Labels: {labels}")
        print(f"Next Sentence Labels: {next_sentence_label}")
        break  # Only print the first batch for debugging purposes
    return data_loader


# Create and debug the DataLoader
train_data_loader = create_data_loader_with_debugging(train_dataset, batch_size=16, data_collator=data_collator)


def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


metric = evaluate.load("accuracy", )


# def compute_metrics(eval_preds):
#     preds, labels = eval_preds
#     # preds have the same shape as the labels, after the argmax(-1) has been calculated
#     # by preprocess_logits_for_metrics
#     labels = labels.reshape(-1)
#     preds = preds.reshape(-1)
#     mask = labels != -100
#     labels = labels[mask]
#     preds = preds[mask]
#     return metric.compute(predictions=preds, references=labels)

model.to(device)

# Instantiate the trainer
print("start building trainer=",datetime.now(PST))


class CustomTrainer(Trainer):
    def log(self, logs: dict):
        # Call the parent's log method to ensure proper logging
        super().log(logs)
        
        # Log the training loss for the current epoch
        if 'loss' in logs:
            print(f"Training loss: {logs['loss']}")


trainer = CustomTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,
    data_collator=data_collator,
    eval_dataset=eval_dataset,           # evaluation dataset
    #compute_metrics=compute_metrics  # pass the compute_metrics function     
)


print("finished=",datetime.now(PST))

# Log input_ids right before training to ensure they are all valid
def log_input_ids(data_loader):
    for batch in data_loader:
        input_ids = batch['input_ids']
        if torch.any(input_ids >= tokenizer.vocab_size):
            print("Invalid input_ids detected:", input_ids)
        assert torch.all(input_ids < tokenizer.vocab_size), f"Found input_ids >= vocab size: {input_ids.max().item()}"


# Verifying all input_ids in DataLoader before training
print("Verifying all input_ids in DataLoader before training...")
train_data_loader = DataLoader(train_dataset, batch_size=16, collate_fn=data_collator)
log_input_ids(train_data_loader)


optimizer = AdamW(model.parameters(), lr=1e-5)

# Create DataLoader for debugging
train_data_loader = DataLoader(train_dataset, batch_size=16, collate_fn=data_collator)

print("Verifying all input_ids in DataLoader before training...")
log_input_ids(train_data_loader)  # This will raise an error if any input_id is out of range

# Testing DataLoader outputs
for batch in train_data_loader:
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    print(f"Batch input_ids shape: {input_ids.shape}")  # Should show consistent shapes within a batch
    print(f"Batch attention mask shape: {attention_mask.shape}")  # Verify attention masks are correct

    # Check if any input_id in the batch exceeds the tokenizer's vocab size
    if torch.any(input_ids >= tokenizer.vocab_size):
        print("Invalid input_ids detected:", input_ids)
        break

    # Optionally, break after the first batch to just test the setup
    break

# Example to check how the tokenizer handles one of your sequences
sample_sequence = "Q V Q L Q E S G P G L V K P S E T L S L T C T V S G G S I S G F Y W S W I R Q S P G K G L E"
tokens = tokenizer.tokenize(sample_sequence)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Tokens:", tokens)
print("Token IDs:", token_ids)

# turn txt fiel with sequences into a list of sequences
def read_txt_file(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    return [line.strip() for line in lines]

# Load the sequences from the txt file
train_data = read_txt_file(small_train_dataset_path)

# Check if all characters are in the tokenizer's vocab
unique_chars = set(''.join(train_data))  # Assuming train_data is a list of your sequences
unknown_chars = [char for char in unique_chars if char not in tokenizer.get_vocab()]
print("Unknown Characters:", unknown_chars)

# for batch in train_data_loader:
#     input_ids = batch['input_ids']
#     if torch.any(input_ids >= tokenizer.vocab_size):
#         print("Invalid input_ids detected:", input_ids)
#     try:
#         outputs = model(input_ids=input_ids, attention_mask=batch['attention_mask'])
#     except IndexError as e:
#         print(f"Caught IndexError: {str(e)}")
#         print(f"Problematic input_ids: {input_ids}")
#         break  # Break or handle error accordingly

# Example function to verify token type ids
def verify_token_type_ids(data_loader):
    for i, batch in enumerate(data_loader):
        input_ids = batch['input_ids']
        token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids))  # Assuming default of all zeros if not provided
        
        # Printing unique token type ids to verify them
        print(f"Batch {i}: Unique token type ids:", torch.unique(token_type_ids))

        # Optionally check max input id
        if input_ids.max().item() >= tokenizer.vocab_size:
            print("Error: input_ids exceed vocab size")
            break

        # Print a message if everything seems fine
        print(f"Batch {i}: All token type IDs are within expected range.")
        
        # Breaking after a few batches for demonstration
        if i > 5:  # Check first 5 batches
            break

# Run this function to verify token type ids in your DataLoader
verify_token_type_ids(train_data_loader)

# Prepare the learning rate scheduler
num_training_steps = len(train_data_loader) * training_args.num_train_epochs
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=num_training_steps
)

def find_nan(tensor):
    nan_mask = torch.isnan(tensor)
    if torch.any(nan_mask):
        first_nan_idx = torch.where(nan_mask)[0][0].item()
        return first_nan_idx
    return None


# class CustomBertForPreTraining(BertForPreTraining):
#     def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, next_sentence_label=None, **kwargs):
#         # Call the base BERT model to get the outputs
#         outputs = self.bert(
#             input_ids,
#             attention_mask=attention_mask,
#             token_type_ids=token_type_ids,
#             **kwargs
#         )
        
#         sequence_output, pooled_output = outputs[:2]

#         # Check for NaNs in BERT outputs
#         if torch.isnan(sequence_output).any():
#             print("NaN found in sequence_output")
#         if torch.isnan(pooled_output).any():
#             print("NaN found in pooled_output")

#         # MLM task
#         prediction_scores = self.cls.predictions(sequence_output)
#         print(f"prediction_scores: {prediction_scores}")
        
#         # Check for NaNs in MLM task
#         if torch.isnan(prediction_scores).any():
#             print("NaN found in prediction_scores")

#         # NSP task
#         seq_relationship_score = self.cls.seq_relationship(pooled_output)
        
#         # Check for NaNs in NSP task
#         if torch.isnan(seq_relationship_score).any():
#             print("NaN found in seq_relationship_score")

#         total_loss = None
#         if labels is not None and next_sentence_label is not None:
#             # MLM loss
#             loss_fct = torch.nn.CrossEntropyLoss()  # labels are shifted inside the model
#             masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
#             print(f"labels: {labels}")
#             print(f"masked_lm_loss: {masked_lm_loss}")
            
#             # Check for NaNs in MLM loss
#             if torch.isnan(masked_lm_loss).any():
#                 print("NaN found in masked_lm_loss")

#             # NSP loss
#             loss_fct = torch.nn.CrossEntropyLoss()
#             next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))

#             # Check for NaNs in NSP loss
#             if torch.isnan(next_sentence_loss).any():
#                 print("NaN found in next_sentence_loss")

#             # Combine MLM and NSP loss
#             total_loss = masked_lm_loss + next_sentence_loss
            
#             # Check for NaNs in total_loss
#             if torch.isnan(total_loss).any():
#                 print("NaN found in total_loss")

#         output = (prediction_scores, seq_relationship_score) + outputs[2:]
#         return ((total_loss,) + output) if total_loss is not None else output


# # Instantiate the custom model
# model = BertForPreTraining(config=config)
# model.to(device)

# Custom training loop with gradient clipping and detailed logging
print("Starting training...")

################# Custom Training Loop #################
model.train()

# for epoch in range(training_args.num_train_epochs):
#     for step, batch in enumerate(train_data_loader):
#         batch = {k: v.to(model.device) for k, v in batch.items()}

#         # Check for NaNs in the input batch
#         for key, value in batch.items():
#             if torch.any(torch.isnan(value)):
#                 print(f"NaN found in {key} at epoch {epoch}, step {step}")

#         # Forward pass
#         outputs = model(**batch)
        
#         # Extract the loss and logits from the model output
#         loss, prediction_logits, seq_relationship_logits = outputs
        
#         # Print the loss for debugging
#         print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")

#         # Check for NaNs in the output tensors
#         if torch.isnan(loss):
#             print(f"NaN loss encountered at epoch {epoch}, step {step}")

#             # Check prediction_logits for NaNs
#             pred_logits_nan_idx = find_nan(outputs.prediction_logits)
#             if pred_logits_nan_idx is not None:
#                 print(f"NaN found in prediction_logits at index {pred_logits_nan_idx}")
#                 print(f"prediction_logits: {outputs.prediction_logits}")

#             # Check seq_relationship_logits for NaNs
#             seq_rel_logits_nan_idx = find_nan(outputs.seq_relationship_logits)
#             if seq_rel_logits_nan_idx is not None:
#                 print(f"NaN found in seq_relationship_logits at index {seq_rel_logits_nan_idx}")
#                 print(f"seq_relationship_logits: {outputs.seq_relationship_logits}")

#             continue  # Skip the rest of the loop to avoid backpropagation on NaN loss

#         # Perform backpropagation
#         loss.backward()

#         # Clip gradients to prevent explosion
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

#         # Update parameters and learning rate
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()

#         print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")

#         # Additional logging
#         if step % 10 == 0:
#             print(f"Detailed logging at Epoch: {epoch}, Step: {step}")
#             print(f"Input IDs: {batch['input_ids']}")
#             print(f"Attention Mask: {batch['attention_mask']}")
#             print(f"Loss: {loss.item()}")


# for epoch in range(training_args.num_train_epochs):
#     for step, batch in enumerate(train_data_loader):
#         batch = {k: v.to(model.device) for k, v in batch.items()}
#         outputs = model(**batch)
#         loss = outputs.loss

#         # Check for NaN loss
#         if torch.isnan(loss):
#             print(f"NaN loss encountered at epoch {epoch}, step {step}. Skipping this batch.")
#             continue

#         # Perform backpropagation
#         loss.backward()

#         # Clip gradients to prevent explosion
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

#         # Update parameters and learning rate
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()

#         print(f"Epoch: {epoch}, Step: {step}, Loss: {loss.item()}")

#         # Additional logging
#         if step % 10 == 0:
#             print(f"Detailed logging at Epoch: {epoch}, Step: {step}")
#             print(f"Input IDs: {batch['input_ids']}")
#             print(f"Attention Mask: {batch['attention_mask']}")
#             print(f"Loss: {loss.item()}")


# # Prepare the learning rate scheduler
# num_training_steps = len(train_data_loader) * 10  # 3 is the number of epochs
# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=500,
#     num_training_steps=num_training_steps
# )

# model.train()
# for epoch in range(10):  # Loop over epochs
#     for batch in train_data_loader:
#         batch = {k: v.to(model.device) for k, v in batch.items()}
#         outputs = model(**batch)
#         loss = outputs.loss

#         # Perform backpropagation
#         loss.backward()

#         # Update parameters and learning rate
#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()

#         print(f"Epoch: {epoch}, Loss: {loss.item()}")

#         try:
#             outputs = model(**batch)
#             loss = outputs.loss
#         except RuntimeError as e:
#             print(f"Runtime error: {e}")
#             print(f"Batch input IDs: {batch['input_ids']}")
#             continue  # Skip this batch or additional handling


################# Training Loop from transformers #################

# now do training
trainer.train()
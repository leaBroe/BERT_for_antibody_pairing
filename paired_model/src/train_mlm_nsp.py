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

# Load the updated tokenizer and configuration
tokenizer = AutoTokenizer.from_pretrained('UpdatedProteinTokenizer', force_download=True)
config = AutoConfig.from_pretrained(pretrained_model_name_or_path="UpdatedProteinTokenizer/config.json", vocab_size=len(tokenizer), force_download=True)

# print tokenizer and config
print(f'tokenizer: {tokenizer}')
print(f'config: {config}')


PST = pytz.timezone('Europe/Zurich')

#instantiate the model
print("start loading model=",datetime.now(PST))
#model = BertLMHeadModel.from_pretrained("bert-base-uncased")
#model = BertForPreTraining.from_pretrained("bert-base-uncased")
model = BertForPreTraining(config=config)

# Load your model's configuration and check the vocab_size parameter. It must match the total number of tokens in your tokenizer’s vocabulary.
print(f"Model's vocab_size: {model.config.vocab_size}")
print(f"Tokenizer's vocab_size: {tokenizer.vocab_size}")

print("Model's vocab size from embeddings:", model.bert.embeddings.word_embeddings.num_embeddings)


# define the arguments for the trainer
training_args = TrainingArguments(
    output_dir='pytorch_finetuned_model',          # output directory
    num_train_epochs=3,              # total # of training epochs
    per_device_train_batch_size=16,  # batch size per device during training (try 16 if needed)
    per_device_eval_batch_size=64,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='pytorch_finetuned_log',     # directory for storing logs
    do_train=True,
    eval_strategy="steps",
    eval_steps=2
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



# Prepare the train_dataset
print("start building train_dataset=", datetime.now(PST))
train_dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer,
    file_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/src/redo_ch/test.txt",
    block_size=128
)

# Check train_dataset input_ids
print("Checking train_dataset input_ids...")
check_input_ids_validity(train_dataset, tokenizer)

# Prepare the eval_dataset
print("start building eval_dataset=", datetime.now(PST))
eval_dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer,
    file_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/src/redo_ch/val.txt",
    block_size=128
)

# Check eval_dataset input_ids
print("Checking eval_dataset input_ids...")
check_input_ids_validity(eval_dataset, tokenizer)

# Initialize the Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)


# Instantiate the trainer
print("start building trainer=",datetime.now(PST))
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,
    data_collator=data_collator,
    eval_dataset=eval_dataset            # evaluation dataset
)

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


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # preds have the same shape as the labels, after the argmax(-1) has been calculated
    # by preprocess_logits_for_metrics
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)

print("finished=",datetime.now(PST))

os.environ["WANDB_PROJECT"] = "test_mlm_nsp"

# define run name
run_name = "debugging"
os.environ["WANDB_RUN_NAME"] = run_name

# Log input_ids right before training to ensure they are all valid
def log_input_ids(data_loader):
    for batch in data_loader:
        input_ids = batch['input_ids']
        if torch.any(input_ids >= tokenizer.vocab_size):
            print("Invalid input_ids detected:", input_ids)
        assert torch.all(input_ids < tokenizer.vocab_size), f"Found input_ids >= vocab size: {input_ids.max().item()}"

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


# If everything is verified, start the training
print("Starting training...")


# now do training
trainer.train()

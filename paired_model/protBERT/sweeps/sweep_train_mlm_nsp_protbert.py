# used env: bug_env
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
from torch.optim import AdamW
from transformers import get_scheduler
import wandb


sweep_config = {
    "method": "bayes",  # You can also use "grid", "random", etc.
    "metric": {
        "name": "avg_eval_loss",  # The metric to optimize
        "goal": "minimize"        # Could be "minimize" or "maximize"
    },
    "parameters": {
        "learning_rate": {
            "min": 1e-7,
            "max": 1e-4
        },
        "num_train_epochs": {
            "values": [3, 5, 10]
        },
        "weight_decay": {
            "min": 0.01,
            "max": 0.8
        }
    }
}

batch_size = 16

# small dataset with input heavyseq[SEP]lightseq with each AA SPACE SEPARATED!!
small_train_dataset_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_train_no_ids_small_SPACE_separated.txt"
small_val_dataset_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_val_no_ids_small_SPACE_separated.txt"

tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )

train_dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer,
    file_path=small_train_dataset_path,
    block_size=512
)

# Prepare the eval_dataset
eval_dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer,
    file_path=small_val_dataset_path,
    block_size=512
)

# Initialize the Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Create DataLoader
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator)
eval_data_loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator)



def train():
    # Initialize a new W&B run with each set of hyperparameters
    with wandb.init() as run:
        config = wandb.config

        # Load the tokenizer and configuration
        model_config = AutoConfig.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False, vocab_size=len(tokenizer))

        # Device setup
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model, optimizer, scheduler with specific hyperparameters
        model = BertForPreTraining(config=model_config).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        num_training_steps = len(train_data_loader) * config.num_train_epochs
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=500,
            num_training_steps=num_training_steps
        )

        for epoch in range(config.num_train_epochs):
            model.train()
            train_loss = 0
            for step, batch in enumerate(train_data_loader):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                if step % 10 == 0:
                    wandb.log({"train_loss": loss.item()})

            avg_train_loss = train_loss / len(train_data_loader)
            wandb.log({"avg_train_loss": avg_train_loss})

            # Evaluation
            model.eval()
            eval_loss = 0
            with torch.no_grad():
                for batch in eval_data_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    eval_loss += outputs.loss.item()

            avg_eval_loss = eval_loss / len(eval_data_loader)
            wandb.log({"avg_eval_loss": avg_eval_loss})

        # Save the model and any important metrics
        model.save_pretrained(f'./model_checkpoint_{run.id}')


sweep_id = wandb.sweep(sweep_config, project="sweep_train_mlm_nsp_protbert")
wandb.agent(sweep_id, train, count=10)  # Number of runs to execute



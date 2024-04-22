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
config = AutoConfig.from_pretrained(pretrained_model_name_or_path="ProteinTokenizer/config.json")
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="ProteinTokenizer")

PST = pytz.timezone('US/Pacific')

#instantiate the model
print("start loading model=",datetime.now(PST))
#model = BertLMHeadModel.from_pretrained("bert-base-uncased")
#model = BertForPreTraining.from_pretrained("bert-base-uncased")
model = BertForPreTraining(config=config)

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
    evaluation_strategy="steps",
    eval_steps=2
)

# prepare the train_dataset
print("start building train_dataset=",datetime.now(PST))
train_dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer,
    file_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/src/redo_ch/test.txt",
    block_size=128
)

print("start building eval_dataset=",datetime.now(PST))
eval_dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer,
    file_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/src/redo_ch/val.txt",
    block_size=128
)

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
run_name = "bert_config"
os.environ["WANDB_RUN_NAME"] = run_name

# now do training
trainer.train()

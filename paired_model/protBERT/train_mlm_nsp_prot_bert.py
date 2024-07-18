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
from torch.optim import AdamW
from transformers import get_scheduler
import wandb

# NSP and MLM tasks using ProtBERT bfd -> https://huggingface.co/Rostlab/prot_bert_bfd
# Input data has to be of the form heavyseq[SEP]lightseq with each AA single space separated!!
# example:
# Q V Q L Q E S G P G L V K P S E T L S L T C T V S G G S I S G F Y W S W I R Q S P G K G L E [SEP] Q V Q L Q E S G P G L V K P S E T L S L T C T V S G G S I S G F Y W S W I R Q S P G K G L E

os.environ['CUDA_VISIBLE_DEVICES'] ='0'

# print used device cpu or cuda/GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

# Load the tokenizer and configuration
tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )
config = AutoConfig.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False, vocab_size=len(tokenizer), force_download=True)

# print tokenizer and config
print(f'tokenizer: {tokenizer}')
print(f'config: {config}')

PST = pytz.timezone('Europe/Zurich')

#instantiate the model
print("start loading model=",datetime.now(PST))
model = BertForPreTraining(config=config)

# Move model to the appropriate device (GPU or CPU)
model.to(device)  # 'device' is determined by torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # confirm model is on the right device
# print(f"Model is on device: {next(model.parameters()).device}")  # It will show cuda:0 if on GPU

# # Load your model's configuration and check the vocab_size parameter. It must match the total number of tokens in your tokenizer’s vocabulary!
# print(f"Model's vocab_size: {model.config.vocab_size}")
# print(f"Tokenizer's vocab_size: {tokenizer.vocab_size}")

# print("Model's vocab size from embeddings:", model.bert.embeddings.word_embeddings.num_embeddings)

batch_size=32
num_train_epochs = 20
learning_rate = 2e-6
weight_decay = 0.3

# Initialize wandb
run_name = f"exp_Small_data_{num_train_epochs}_epochs_lr{learning_rate}_batch_size_{batch_size}_weight_decay_{weight_decay}"

wandb.init(project="paired_model_nsp_mlm_protbert", name=run_name)

output_dir = f"./{run_name}"
logging_dir = f"./{run_name}_logging"


# define the arguments for the trainer
training_args = TrainingArguments(
    output_dir=output_dir,          # output directory
    num_train_epochs=num_train_epochs,              # total # of training epochs
    per_device_train_batch_size=batch_size,  # batch size per device during training 
    per_device_eval_batch_size=batch_size,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=weight_decay,               # strength of weight decay
    logging_dir=logging_dir,     # directory for storing logs
    do_train=True,
    eval_strategy="epoch",
    logging_steps=5
)
 
# # Debugging: Check if all input_ids are within the tokenizer's vocabulary size
# def check_input_ids_validity(dataset, tokenizer):
#     vocab_size = tokenizer.vocab_size
#     for example in dataset:
#         # Extracting input_ids from each example
#         input_ids = example['input_ids'] if isinstance(example, dict) else example.input_ids
#         max_id = max(input_ids)
#         if max_id >= vocab_size:
#             raise ValueError(f"An input_id ({max_id}) exceeds the tokenizer's vocabulary size ({vocab_size}).")
#     print(f"All input_ids are within the vocabulary size.")


# # Debugging: Log input_ids right before training to ensure they are all valid
# def log_input_ids(data_loader):
#     for batch in data_loader:
#         input_ids = batch['input_ids']
#         if torch.any(input_ids >= tokenizer.vocab_size):
#             print("Invalid input_ids detected:", input_ids)
#         assert torch.all(input_ids < tokenizer.vocab_size), f"Found input_ids >= vocab size: {input_ids.max().item()}"


# small dataset with input heavyseq[SEP]lightseq with each AA SPACE SEPARATED!!
small_train_dataset_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_train_no_ids_small_SPACE_separated.txt"
small_val_dataset_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_val_no_ids_small_SPACE_separated.txt"

# FULL dataset with input heavyseq[SEP]lightseq with each AA SPACE SEPARATED!!
full_train_dataset_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_train_no_ids_space_separated.txt"
full_val_dataset_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_val_no_ids_space_separated.txt"

# Prepare the train_dataset
print("start building train_dataset=", datetime.now(PST))
train_dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer,
    file_path=small_train_dataset_path,
    block_size=256
)

# Check train_dataset input_ids
#print("Checking train_dataset input_ids...")
#check_input_ids_validity(train_dataset, tokenizer)

# Prepare the eval_dataset
print("start building eval_dataset=", datetime.now(PST))
eval_dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer,
    file_path=small_val_dataset_path,
    block_size=256
)

# Check eval_dataset input_ids
#print("Checking eval_dataset input_ids...")
#check_input_ids_validity(eval_dataset, tokenizer)

# Initialize the Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Function to create a DataLoader with debugging -> to see the first batch before and after data collator, see if it handles the data correctly
# def create_data_loader_with_debugging(dataset, batch_size, data_collator):
#     data_loader = DataLoader(dataset, batch_size=batch_size, collate_fn=data_collator)
#     for batch in data_loader:
#         input_ids = batch['input_ids']
#         attention_mask = batch['attention_mask']
#         labels = batch['labels']
#         next_sentence_label = batch['next_sentence_label']
        
#         # Print the first batch to debug
#         print("First batch before data collator:")
#         for i, example in enumerate(dataset):
#             if i == batch_size:
#                 break
#             print(f"Example {i}: {tokenizer.decode(example['input_ids'])}")
        
#         print("First batch after data collator:")
#         print(f"Input IDs: {input_ids}")
#         print(f"Attention Mask: {attention_mask}")
#         print(f"Labels: {labels}")
#         print(f"Next Sentence Labels: {next_sentence_label}")
#         break
#     return data_loader


# # Create and debug the DataLoader
# train_data_loader = create_data_loader_with_debugging(train_dataset, batch_size=16, data_collator=data_collator)


# def compute_metrics(preds, labels):
#     true_labels = []
#     true_preds = []
    
#     for i in range(len(labels)):
#         true_labels.extend(labels[i])
#         true_preds.extend(preds[i])

#     print(f"length of true labels {len(true_labels)}")
#     print(f"length of true preds {len(true_preds)}")

#     print(f"true labels {true_labels}")
#     print(f"true predictions {true_preds}")
    
#     # Filter out -100 labels (which are ignored in the evaluation)
#     filtered_labels = [label for label in true_labels if label != -100]
#     filtered_preds = [pred for label, pred in zip(true_labels, true_preds) if label != -100]

#     print(f"length of filtered labels {len(filtered_labels)}")
#     print(f"length of filtered preds {len(filtered_preds)}")

#     print(f"Filtered labels: {filtered_labels}")
#     print(f"Filtered preds: {filtered_preds}")
    
#     precision, recall, f1, _ = precision_recall_fscore_support(filtered_labels, filtered_preds, average='macro', zero_division=0)
#     acc = accuracy_score(filtered_labels, filtered_preds)
    
#     return {
#         'accuracy': acc, 
#         'f1': f1,
#         'precision': precision,
#         'recall': recall
#     }

def compute_metrics(mlm_preds, mlm_labels, nsp_preds, nsp_labels):
    print("Computing metrics...")
    true_mlm_labels = []
    true_mlm_preds = []
    true_nsp_labels = nsp_labels
    true_nsp_preds = nsp_preds
    
    for i in range(len(mlm_labels)):
        true_mlm_labels.extend(mlm_labels[i])
        true_mlm_preds.extend(mlm_preds[i])

    # for i in range(len(nsp_labels)):
    #     true_nsp_labels.extend(nsp_labels[i])
    #     true_nsp_preds.extend(nsp_preds[i])

    print(f"length of true mlm labels {len(true_mlm_labels)}")
    print(f"length of mlm preds {len(true_mlm_preds)}")

    print(f"length of true nsp labels {len(true_nsp_labels)}")
    print(f"length of nsp preds {len(true_nsp_preds)}")

    #print(f"true mlm labels {true_mlm_labels}")
    #print(f"mlm predictions {true_mlm_preds}")

    print(f"true nsp labels {true_nsp_labels}")
    print(f"nsp predictions {true_nsp_preds}")
    
    # Filter out -100 labels (which are ignored in the evaluation)
    filtered_mlm_labels = [label for label in true_mlm_labels if label != -100]
    filtered_mlm_preds = [pred for label, pred in zip(true_mlm_labels, true_mlm_preds) if label != -100]

    print(f"length of filtered mlm labels {len(filtered_mlm_labels)}")
    print(f"length of filtered mlm preds {len(filtered_mlm_preds)}")

    #print(f"Filtered mlm labels: {filtered_mlm_labels}")
    #print(f"Filtered mlm preds: {filtered_mlm_preds}")
    
    mlm_precision, mlm_recall, mlm_f1, _ = precision_recall_fscore_support(filtered_mlm_labels, filtered_mlm_preds, average='macro', zero_division=0)
    acc_mlm = accuracy_score(filtered_mlm_labels, filtered_mlm_preds)

    nsp_precision, nsp_recall, nsp_f1, _ = precision_recall_fscore_support(true_nsp_labels, true_nsp_preds, average='macro', zero_division=0)
    acc_nsp = accuracy_score(true_nsp_labels, true_nsp_preds)

    print("done computing metrics")
    
    return {
        'mlm accuracy': acc_mlm, 
        'mlm f1': mlm_f1,
        'mlm precision': mlm_precision,
        'mlm recall': mlm_recall,
        'nsp accuracy': acc_nsp,
        'nsp precision': nsp_precision,
        'nsp f1': nsp_f1,
        'nsp recall': nsp_recall
    }

#metric = evaluate.load("accuracy", )

# Instantiate the trainer
print("start building trainer=",datetime.now(PST))


# Instantiate the custom trainer
class CustomTrainer(Trainer):
    def log(self, logs: dict):
        super().log(logs)
        if 'loss' in logs:
            print(f"Training loss: {logs['loss']}")
    
    def evaluate_and_log(self):
        metrics = self.evaluate()
        print(f"Evaluation metrics: {metrics}")
        return metrics
    
    def train(self, *args, **kwargs):
        for epoch in range(int(self.args.num_train_epochs)):
            # Call the parent's train method for each epoch
            super().train(*args, **kwargs)
            # Evaluate and log after each epoch
            self.evaluate_and_log()



trainer = CustomTrainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,
    data_collator=data_collator,
    eval_dataset=eval_dataset,           # evaluation dataset
    compute_metrics=compute_metrics,  # pass the compute_metrics function     
)


print("finished=",datetime.now(PST))

# Verifying all input_ids in DataLoader before training
#print("Verifying all input_ids in DataLoader before training...")

#log_input_ids(train_data_loader)


optimizer = AdamW(model.parameters(), lr=learning_rate)

# Create DataLoader for debugging
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator)
eval_data_loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator)


#print("Verifying all input_ids in DataLoader before training...")
#log_input_ids(train_data_loader)  # This will raise an error if any input_id is out of range

# # Testing DataLoader outputs
# for batch in train_data_loader:
#     input_ids = batch['input_ids']
#     attention_mask = batch['attention_mask']
#     print(f"Batch input_ids shape: {input_ids.shape}")  # Should show consistent shapes within a batch
#     print(f"Batch attention mask shape: {attention_mask.shape}")  # Verify attention masks are correct

#     # Check if any input_id in the batch exceeds the tokenizer's vocab size
#     if torch.any(input_ids >= tokenizer.vocab_size):
#         print("Invalid input_ids detected:", input_ids)
#         break


# Example to check how the tokenizer handles one of your sequences
# sample_sequence = "Q V Q L Q E S G P G L V K P S E T L S L T C T V S G G S I S G F Y W S W I R Q S P G K G L E"
# tokens = tokenizer.tokenize(sample_sequence)
# token_ids = tokenizer.convert_tokens_to_ids(tokens)

# print("Tokens:", tokens)
# print("Token IDs:", token_ids)

# turn txt fiel with sequences into a list of sequences
# def read_txt_file(file_path):
#     with open(file_path, 'r') as f:
#         lines = f.readlines()
#     return [line.strip() for line in lines]

# # Load the sequences from the txt file
# train_data = read_txt_file(small_train_dataset_path)

# # Check if all characters are in the tokenizer's vocab
# unique_chars = set(''.join(train_data))  # Assuming train_data is a list of your sequences
# unknown_chars = [char for char in unique_chars if char not in tokenizer.get_vocab()]
# # single space is not in the vocab -> unknown_chars = [' ']
# print("Unknown Characters:", unknown_chars)

# verify token type ids
# def verify_token_type_ids(data_loader):
#     for i, batch in enumerate(data_loader):
#         input_ids = batch['input_ids']
#         token_type_ids = batch.get('token_type_ids', torch.zeros_like(input_ids))  # Assuming default of all zeros if not provided
        
#         # Printing unique token type ids to verify them
#         print(f"Batch {i}: Unique token type ids:", torch.unique(token_type_ids))

#         # Optionally check max input id
#         if input_ids.max().item() >= tokenizer.vocab_size:
#             print("Error: input_ids exceed vocab size")
#             break

#         # Print a message if everything seems fine
#         print(f"Batch {i}: All token type IDs are within expected range.")
        
#         # Breaking after a few batches for demonstration
#         if i > 5:  # Check first 5 batches
#             break

# Run this function to verify token type ids in your DataLoader
#verify_token_type_ids(train_data_loader)

# Prepare the learning rate scheduler
num_training_steps = len(train_data_loader) * training_args.num_train_epochs

print(f"Number of training steps: {num_training_steps}")

lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=num_training_steps
)

# Move the model to the appropriate device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Create directories if they do not exist
os.makedirs(training_args.output_dir, exist_ok=True)
os.makedirs(training_args.logging_dir, exist_ok=True)

# def find_nan(tensor):
#     nan_mask = torch.isnan(tensor)
#     if torch.any(nan_mask):
#         first_nan_idx = torch.where(nan_mask)[0][0].item()
#         return first_nan_idx
#     return None


# # Custom model for debugging -> to check for NaNs in the model outputs
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


# Custom training loop with gradient clipping and detailed logging
print("Starting training...")

################# Custom Training Loop #################

for epoch in range(training_args.num_train_epochs):
    model.train()
    train_loss = 0

    for step, batch in enumerate(train_data_loader):
        batch = {k: v.to(model.device) for k, v in batch.items()}

        # Forward pass
        outputs = model(**batch)
        
        # Extract the loss and logits from the model output
        loss = outputs.loss
        train_loss += loss.item()
        #loss, prediction_logits, seq_relationship_logits = outputs
        
        # Perform backpropagation
        loss.backward()

        # Clip gradients to prevent explosion
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update parameters and learning rate
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        print(f"Epoch: {epoch}, Step: {step}, Training Loss: {loss.item()}")

        # Additional logging
        if step % 10 == 0:
            print(f"Detailed logging at Epoch: {epoch}, Step: {step}")
            print(f"Input IDs: {batch['input_ids']}")
            print(f"Attention Mask: {batch['attention_mask']}")
            print(f"Training Loss: {loss.item()}")
            wandb.log({"train_loss": loss.item(), "epoch": epoch, "step": step})

    avg_train_loss = train_loss / len(train_data_loader)
    print(f"Epoch {epoch} avg Training Loss of this epoch: {avg_train_loss}")
    print(f"Train dataloader length: {len(train_data_loader)}")
    wandb.log({"avg_train_loss": avg_train_loss, "epoch": epoch})
        
    # Evaluation
    model.eval()
    eval_loss = 0
    all_preds = []
    all_labels = []
    nsp_true_labels = []
    nsp_all_preds = []
    #with torch.no_grad():
    for step, batch in enumerate(eval_data_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
            
        outputs = model(**batch)
        loss = outputs.loss
        eval_loss += loss.item()

        prediction_logits = outputs.prediction_logits
        seq_relationship_logits = outputs.seq_relationship_logits
            
        # For MLM
        mlm_preds = torch.argmax(prediction_logits, dim=-1)
        mlm_preds_cpu = mlm_preds.cpu().numpy()

        #print(f"mlm_preds {mlm_preds}")

        mlm_labels = batch['labels']
        mlm_labels_cpu = mlm_labels.cpu().numpy()

        all_preds.extend(mlm_preds.cpu().numpy())
        all_labels.extend(mlm_labels.cpu().numpy())

        #print(f"all_labels {all_labels}")

        # we still need softmax to convert the logits into probabilities

        # For NSP 
        nsp_true_labels_per_batch = batch['next_sentence_label']
        nsp_true_labels_per_batch_to_cpu = nsp_true_labels_per_batch.cpu()
        print(f"nsp_true_labels_per_batch: {nsp_true_labels_per_batch}")
        print(f"length of nsp_true_labels_per_batch: {len(nsp_true_labels_per_batch)}")
        nsp_true_labels.extend(nsp_true_labels_per_batch.cpu().numpy())

        nsp_preds = torch.softmax(seq_relationship_logits, dim=1)
        #print(f"seq_relationship_logits: {seq_relationship_logits}")
        #print(f"nsp_preds: {nsp_preds}")

        nsp_preds_labels = torch.argmax(nsp_preds, dim=1)
        nsp_predictions = nsp_preds_labels.cpu().numpy()
        nsp_all_preds.extend(nsp_preds_labels.cpu().numpy())

        print(f"nsp_preds_labels per batch: {nsp_preds_labels}")
        print(f"length of nsp_preds_labels per batch: {len(nsp_preds_labels)}")


        print(f"true nsp labels (concatenated): {nsp_true_labels}")

        print(f"Epoch: {epoch}, Step: {step}, evaluation Loss: {loss.item()}")

        # compute metrics for each batch
        #print("Computing metrics for batch...")
        #metrics_for_batch = compute_metrics(mlm_preds = mlm_preds_cpu, mlm_labels = mlm_labels_cpu, nsp_preds = nsp_predictions, nsp_labels = nsp_true_labels_per_batch_to_cpu)
        #wandb.log({"metrics_for_batch": metrics_for_batch, "epoch": epoch, "step": step})


        # Additional logging
        if step % 10 == 0:
            print(f"Detailed logging at Epoch: {epoch}, Step: {step}")
            #print(f"Input IDs: {batch['input_ids']}")
            #print(f"Attention Mask: {batch['attention_mask']}")
            print(f"Evaluation Loss: {loss.item()}")
            wandb.log({"eval_loss": loss.item(), "epoch": epoch, "step": step})
    
    avg_eval_loss = eval_loss / len(eval_data_loader)
    # print len(eval_data_loader)
    print(f"Eval dataloader length: {len(eval_data_loader)}")
    print("compute metrics for whole epoch:")
    metrics = compute_metrics(mlm_preds = all_preds, mlm_labels = all_labels, nsp_preds = nsp_all_preds, nsp_labels = nsp_true_labels)
    print("length nsp_all_preds", len(nsp_all_preds))
    print("length nsp_true_labels ", len(nsp_true_labels))
    print(f"Epoch {epoch} avg Evaluation Loss of this epoch: {avg_eval_loss}")
    print(f"Evaluation Metrics: {metrics}")

    # Log evaluation metrics
    wandb.log({"avg eval_loss": avg_eval_loss, "epoch": epoch})
    wandb.log({"metrics for epoch": metrics, "epoch": epoch})
    
    # Save model checkpoint
    checkpoint_dir = os.path.join(training_args.output_dir, f"checkpoint-epoch-{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(optimizer.state_dict(), os.path.join(checkpoint_dir, "optimizer.pt"))
    torch.save(lr_scheduler.state_dict(), os.path.join(checkpoint_dir, "scheduler.pt"))
    
    # Save logs
    with open(os.path.join(training_args.logging_dir, "training_log.txt"), "a") as log_file:
        log_file.write(f"Epoch {epoch}, Training Loss: {train_loss}\n")
        log_file.write(f"Epoch {epoch}, Evaluation Loss: {eval_loss}\n")
        log_file.write(f"Evaluation Metrics: {metrics}\n")

################# Training Loop from transformers #################

# now do training
#trainer.train()

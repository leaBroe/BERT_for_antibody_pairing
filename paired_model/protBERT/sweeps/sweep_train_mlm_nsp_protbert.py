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
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
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
        },
    }
}

batch_size = 32

# small dataset with input heavyseq[SEP]lightseq with each AA SPACE SEPARATED!!
small_train_dataset_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_train_no_ids_small_SPACE_separated.txt"
small_val_dataset_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_val_no_ids_small_SPACE_separated.txt"

# FULL dataset with input heavyseq[SEP]lightseq with each AA SPACE SEPARATED!!
full_train_dataset_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_train_no_ids_space_separated.txt"
full_val_dataset_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_val_no_ids_space_separated.txt"


tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False )

train_dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer,
    file_path=small_train_dataset_path,
    block_size=256
)

# Prepare the eval_dataset
eval_dataset = TextDatasetForNextSentencePrediction(
    tokenizer=tokenizer,
    file_path=small_val_dataset_path,
    block_size=256
)

# Initialize the Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Create DataLoader
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=data_collator)
eval_data_loader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=data_collator)

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
        'acc_nsp': acc_nsp,
        'nsp precision': nsp_precision,
        'nsp f1': nsp_f1,
        'nsp recall': nsp_recall
    }


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

                #forward pass
                outputs = model(**batch)

                # extract loss 
                loss = outputs.loss
                train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                print(f"Epoch: {epoch}, Step: {step}, Training Loss: {loss.item()}")

                if step % 10 == 0:
                    wandb.log({"train_loss": loss.item()})

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
            with torch.no_grad():
                for step, batch in enumerate(eval_data_loader):
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    eval_loss += outputs.loss.item()
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
                    #nsp_true_labels_per_batch_to_cpu = nsp_true_labels_per_batch.cpu()
                    print(f"nsp_true_labels_per_batch: {nsp_true_labels_per_batch}")
                    print(f"length of nsp_true_labels_per_batch: {len(nsp_true_labels_per_batch)}")
                    nsp_true_labels.extend(nsp_true_labels_per_batch.cpu().numpy())

                    nsp_preds = torch.softmax(seq_relationship_logits, dim=1)
                    #print(f"seq_relationship_logits: {seq_relationship_logits}")
                    #print(f"nsp_preds: {nsp_preds}")

                    nsp_preds_labels = torch.argmax(nsp_preds, dim=1)
                    #nsp_predictions = nsp_preds_labels.cpu().numpy()
                    nsp_all_preds.extend(nsp_preds_labels.cpu().numpy())

                    print(f"nsp_preds_labels per batch: {nsp_preds_labels}")
                    print(f"length of nsp_preds_labels per batch: {len(nsp_preds_labels)}")


                    print(f"true nsp labels (concatenated): {nsp_true_labels}")

                    print(f"Epoch: {epoch}, Step: {step}, evaluation Loss: {loss.item()}")

                    # compute metrics for each batch
                    #print("Computing metrics for batch...")
                    #metrics_for_batch = compute_metrics(mlm_preds = mlm_preds_cpu, mlm_labels = mlm_labels_cpu, nsp_preds = nsp_predictions, nsp_labels = nsp_true_labels_per_batch_to_cpu)
                    #wandb.log({"metrics_for_batch": metrics_for_batch, "epoch": epoch, "step": step})

        avg_eval_loss = eval_loss / len(eval_data_loader)
        wandb.log({"avg_eval_loss": avg_eval_loss})
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

        # Save the model and any important metrics
        model.save_pretrained(f'model_checkpoints_small_data_batch_size_8_7.july/model_checkpoint_{run.id}')


sweep_id = wandb.sweep(sweep_config, project="sweep_train_mlm_nsp_protbert")
wandb.agent(sweep_id, train, count=20)  # Number of runs to execute



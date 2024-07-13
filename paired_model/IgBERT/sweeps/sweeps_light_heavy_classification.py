import torch
import os
from torch import nn
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel, BertForSequenceClassification, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import logging
import wandb
from adapters import BnConfig, Seq2SeqAdapterTrainer, AdapterTrainer, BertAdapterModel, init

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Define the sweep configuration
sweep_configuration = {
    'method': 'bayes',  # grid, random
    'metric': {
        'name': 'eval/loss', # metric to optimize
        'goal': 'minimize' # minimize, maximize
    },
    'parameters': {
        'learning_rate': {
            "min": 1e-7,
            "max": 1e-5
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'num_epochs': {
            'values': [10, 50, 100]
        },
        'weight_decay': {
            "min": 0.01,
            "max": 0.8
        },
        'warmup_steps': {
            'values': [500, 1000]
        },
        'max_grad_norm': {
            "min": 0.1,
            "max": 1.0
        }
    }
}

# Define the data files

#small data
train_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_train_with_unpaired_small_space_separated_rm.csv"
val_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_val_with_unpaired_small_space_separated_rm.csv"

# small data 2000 lines in val, 3000 lines in train
#train_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_train_with_unpaired_SMALL_3000_lines_SPACE_sep_space_removed.csv"
#val_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_VAL_SMALL_DATASET_2000_lines_SPACE_sep_space_removed.csv"

# FULL data
#train_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_train_with_unpaired_SPACE_sep_space_removed.csv"
#val_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_val_with_unpaired_SPACE_sep_space_removed.csv"


def load_paired_data(data_file):
    df = pd.read_csv(data_file)
    heavy_chains = df['heavy'].tolist()
    light_chains = df['light'].tolist()
    labels = df['label'].tolist()
    return heavy_chains, light_chains, labels

class PairedChainsDataset(Dataset):
    def __init__(self, heavy_chains, light_chains, labels, tokenizer, max_length):
        self.heavy_chains = heavy_chains
        self.light_chains = light_chains
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.heavy_chains)

    def __getitem__(self, idx):
        heavy = self.heavy_chains[idx]
        light = self.light_chains[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text=heavy, 
            text_pair=light,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(0),  # Remove batch dimension
            'labels': torch.tensor(label, dtype=torch.long)
        }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision = precision_score(labels, preds, average='binary', zero_division=1)
    recall = recall_score(labels, preds, average='binary', zero_division=1)
    f1 = f1_score(labels, preds, average='binary', zero_division=1)
    accuracy = accuracy_score(labels, preds)
    return {
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train():
    import wandb
    from transformers import Trainer, TrainingArguments

    # Initialize a new wandb run
    wandb.init()

    # Set up parameters
    config = wandb.config
    bert_model_name = 'Exscientia/IgBERT'
    num_classes = 2
    max_length = 256
    run_name = f'adapters_FULL_data_lr_{config.learning_rate}_batch_{config.batch_size}_epochs_{config.num_epochs}_weight_decay_{config.weight_decay}_warmup_steps_{config.warmup_steps}_max_grad_norm{config.max_grad_norm}'
    output_dir = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/checkpoints_light_heavy_classification/{run_name}"

    # Create checkpoint directory
    os.makedirs(output_dir, exist_ok=True)

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(bert_model_name)
    model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_classes).to(device)
    init(model)
    model.add_adapter("class_adap", config=BnConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu"))
    model.set_active_adapters("class_adap")
    model.train_adapter("class_adap")

    # Load datasets
    train_heavy, train_light, train_labels = load_paired_data(train_file)
    val_heavy, val_light, val_labels = load_paired_data(val_file)
    train_dataset = PairedChainsDataset(train_heavy, train_light, train_labels, tokenizer, max_length)
    val_dataset = PairedChainsDataset(val_heavy, val_light, val_labels, tokenizer, max_length)

    # Define training arguments
    training_args = TrainingArguments(
        do_train=True,
        do_eval=True,
        output_dir=output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        logging_strategy="epoch",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        report_to="wandb",
        learning_rate=config.learning_rate,
        max_grad_norm=config.max_grad_norm,
        )

    # Initialize Trainer
    trainer = AdapterTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        adapter_names=["class_adap"],
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    #trainer.evaluate()

    # Save the model
    trainer.save_model(output_dir)


sweep_id = wandb.sweep(sweep=sweep_configuration, project="sweeps_classification_heavy_light")

number_of_runs = 20
wandb.agent(sweep_id, train, count=number_of_runs)  


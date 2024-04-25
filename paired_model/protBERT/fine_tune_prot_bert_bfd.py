import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import BertTokenizer, BertForPreTraining, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from transformers.data.datasets.language_modeling import TextDatasetForNextSentencePrediction
import evaluate
from datetime import datetime
import pytz
from sklearn.metrics import accuracy_score
from transformers import AutoConfig


os.environ["WANDB_PROJECT"] = "test_prot_bert_bfd"

# define run name
run_name = "test"
os.environ["WANDB_RUN_NAME"] = run_name

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

# Load the ProtBERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
config = AutoConfig.from_pretrained(pretrained_model_name_or_path="tokenizer/config.json")
model = BertForPreTraining.from_pretrained('Rostlab/prot_bert_bfd', config=config)

model.to(device)
print(f"Model is on device: {next(model.parameters()).device}")

# Define TrainingArguments
training_args = TrainingArguments(
    output_dir='pytorch_finetuned_model',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='pytorch_finetuned_log',
    do_train=True,
    eval_strategy="steps",
    eval_steps=100
)

# Prepare dataset for Next Sentence Prediction
def prepare_dataset(file_path, tokenizer):
    return TextDatasetForNextSentencePrediction(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

train_dataset = prepare_dataset("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/data/small_training_set.txt", tokenizer)
eval_dataset = prepare_dataset("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/data/small_val_set.txt", tokenizer)

# Initialize Data Collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# Accuracy Metric Setup
accuracy_metric = evaluate.load("accuracy")

# def compute_metrics(eval_preds):
#     preds, labels = eval_preds
#     # print labels and preds
#     print(f"labels: {labels}")
#     print(f"preds: {preds}")
#     labels = labels.reshape(-1)
#     preds = preds.reshape(-1)
#     mask = labels != -100
#     labels = labels[mask]
#     preds = preds[mask]
#     return accuracy_metric.compute(predictions=preds, references=labels)


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    print(f"Type of preds: {type(preds)}, Preds structure: {len(preds)} if isinstance(preds, tuple) else 'Not a tuple'")
    print(f"Type of labels: {type(labels)}, Labels structure: {len(labels)} if isinstance(labels, tuple) else 'Not a tuple'")

    # Assuming preds and labels are tuples of (MLM logits, NSP logits) and (MLM labels, NSP labels)
    if isinstance(preds, tuple) and isinstance(labels, tuple):
        mlm_logits, nsp_logits = preds
        mlm_labels, nsp_labels = labels

        # Ensure nsp_logits is a tensor
        if isinstance(nsp_logits, np.ndarray):
            nsp_logits = torch.tensor(nsp_logits)

        # Apply softmax to convert logits to probabilities
        nsp_probs = torch.nn.functional.softmax(nsp_logits, dim=-1)
        nsp_preds = torch.argmax(nsp_probs, dim=-1).detach().cpu().numpy()

        # Ensure nsp_labels is a numpy array for accuracy calculation
        if isinstance(nsp_labels, torch.Tensor):
            nsp_labels = nsp_labels.detach().cpu().numpy()

        # Calculate accuracy for NSP
        nsp_accuracy = accuracy_score(nsp_labels, nsp_preds)

        # print the accuracy
        print(f"NSP Accuracy: {nsp_accuracy}")

        return {
            'accuracy': nsp_accuracy
        }
    else:
        raise TypeError("The predictions and labels must be tuples of (MLM logits/labels, NSP logits/labels).")



# def compute_metrics(eval_preds):
#     preds, labels = eval_preds
#     # print labels and preds
#     print(f"labels: {labels}")
#     print(f"preds: {preds}")

#     # Check the structure and type of preds and labels
#     if isinstance(preds, (list, tuple)) and isinstance(labels, (list, tuple)):
#         # Assuming preds and labels are tuples of (logits, ), unpack them
#         preds = preds[0]
#         labels = labels[0]

#     # Convert to numpy arrays if not already
#     if isinstance(preds, torch.Tensor):
#         preds = preds.detach().cpu().numpy()
#     if isinstance(labels, torch.Tensor):
#         labels = labels.detach().cpu().numpy()

#     # Flatten arrays and apply mask
#     labels = labels.reshape(-1)
#     preds = np.argmax(preds, axis=-1).reshape(-1)
    
#     # Filter out the labels with value -100 (used for ignored tokens in some Hugging Face models)
#     mask = labels != -100
#     labels = labels[mask]
#     preds = preds[mask]

#     return accuracy_metric.compute(predictions=preds, references=labels)


# def compute_metrics(eval_preds):
#     preds, labels = eval_preds
    
#     # Unpack the tuple for MLM and NSP
#     mlm_preds, nsp_preds = preds
#     mlm_labels, nsp_labels = labels

#     # For MLM: We ignore since you are mainly interested in NSP for accuracy calculation
#     # For NSP: Calculate accuracy
#     nsp_preds = np.argmax(nsp_preds, axis=1)  # Select the class with the highest probability

#     # Compute accuracy for NSP
#     nsp_accuracy = accuracy_score(nsp_labels, nsp_preds)

#     return {
#         'accuracy': nsp_accuracy  # You can return more metrics as needed
#     }



# Instantiate the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Start Training
trainer.train()

# Training complete, now run a full evaluation on the eval_dataset
evaluation_results = trainer.evaluate(eval_dataset=eval_dataset)

# Print the evaluation results
print(evaluation_results)



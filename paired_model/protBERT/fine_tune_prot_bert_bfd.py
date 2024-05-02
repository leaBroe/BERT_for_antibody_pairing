import os
import torch
from torch.utils.data import DataLoader
from transformers import BertForPreTraining, Trainer, TrainingArguments
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
from transformers.data.datasets.language_modeling import TextDatasetForNextSentencePrediction
import evaluate
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# Set environment variables for Weights and Biases
os.environ["WANDB_PROJECT"] = "test_prot_bert_bfd"
os.environ["WANDB_RUN_NAME"] = "test_prot_bert_bfd"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Define device based on CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'device: {device}')

# Load the tokenizer and model from Hugging Face's Transformers
tokenizer = AutoTokenizer.from_pretrained('Rostlab/prot_bert_bfd', do_lower_case=False)
model = BertForPreTraining.from_pretrained('Rostlab/prot_bert_bfd')
model.to(device)
print(f"Model is on device: {next(model.parameters()).device}")

# Define training arguments
training_args = TrainingArguments(
    output_dir='pytorch_finetuned_model',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='pytorch_finetuned_log',
    eval_strategy="steps",
    eval_steps=100
)

def prepare_dataset(file_path, tokenizer):
    """Prepare dataset for Next Sentence Prediction."""
    return TextDatasetForNextSentencePrediction(
        tokenizer=tokenizer,
        file_path=file_path,
        block_size=128
    )

#train_dataset = prepare_dataset("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/data/small_training_set.txt", tokenizer)
#eval_dataset = prepare_dataset("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/data/small_val_set.txt", tokenizer)

train_dataset = prepare_dataset("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_train_no_ids.txt", tokenizer)
eval_dataset = prepare_dataset("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_val_no_ids.txt", tokenizer)


data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)

# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
#     acc = accuracy_score(labels, preds)
#     return {
#         'accuracy': acc,
#         'f1': f1,
#         'precision': precision,
#         'recall': recall
#     }

# def compute_metrics(pred):
#     # Extract MLM logits and NSP logits from the predictions tuple
#     # (assuming pred.predictions is a tuple like (mlm_logits, nsp_logits))
#     mlm_logits, nsp_logits = pred.predictions

#     # Extract labels, also assuming it's a tuple (mlm_labels, nsp_labels)
#     mlm_labels, nsp_labels = pred.label_ids

#     # print nsp_labels
#     print(f"nsp_labels: {nsp_labels}")

#     #print nsp_logits
#     print(f"nsp_logits: {nsp_logits}")

#     # Convert NSP logits to probabilities and then get the class predictions
#     nsp_probs = torch.nn.functional.softmax(nsp_logits, dim=-1)
#     nsp_preds = torch.argmax(nsp_probs, dim=-1).detach().cpu().numpy()

#     # Assuming nsp_labels is aligned with the format of nsp_preds
#     if isinstance(nsp_labels, torch.Tensor):
#         nsp_labels = nsp_labels.detach().cpu().numpy()

#     # Compute metrics
#     precision, recall, f1, _ = precision_recall_fscore_support(nsp_labels, nsp_preds, average='binary')
#     acc = accuracy_score(nsp_labels, nsp_preds)
#     return {
#         'accuracy': acc,
#         'f1': f1,
#         'precision': precision,
#         'recall': recall
#     }


def compute_metrics(pred):
    mlm_logits, nsp_logits = pred.predictions

    mlm_labels, nsp_labels = pred.label_ids

    # Check if nsp_logits is a numpy array and convert it to a tensor if it is
    if isinstance(nsp_logits, np.ndarray):
        nsp_logits = torch.tensor(nsp_logits)

    # Applying softmax to convert logits to probabilities
    nsp_probs = torch.nn.functional.softmax(nsp_logits, dim=-1)
    nsp_preds = torch.argmax(nsp_probs, dim=-1).detach().cpu().numpy()

    if isinstance(nsp_labels, torch.Tensor):
        nsp_labels = nsp_labels.detach().cpu().numpy()

    precision, recall, f1, _ = precision_recall_fscore_support(nsp_labels, nsp_preds, average='binary')
    acc = accuracy_score(nsp_labels, nsp_preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }


metric = evaluate.load("accuracy", )


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()
evaluation_results = trainer.evaluate(eval_dataset=eval_dataset)
print(evaluation_results)

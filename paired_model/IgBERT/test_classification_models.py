import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import wandb
from adapters import BnConfig, Seq2SeqAdapterTrainer, AdapterTrainer, BertAdapterModel, init


# Load the test dataset from a CSV file
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


# Define compute_metrics function to evaluate the model's performance
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

#run_name = "FULL_DATA_lr_2e-06_batch_64_epochs_40_weight_decay_0.3_warmup_steps_1000_max_grad_norm_1.0"
#run_name = "adapters_FULL_data_lr_2e-06_batch_64_epochs_10_weight_decay_0.3_warmup_steps_1000_max_grad_norm1.0"
run_name="FULL_DATA_lr_2e-05_batch_64_epochs_10_weight_decay_0.1_warmup_steps_500_max_grad_norm_1.0"
print(f"Running full evaluation of test set for: {run_name}")


# Initialize W&B for logging
wandb.init(
    project="classification_heavy_light_own_classifier",  # Set your W&B project name
    name=f"test_set_eval_{run_name}",   # Name your evaluation run
    config={
        "model": "BertForSequenceClassification",  # Log any relevant model configurations
        "batch_size": 64,
        "max_length": 256
    }
)
# Load the test data

# small test file
#test_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/small_paired_full_seqs_sep_test_with_unpaired.csv"

# full test file
test_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_test_with_unpaired.csv"

test_heavy, test_light, test_labels = load_paired_data(test_file)

# if full model was saved, use this for BertForSequenceClassification.from_pretrained and then load the adapter, if only the adapter was saved, load IgBERT and then load the adapter
# Set up the tokenizer and model
bert_model_name = 'Exscientia/IgBERT'
#adapter_output_dir = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/checkpoints_light_heavy_classification/FULL_DATA_lr_2e-06_batch_64_epochs_40_weight_decay_0.3_warmup_steps_1000_max_grad_norm_1.0"
adapter_output_dir = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/checkpoints_light_heavy_classification/FULL_DATA_lr_2e-05_batch_64_epochs_10_weight_decay_0.1_warmup_steps_500_max_grad_norm_1.0"
max_length = 256
num_classes = 2
batch_size = 64

# Load the tokenizer and model
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = BertForSequenceClassification.from_pretrained(adapter_output_dir, num_labels=num_classes)
#model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_classes)

# Load the adapter (if using adapters)
init(model)
#model.load_adapter("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/checkpoints_light_heavy_classification/FULL_DATA_lr_2e-06_batch_64_epochs_40_weight_decay_0.3_warmup_steps_1000_max_grad_norm_1.0/checkpoint-672080/class_adap")
#model.load_adapter("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/checkpoints_light_heavy_classification/adapters_FULL_data_lr_2e-06_batch_64_epochs_10_weight_decay_0.3_warmup_steps_1000_max_grad_norm1.0/class_adap")
model.load_adapter("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/checkpoints_light_heavy_classification/FULL_DATA_lr_2e-05_batch_64_epochs_10_weight_decay_0.1_warmup_steps_500_max_grad_norm_1.0/checkpoint-50406/class_adap")
model.set_active_adapters("class_adap")
model.to('cuda' if torch.cuda.is_available() else 'cpu')

# Create the test dataset
test_dataset = PairedChainsDataset(test_heavy, test_light, test_labels, tokenizer, max_length)

# print lengths of test_heavy, test_light, test_labels
print(f"Length of test_heavy: {len(test_heavy)}")
print(f"Length of test_light: {len(test_light)}")
print(f"Length of test_labels: {len(test_labels)}")


# Define training arguments (for evaluation)
training_args = TrainingArguments(
    do_eval=True,
    output_dir=adapter_output_dir,
    per_device_eval_batch_size=batch_size,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    report_to="none"  # Do not log to anything else other than W&B
)

# Create the AdapterTrainer for evaluation
trainer = AdapterTrainer(
    model=model,
    args=training_args,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
    adapter_names=["class_adap"]
)

# Evaluate the model on the test dataset
test_results = trainer.evaluate(eval_dataset=test_dataset)
print("Test set results:", test_results)

# Log the test results to W&B
wandb.log(test_results)

# Finish the W&B run
wandb.finish()

# Optionally, test individual predictions on the test set
for heavy, light in zip(test_heavy[:5], test_light[:5]):
    encoding = tokenizer(
        text=heavy, 
        text_pair=light,
        return_tensors='pt',
        max_length=max_length,
        padding='max_length',
        truncation=True
    )
    input_ids = encoding['input_ids'].to(model.device)
    attention_mask = encoding['attention_mask'].to(model.device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

    pairing = "paired" if preds.item() == 1 else "not paired"
    print(f"Predicted pairing for heavy: {heavy} and light: {light}: {pairing}")




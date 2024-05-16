import torch
import os
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import logging
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, classification_report
import wandb

# initialize Weights & Biases

wandb.init(project='paired_classification_heavy_light', name='igbert_full_1e-2')

logging.basicConfig(level=logging.INFO)

# Setup model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print device
print(f"Device: {device}")

# the input data is a csv file with columns 'heavy', 'light', 'label'
# 1 for paired, 0 for not paired
# example:
# heavy,light,label
# GLEWIAYIYFSGSTNYNPSLKSRVTLSVDTSKNQFSLKLSSVTAADSAVYYCARDVGPYNSISPGRYYFDYWGPGTLVTVSS,QSALTQPASVSGSPGQSITISCTGTSSDVGNYNLVSWYQHHPGKAPKLMIYEVSKRPSGISNRFSGSKSGNTASLTISGLQADDEADYYCCSYAGSRILYVFGSGTKVTVL,1
# QLQLQESGPGLVKPSETLSLTCTVSGGSISSSSYYWGWIRQPPGKGLEWIGNFFYSGSTNYNPSLKSRATISLDTSKNELSLKLSSVTAADTAVYYCASNTLMAEATFDYWGQGTLVTVSS,SYEVTQAPSVSVSPGQTASVTCSGDKLDKKYTSWYQQRPGQSPTVVIYQNNKRPSGIPERFSASKSGNTATLTISGTQAVDEADYYCQAWDDSDGVFGPGTTVTVL,0

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
            heavy, light,
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }


class BERTPairedClassifier(nn.Module):
    def __init__(self, bert_model_name, num_classes):
        super(BERTPairedClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.dropout = nn.Dropout(0.1)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes) # linear layer that uses the pooled output to compute logits for the final classification

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        x = self.dropout(pooled_output)
        logits = self.fc(x)
        return logits

# num classes 1 

def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = nn.CrossEntropyLoss()(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    wandb.log({"Train Loss": average_loss})


def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    total_loss = 0
    loss_fn = nn.CrossEntropyLoss()  # Define the loss function

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            # Calculate loss
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()

            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())

    accuracy = accuracy_score(actual_labels, predictions)
    classification_rep = classification_report(
        actual_labels, predictions, zero_division=1
    )
    precision = precision_score(
        actual_labels, predictions, average='binary', zero_division=1
    )
    recall = recall_score(
        actual_labels, predictions, average='binary', zero_division=1
    )
    f1 = f1_score(
        actual_labels, predictions, average='binary', zero_division=1
    )

    average_loss = total_loss / len(data_loader) if len(data_loader) > 0 else 0

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'average_loss': average_loss,
        'classification_report': classification_rep
    }
    wandb.log({
        "Validation Loss": average_loss,
        "Validation Accuracy": accuracy,
        "Validation F1 Score": f1,
        "Validation Precision": precision,
        "Validation Recall": recall
    })
    return metrics



def predict_pairing(heavy, light, model, tokenizer, device, max_length=512):
    model.eval()
    encoding = tokenizer(
        heavy, light,
        return_tensors='pt',
        max_length=max_length,
        padding='max_length',
        truncation=True
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)

    return "paired" if preds.item() == 1 else "not paired"


# Set up parameters
bert_model_name = 'Exscientia/IgBERT'
#bert_model_name = 'Rostlab/prot_bert_bfd'
num_classes = 2
max_length = 512
batch_size = 16
num_epochs = 11
learning_rate = 2e-5

# Load the training and validation data from separate files
#train_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_train_with_unpaired.csv"
#val_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_val_with_unpaired.csv"

train_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_train_with_unpaired_small.csv"
val_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_val_with_unpaired_small.csv"

train_heavy, train_light, train_labels = load_paired_data(train_file)
val_heavy, val_light, val_labels = load_paired_data(val_file)

# Tokenizer and DataLoader setup
tokenizer = BertTokenizer.from_pretrained(bert_model_name)
train_dataset = PairedChainsDataset(train_heavy, train_light, train_labels, tokenizer, max_length)
val_dataset = PairedChainsDataset(val_heavy, val_light, val_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

model = BERTPairedClassifier(bert_model_name, num_classes).to(device)

optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

# Log hyperparameters to Weights & Biases
wandb.config.update({
    "bert_model_name": bert_model_name,
    "num_classes": num_classes,
    "max_length": max_length,
    "batch_size": batch_size,
    "num_epochs": num_epochs,
    "learning_rate": learning_rate,
    "total_steps": total_steps
})


for epoch in range(num_epochs):
    logging.info(f"Epoch {epoch + 1}/{num_epochs}")
    train(model, train_dataloader, optimizer, scheduler, device)
    metrics = evaluate(model, val_dataloader, device)
    logging.info(f"Validation Accuracy: {metrics['accuracy']:.4f}")
    logging.info(f"F1 Score: {metrics['f1']:.4f}")
    logging.info(f"Precision: {metrics['precision']:.4f}")
    logging.info(f"Recall: {metrics['recall']:.4f}")
    logging.info(metrics['classification_report'])

# Save the model
model_name = 'paired_antibody_classifier_full_seqs_2e-5.pth'
torch.save(model.state_dict(), model_name)
#wandb.save('paired_antibody_classifier_state_dict_2e-5.pth')

# Test pairing prediction
test_heavy = "GLEWIAYIYFSGSTNYNPSLKSRVTLSVDTSKNQFSLKLSSVTAADSAVYYCARDVGPYNSISPGRYYFDYWGPGTLVTVSS"
test_light = "QSALTQPASVSGSPGQSITISCTGTSSDVGNYNLVSWYQHHPGKAPKLMIYEVSKRPSGISNRFSGSKSGNTASLTISGLQADDEADYYCCSYAGSRILYVFGSGTKVTVL"
pairing = predict_pairing(test_heavy, test_light, model, tokenizer, device)
print(f"Predicted pairing: {pairing}")

test_heavy = "QLQLQESGPGLVKPSETLSLTCTVSGGSISSSSYYWGWIRQPPGKGLEWIGNFFYSGSTNYNPSLKSRATISLDTSKNELSLKLSSVTAADTAVYYCASNTLMAEATFDYWGQGTLVTVSS"
test_light = "SYEVTQAPSVSVSPGQTASVTCSGDKLDKKYTSWYQQRPGQSPTVVIYQNNKRPSGIPERFSASKSGNTATLTISGTQAVDEADYYCQAWDDSDGVFGPGTTVTVL"
pairing = predict_pairing(test_heavy, test_light, model, tokenizer, device)
print(f"Predicted pairing: {pairing}")

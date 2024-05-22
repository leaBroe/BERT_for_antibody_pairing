import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import logging
import wandb

########################################################################################################################################################################################################################
# sequence classification with AutoModelForSequenceClassification
########################################################################################################################################################################################################################

# the input data is a csv file with columns 'heavy', 'light', 'label' and single space separated AAs
# 1 for paired, 0 for not paired
# example:
# heavy,light,label
# E V Q L V E S G G G L V Q P G G S L R L S C A A S G F T F S S Y D M H W V R Q A T G K G L E W V S A I G T A G D T Y Y P G S G K G R F T I S R E N A K N S L Y L Q M N S L R A G D T A V Y Y C A R A R P V G Y C S G G L G C G A F D I W G Q G T M V T V S S , S Y E L T Q P P S V S V S P G Q T A R I T C S G D A L P K Q Y A Y W Y Q H K P G Q A P V L V I Y K D S E R P S G I P E R F S G S S S G T T V T L T I S G V Q A E D E A D Y Y C Q S A D S S G T Y V V F G G G T K L T V L ,1
# Q V Q L Q E S G P G L V K P S E T L S L T C A V S G Y S I S S G Y Y W G W I R Q P P G K G L E W I G S I Y H S G S T Y Y N P S L K S R V T I S V D T S K N Q F S L K L S S V T A A D T A V Y Y C A R Y C G G D C Y Y V P D Y W G Q G T L V T V S S , S Y E L T Q P P S V S V S P G Q T A S I T C S G D K L G D K Y A C W Y Q Q K P G Q S P V L V I Y Q D S K R P S G I P E R F S G S N S G N T A T L T I S G T Q A M D E A D Y Y C Q A W D S S T E V V F G G G T K L T V L ,0

# Initialize Weights & Biases
run_name = "igbert_test_2e-5_10_epochs"
wandb.init(project='paired_classification_heavy_light', name=run_name)  

logging.basicConfig(level=logging.INFO)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

def load_paired_data(data_file):
    df = pd.read_csv(data_file)
    return df['heavy'].tolist(), df['light'].tolist(), df['label'].tolist()

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
        encoding = self.tokenizer(
            self.heavy_chains[idx], self.light_chains[idx],
            return_tensors='pt',
            max_length=self.max_length,
            padding='max_length',
            truncation=True
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train(model, data_loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in data_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += loss.item()

    average_loss = total_loss / len(data_loader)
    wandb.log({"Train Loss": average_loss})

def evaluate(model, data_loader, device):
    model.eval()
    total_loss = 0
    predictions, actual_labels = [], []
    with torch.no_grad():
        for batch in data_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item()
            _, preds = torch.max(outputs.logits, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(batch['labels'].cpu().tolist())

    metrics = {
        'accuracy': accuracy_score(actual_labels, predictions),
        'precision': precision_score(actual_labels, predictions, average='binary', zero_division=1),
        'recall': recall_score(actual_labels, predictions, average='binary', zero_division=1),
        'f1': f1_score(actual_labels, predictions, average='binary', zero_division=1),
        'average_loss': total_loss / len(data_loader),
        'classification_report': classification_report(actual_labels, predictions, zero_division=1)
    }
    wandb.log(metrics)
    return metrics


bert_model_name = 'Exscientia/IgBERT'
num_classes = 2
max_length = 512
batch_size = 16
num_epochs = 10
learning_rate = 2e-5

tokenizer = BertTokenizer.from_pretrained(bert_model_name)
model = AutoModelForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_classes).to(device)

# Example to check how the tokenizer handles one of your sequences
sample_sequence = "E V Q L V E S G G G L V Q P G G S L R L S C A A S G F T F S S Y D M H W V R Q A T G K G L E W V S A I G T A G D T Y Y P G S G K G R F T I S R E N A K N S L Y L Q M N S L R A G D T A V Y Y C A R A R P V G Y C S G G L G C G A F D I W G Q G T M V T V S S"
tokens = tokenizer.tokenize(sample_sequence)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print("Tokens:", tokens)
print("Token IDs:", token_ids)

train_heavy, train_light, train_labels = load_paired_data("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_train_with_unpaired_small_space_separated_rm.csv")
val_heavy, val_light, val_labels = load_paired_data("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_val_with_unpaired_small_space_separated_rm.csv")

train_dataset = PairedChainsDataset(train_heavy, train_light, train_labels, tokenizer, max_length)
val_dataset = PairedChainsDataset(val_heavy, val_light, val_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

wandb.config.update({ # Log hyperparameters
    "bert_model_name": bert_model_name, "num_classes": num_classes, "max_length": max_length,
    "batch_size": batch_size, "num_epochs": num_epochs, "learning_rate": learning_rate,
    "total_steps": total_steps
})

for epoch in range(num_epochs):
    logging.info(f"Epoch {epoch+1}/{num_epochs}")
    train(model, train_dataloader, optimizer, scheduler, device)
    metrics = evaluate(model, val_dataloader, device)
    logging.info(f"Metrics: {metrics}")

# Save the model
model.save_pretrained(run_name)
#wandb.save('igbert_test_2e-5.pth')

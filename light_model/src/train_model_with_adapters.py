import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForMaskedLM, AdamW
from adapters import AdapterConfig, AutoAdapterModel

# Assuming tokenization and dataset preparation functions are defined elsewhere
from tokenization import AminoAcidDataset, tokenize_and_mask_sequences, load_sequences

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
special_tokens = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}

# Create a dictionary mapping each amino acid and special token to a unique integer
aa_to_id = {aa: i+5 for i, aa in enumerate(amino_acids)}
aa_to_id.update(special_tokens)


# Load and prepare data
training_sequences = load_sequences('/ibmm_data2/oas_database/paired_lea_tmp/light_model/data/training_set_test.txt')
tokenized_training_sequences, training_labels, training_masks = tokenize_and_mask_sequences(training_sequences, aa_to_id, max_len=128)
train_dataset = AminoAcidDataset(tokenized_training_sequences, training_masks, training_labels)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Initialize the tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = AutoAdapterModel.from_pretrained("bert-base-uncased")

# Add a new adapter
config = AdapterConfig.load("pfeiffer", reduction_factor=2)
model.add_adapter("amino_acid_mlm", config=config)
model.train_adapter("amino_acid_mlm")

# Freeze all model weights except adapters
model.freeze_model(False)

model.to(device)

# Prepare optimizer
optimizer = AdamW(model.parameters(), lr=1e-4)

# Training loop
model.train()
for epoch in range(3):  # Number of epochs
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader)}")

test_sequences = load_sequences('/ibmm_data2/oas_database/paired_lea_tmp/light_model/data/test_set_test.txt')
tokenized_test_sequences, test_labels, test_masks = tokenize_and_mask_sequences(test_sequences, aa_to_id, max_len=128)

test_dataset = AminoAcidDataset(tokenized_test_sequences, test_masks, test_labels)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

def evaluate(model, dataloader, device):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():  # No gradients needed
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            
            # Calculate accuracy for masked tokens only
            predictions = torch.argmax(outputs.logits, dim=-1)
            mask = labels != -100  # Only consider masked tokens
            correct_predictions += (predictions == labels)[mask].sum().item()
            total_predictions += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    return avg_loss, accuracy


avg_loss, accuracy = evaluate(model, test_loader, device)
print(f"Average loss: {avg_loss:.2f}")
print(f"Accuracy: {accuracy:.2f}")

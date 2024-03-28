import torch
import torch.optim as optim
from transformers import BertForMaskedLM
from transformers import BertModel, BertConfig
from transformers import AdamW
from tokenization import AminoAcidDataset, tokenize_and_mask_sequences, load_sequences
from torch.utils.data import Dataset, DataLoader

# Check if CUDA (GPU support) is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using {device} device for training.")

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
special_tokens = {'[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, '[MASK]': 4}

# Create a dictionary mapping each amino acid and special token to a unique integer
aa_to_id = {aa: i+5 for i, aa in enumerate(amino_acids)}
aa_to_id.update(special_tokens)

max_len = 128

#training_sequences = load_sequences('/ibmm_data2/oas_database/paired_lea_tmp/light_model/data/training_set_light_seq_100_pident.txt')
#test_sequences = load_sequences('/ibmm_data2/oas_database/paired_lea_tmp/light_model/data/test_set_light_seq_100_pident.txt')

#training_sequences = load_sequences('/ibmm_data2/oas_database/paired_lea_tmp/light_model/data/training_set_test.txt')
#test_sequences = load_sequences('/ibmm_data2/oas_database/paired_lea_tmp/light_model/data/test_set_test.txt')

#training_sequences = load_sequences('/ibmm_data2/oas_database/paired_lea_tmp/light_model/data/training_set_light_seq_70_pident_subset.txt')
#test_sequences = load_sequences('/ibmm_data2/oas_database/paired_lea_tmp/light_model/data/test_set_light_seq_70_pident_subset.txt')

training_sequences = load_sequences('/ibmm_data2/oas_database/paired_lea_tmp/light_model/data/training_set_light_seq_70_pident.txt')
test_sequences = load_sequences('/ibmm_data2/oas_database/paired_lea_tmp/light_model/data/test_set_light_seq_70_pident.txt')

tokenized_training_sequences, training_labels, training_masks = tokenize_and_mask_sequences(training_sequences, aa_to_id, max_len=max_len)
tokenized_test_sequences, test_labels, test_masks = tokenize_and_mask_sequences(test_sequences, aa_to_id, max_len=max_len)

# Create dataset instances
# train_dataset = AminoAcidDataset(tokenized_training_sequences, training_masks)
# test_dataset = AminoAcidDataset(tokenized_test_sequences, test_masks)

# train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

test_dataset = AminoAcidDataset(tokenized_test_sequences, test_masks, test_labels)
train_dataset = AminoAcidDataset(tokenized_training_sequences, training_masks, training_labels)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Step 1: Custom Token Embeddings
# Since BERT is pre-trained with a specific vocabulary (English words and subwords), 
# using it directly on amino acid sequences requires replacing the embedding layer to match your vocabulary. 
# This means initializing a new embedding layer with the correct vocabulary size for your amino acids and special tokens.

# Load pre-trained BERT configuration
config = BertConfig.from_pretrained('bert-base-uncased')


# Adjust the vocab size to match your amino acid vocabulary size
config.vocab_size = len(aa_to_id)

# Initialize BERT model with adjusted config
model = BertModel(config)

# Initialize BertForMaskedLM with the custom config
mlm_model = BertForMaskedLM(config=config).to(device)


optimizer = AdamW(mlm_model.parameters(), lr=5e-5)

mlm_model.train()
num_epochs = 3

for epoch in range(num_epochs):
    total_loss = 0
    for batch in train_loader:
        # Ensure batch items are moved to the device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        # Now correctly accessing labels
        labels = batch['labels'].to(device) if 'labels' in batch else None

        outputs = mlm_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()


def evaluate_mlm(model, dataloader, device):
    model.eval()  # Put model in evaluation mode
    
    total_loss = 0
    total_accuracy = 0
    total_correct_predictions = 0
    total_masked_tokens = 0
    
    with torch.no_grad():  # No need to compute gradients during evaluation
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, 
                            attention_mask=attention_mask, 
                            labels=labels)

            loss = outputs.loss
            total_loss += loss.item()

            # Compute accuracy for masked tokens
            predictions = torch.argmax(outputs.logits, dim=-1)
            mask = labels != -100  # Only consider masked tokens for accuracy calculation

            correct_predictions = (predictions == labels) & mask
            total_correct_predictions += correct_predictions.sum().item()
            total_masked_tokens += mask.sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = (total_correct_predictions / total_masked_tokens) if total_masked_tokens > 0 else 0

    return avg_loss, accuracy


test_loss, test_accuracy = evaluate_mlm(mlm_model, test_loader, device)

print(f"Test Loss: {test_loss:.3f}")
print(f"Test Accuracy: {test_accuracy:.3f}")


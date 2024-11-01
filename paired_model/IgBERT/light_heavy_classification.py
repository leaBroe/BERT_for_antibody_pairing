# used env: lea_env
# use adap_2 when using adapters
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


########################################################################################################################################################################################################################
# sequence classification with BertForSequenceClassification
########################################################################################################################################################################################################################

# Set up parameters
bert_model_name = 'Exscientia/IgBERT'
#bert_model_name = 'Rostlab/prot_bert_bfd'
num_classes = 2
max_length = 256
batch_size = 64
num_epochs = 10
learning_rate = 2e-5
weight_decay = 0.1
max_grad_norm = 1.0
warmup_steps = 500

run_name = f'FULL_DATA_lr_{learning_rate}_batch_{batch_size}_epochs_{num_epochs}_weight_decay_{weight_decay}_warmup_steps_{warmup_steps}_max_grad_norm_{max_grad_norm}'
output_dir = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/checkpoints_light_heavy_classification/{run_name}"

# create checkpoint directory
os.makedirs(output_dir)

# initialize Weights & Biases
wandb.init(project='classification_heavy_light_own_classifier', name=run_name)

logging.basicConfig(level=logging.INFO)

# Setup model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# print device
print(f"Device: {device}")

# the input data is a csv file with columns 'heavy', 'light', 'label' and single space separated AAs
# 1 for paired, 0 for not paired
# example:
# heavy,light,label
# E V Q L V E S G G G L V Q P G G S L R L S C A A S G F T F S S Y D M H W V R Q A T G K G L E W V S A I G T A G D T Y Y P G S G K G R F T I S R E N A K N S L Y L Q M N S L R A G D T A V Y Y C A R A R P V G Y C S G G L G C G A F D I W G Q G T M V T V S S , S Y E L T Q P P S V S V S P G Q T A R I T C S G D A L P K Q Y A Y W Y Q H K P G Q A P V L V I Y K D S E R P S G I P E R F S G S S S G T T V T L T I S G V Q A E D E A D Y Y C Q S A D S S G T Y V V F G G G T K L T V L ,1
# Q V Q L Q E S G P G L V K P S E T L S L T C A V S G Y S I S S G Y Y W G W I R Q P P G K G L E W I G S I Y H S G S T Y Y N P S L K S R V T I S V D T S K N Q F S L K L S S V T A A D T A V Y Y C A R Y C G G D C Y Y V P D Y W G Q G T L V T V S S , S Y E L T Q P P S V S V S P G Q T A S I T C S G D K L G D K Y A C W Y Q Q K P G Q S P V L V I Y Q D S K R P S G I P E R F S G S N S G N T A T L T I S G T Q A M D E A D Y Y C Q A W D S S T E V V F G G G T K L T V L ,0

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


#small data
#train_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_train_with_unpaired_small_space_separated_rm.csv"
#val_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_val_with_unpaired_small_space_separated_rm.csv"

# small data 2000 lines in val, 3000 lines in train
#train_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_train_with_unpaired_SMALL_3000_lines_SPACE_sep_space_removed.csv"
#val_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_VAL_SMALL_DATASET_2000_lines_SPACE_sep_space_removed.csv"

# FULL data
train_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_train_with_unpaired_SPACE_sep_space_removed.csv"
val_file = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/IgBERT/paired_full_seqs_sep_val_with_unpaired_SPACE_sep_space_removed.csv"


train_heavy, train_light, train_labels = load_paired_data(train_file)
val_heavy, val_light, val_labels = load_paired_data(val_file)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


tokenizer = BertTokenizer.from_pretrained(bert_model_name)
train_dataset = PairedChainsDataset(train_heavy, train_light, train_labels, tokenizer, max_length)
# print the first item in the dataset
print(f"first item in train dataset: {train_dataset[0]}")

val_dataset = PairedChainsDataset(val_heavy, val_light, val_labels, tokenizer, max_length)
# print the first item in the dataset
print(f"first item in val dataset: {val_dataset[0]}")

config = BnConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")

model = BertForSequenceClassification.from_pretrained(bert_model_name, num_labels=num_classes).to(device)

init(model)

model.add_adapter("class_adap", config=config)
model.set_active_adapters("class_adap")
model.train_adapter("class_adap")

print(f"number of trainable parameters: {count_trainable_params(model)}")
print_trainable_parameters(model)

training_args = TrainingArguments(
    do_train=True,
    do_eval=True,
    output_dir=output_dir,
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=warmup_steps,
    weight_decay=weight_decay,
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    report_to="wandb",
    learning_rate=learning_rate,
    max_grad_norm=max_grad_norm,  # Gradient clipping
)


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

adapter_output_dir = f"{output_dir}/final_adapter"

# Save the full model
#trainer.save_model(output_dir)
model.save_adapter(adapter_output_dir, "class_adap")
#model.save_pretrained(output_dir)
model.base_model.save_pretrained(output_dir)


# Test pairing prediction
def predict_pairing(heavy, light, model, tokenizer, device, max_length=512):
    model.eval()
    encoding = tokenizer(
        text=heavy, 
        text_pair=light,
        return_tensors='pt',
        max_length=max_length,
        padding='max_length',
        truncation=True
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)

    return "paired" if preds.item() == 1 else "not paired"

test_heavy = "GLEWIAYIYFSGSTNYNPSLKSRVTLSVDTSKNQFSLKLSSVTAADSAVYYCARDVGPYNSISPGRYYFDYWGPGTLVTVSS"
test_light = "QSALTQPASVSGSPGQSITISCTGTSSDVGNYNLVSWYQHHPGKAPKLMIYEVSKRPSGISNRFSGSKSGNTASLTISGLQADDEADYYCCSYAGSRILYVFGSGTKVTVL"
pairing = predict_pairing(test_heavy, test_light, model, tokenizer, device)
print(f"Predicted pairing: {pairing}")

test_heavy = "QLQLQESGPGLVKPSETLSLTCTVSGGSISSSSYYWGWIRQPPGKGLEWIGNFFYSGSTNYNPSLKSRATISLDTSKNELSLKLSSVTAADTAVYYCASNTLMAEATFDYWGQGTLVTVSS"
test_light = "SYEVTQAPSVSVSPGQTASVTCSGDKLDKKYTSWYQQRPGQSPTVVIYQNNKRPSGIPERFSASKSGNTATLTISGTQAVDEADYYCQAWDDSDGVFGPGTTVTVL"
pairing = predict_pairing(test_heavy, test_light, model, tokenizer, device)
print(f"Predicted pairing: {pairing}")


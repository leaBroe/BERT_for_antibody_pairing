import pandas as pd
from transformers import EncoderDecoderModel, BertTokenizer, Seq2SeqTrainer, Seq2SeqTrainingArguments, BertModel
from datasets import Dataset
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import wandb

# Log in to Weights & Biases
#wandb.login()

run_name = "2nd_test_num_beam_2"

wandb.init(project="bert2bert-translation", name=run_name)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(line.strip())
    
    sequences = []
    for entry in data:
        split_entry = entry.split(' [SEP] ')
        if len(split_entry) == 2:
            sequences.append(split_entry)
        else:
            print(f"Skipping invalid entry: {entry}")
    
    df = pd.DataFrame(sequences, columns=['heavy', 'light'])
    return df

# Load training and validation data
train_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/data/paired_full_seqs_sep_train_no_ids_small_SPACE_separated.txt'
val_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/data/paired_full_seqs_sep_val_no_ids_small_SPACE_separated.txt'

train_df = load_data(train_file_path)
val_df = load_data(val_file_path)

# Load the tokenizer and model from Hugging Face
tokenizer = BertTokenizer.from_pretrained("Exscientia/IgBert")

encoder_max_length = 512
decoder_max_length = 512

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(batch["heavy"], padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch["light"], padding="max_length", truncation=True, max_length=decoder_max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # Ignore PAD token in the labels
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch

# Convert the dataframes to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df[['heavy', 'light']])
val_dataset = Dataset.from_pandas(val_df[['heavy', 'light']])

batch_size = 4

train_data = train_dataset.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
)

train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

val_data = val_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
)   

val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)   

# print heavy and light seq from the first example in the training data (train_dataset)
print(f"first example heavy and light seq {train_dataset[0]}")

# Print a few examples from the tokenized dataset
for example in train_data.select(range(1)):
    print(example)

#print(f"first example of tokenized sequences {train_data.select(0)}")

encoder = BertModel.from_pretrained("Exscientia/IgBert")
decoder = BertModel.from_pretrained("Exscientia/IgBert")

# Create the EncoderDecoderModel
bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("Exscientia/IgBert", "Exscientia/IgBert")

print(bert2bert)
print(bert2bert.config)

# Set up the Seq2Seq model configuration
bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
bert2bert.config.eos_token_id = tokenizer.sep_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id
bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size

bert2bert.config.max_length = 512
bert2bert.config.min_length = 100
#bert2bert.config.no_repeat_ngram_size = 3
#bert2bert.config.early_stopping = False
#bert2bert.config.length_penalty = 2.0
bert2bert.config.num_beams = 2

# Custom metrics function
# def compute_metrics(pred):
#     labels_ids = pred.label_ids
#     pred_ids = pred.predictions.argmax(-1)  # Assuming the predictions are logits

#     # Decode predicted and true labels
#     pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
#     labels_ids[labels_ids == -100] = tokenizer.pad_token_id
#     label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

#     # Flatten the lists of tokens for evaluation
#     pred_flat = [item for sublist in [s.split() for s in pred_str] for item in sublist]
#     labels_flat = [item for sublist in [s.split() for s in label_str] for item in sublist]

#     # Ensure both lists have the same length
#     min_len = min(len(pred_flat), len(labels_flat))
#     pred_flat = pred_flat[:min_len]
#     labels_flat = labels_flat[:min_len]

#     # Calculate metrics
#     accuracy = accuracy_score(labels_flat, pred_flat)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels_flat, pred_flat, average='weighted')

#     return {
#         "accuracy": round(accuracy, 4),
#         "precision": round(precision, 4),
#         "recall": round(recall, 4),
#         "f1": round(f1, 4),
#     }



# Set up training arguments and train the model
training_args = Seq2SeqTrainingArguments(
    output_dir="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/model_outputs/results_test",
    logging_dir="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/model_outputs/results_test_logs",
    do_train=True,
    do_eval = True,
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    report_to="wandb",
    run_name=run_name,  
)

# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=bert2bert,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)

# Train the model
trainer.train()

# save the model
bert2bert.save_pretrained(f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/model_outputs/pretrained_model_{run_name}")

# Finish the Weights & Biases run
wandb.finish()

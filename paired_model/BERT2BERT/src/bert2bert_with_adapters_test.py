from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, Seq2SeqTrainingArguments, BertTokenizer, Seq2SeqTrainer
from adapters import BnConfig, Seq2SeqAdapterTrainer, AdapterTrainer, BertAdapterModel, init
import wandb
import torch
import pandas as pd
from datasets import Dataset

#encoder = BertGenerationEncoder.from_pretrained("Exscientia/IgBert")

encoder = BertAdapterModel.from_pretrained("Exscientia/IgBert").base_model

#decoder = BertGenerationDecoder.from_pretrained("Exscientia/IgBert", add_cross_attention=True, is_decoder=True)

decoder = BertAdapterModel.from_pretrained("Exscientia/IgBert", add_cross_attention = True, is_decoder=True).base_model

init(encoder)
init(decoder)

config = BnConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")

encoder.add_adapter("encoder_adapter", config=config)
decoder.add_adapter("decoder_adapter", config=config)

encoder.set_active_adapters("encoder_adapter")
decoder.set_active_adapters("decoder_adapter")

# Activate the adapter
encoder.train_adapter("encoder_adapter")
decoder.train_adapter("decoder_adapter")

print(encoder)
print(decoder)

model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

print(model)

batch_size = 8
run_name="test_with_adapters"

training_args = Seq2SeqTrainingArguments(
    output_dir="/results_test",
    logging_dir="/results_test_logs",
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
    #report_to="wandb",
    run_name=run_name,
)

# Log in to Weights & Biases
wandb.login()

run_name = "test_with_adapters"

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
        if (len(split_entry) == 2):
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


# Initialize the trainer
trainer = Seq2SeqTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
)


input_ids = train_data["input_ids"]
decoder_input_ids = train_data["decoder_input_ids"]
labels = train_data["labels"]
attention_mask = train_data["attention_mask"]
decoder_attention_mask = train_data["decoder_attention_mask"]

input_ids.to(device)
decoder_input_ids.to(device)
labels.to(device)
attention_mask.to(device)
decoder_attention_mask.to(device)

model.to(device)

# train...
loss = model(input_ids=input_ids, decoder_input_ids=decoder_input_ids, labels=labels, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask).loss
loss.backward()

# Train the model
trainer.train()
# environment: adapter_env
from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, Seq2SeqTrainingArguments, BertTokenizer, Seq2SeqTrainer, AutoModel, AutoModelForCausalLM, DataCollatorForSeq2Seq, GenerationConfig
from adapters import BnConfig, Seq2SeqAdapterTrainer, AdapterTrainer, BertAdapterModel, init
import wandb
import torch
import pandas as pd
from datasets import Dataset
import os

# print device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


############################################ BERT2BERT with adapters ############################################

# Load the encoder and decoder from Hugging Face

encoder = AutoModel.from_pretrained("Exscientia/IgBert")
decoder = AutoModelForCausalLM.from_pretrained("Exscientia/IgBert", add_cross_attention = True, is_decoder=True)

#encoder = BertGenerationEncoder.from_pretrained("Exscientia/IgBert")
#encoder = BertAdapterModel.from_pretrained("Exscientia/IgBert").base_model

#decoder = BertGenerationDecoder.from_pretrained("Exscientia/IgBert", add_cross_attention=True, is_decoder=True)
#decoder = BertAdapterModel.from_pretrained("Exscientia/IgBert", add_cross_attention = True, is_decoder=True).base_model

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

print(f"print encoder: {encoder}")
print(f"print decoder: {decoder}")

# Create the model
model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

print(f"print EncoderDecoderModel: {model}")

# Load the tokenizer and model from Hugging Face
tokenizer = BertTokenizer.from_pretrained("Exscientia/IgBert")


batch_size = 32
num_train_epochs = 5

# Set up the run name
run_name="MEDIUM_data_with_adapters_batch_size_32_generate_seq_epochs_5_automodel"

output_dir = f"./{run_name}"
logging_dir = f"./{run_name}_logging"

# Set up the Seq2Seq model configuration
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.encoder.vocab_size

model.config.max_length = 512
model.config.min_length = 50
#model.config.no_repeat_ngram_size = 3
#model.config.early_stopping = False
#model.config.length_penalty = 2.0
#model.config.num_beams = 2


generation_config = GenerationConfig(
    num_return_sequences=1,
    max_length=512,
    min_length=50,

    # sampling
    do_sample=True,
    top_k=100,

    # distribution adjustment
    temperature=0.001,
    repetition_penalty=1,

    vocab_size=model.config.encoder.vocab_size,

    # token ids
    pad_token_id=tokenizer.pad_token_id,
    eos_token_id=tokenizer.sep_token_id,
    decoder_start_token_id=tokenizer.cls_token_id,

    # others
    use_cache=True,
    output_logits=True,
    output_scores=True,
    output_hidden_states=True,
    return_dict_in_generate=True, )






training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    logging_dir=logging_dir,
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    report_to="wandb",
    run_name=run_name,
    generation_config=generation_config,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Create directories if they do not exist
os.makedirs(training_args.output_dir, exist_ok=True)
os.makedirs(training_args.logging_dir, exist_ok=True)

# Log in to Weights & Biases
#wandb.login()


wandb.init(project="bert2bert-translation", name=run_name)

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


# SMALL training and validation data
#train_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/data/paired_full_seqs_sep_train_no_ids_small_SPACE_separated.txt'
#val_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/data/paired_full_seqs_sep_val_no_ids_small_SPACE_separated.txt'

# FULL dataset with input heavyseq[SEP]lightseq with each AA SPACE SEPARATED!!
#train_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_train_no_ids_space_separated.txt"
#val_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_val_no_ids_space_separated.txt"

# MEDIUM dataset with input heavyseq[SEP]lightseq with each AA SPACE SEPARATED!!
train_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/data/medium_sized_train_data_seq2seq.txt"
val_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/data/medium_sized_val_data_seq2seq.txt"


train_df = load_data(train_file_path)
val_df = load_data(val_file_path)


encoder_max_length = 200
decoder_max_length = 200

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(batch["light"], padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch["heavy"], padding="max_length", truncation=True, max_length=decoder_max_length)

    batch["input_ids"] = inputs.input_ids
    batch["attention_mask"] = inputs.attention_mask
    #batch["decoder_input_ids"] = outputs.input_ids
    batch["decoder_attention_mask"] = outputs.attention_mask
    batch["labels"] = outputs.input_ids.copy()

    # Ignore PAD token in the labels
    batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

    return batch



# Convert the dataframes to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df[['heavy', 'light']])
val_dataset = Dataset.from_pandas(val_df[['heavy', 'light']])


train_data = train_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
)

# "decoder_input_ids",
train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_attention_mask", "labels"],
)

val_data = val_dataset.map(
    process_data_to_model_inputs,
    batched=True,
    batch_size=batch_size,
)

# "decoder_input_ids",
val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_attention_mask", "labels"],
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
    data_collator=data_collator,
)


# input_ids = train_data["input_ids"].to(device)
# #decoder_input_ids = train_data["decoder_input_ids"].to(device)
# labels = train_data["labels"].to(device)
# attention_mask = train_data["attention_mask"].to(device)
# decoder_attention_mask = train_data["decoder_attention_mask"].to(device)

# print(f"device: {device}")

# # input_ids.to(device)
# # decoder_input_ids.to(device)
# # labels.to(device)
# # attention_mask.to(device)
# # decoder_attention_mask.to(device)

# model.to(device)

# print(f"model is on device: {next(model.parameters()).device}")
# print(f"input_ids is on device: {input_ids.device}")
# #print(f"decoder_input_ids is on device: {decoder_input_ids.device}")
# print(f"labels is on device: {labels.device}")
# print(f"attention_mask is on device: {attention_mask.device}")
# print(f"decoder_attention_mask is on device: {decoder_attention_mask.device}")

# #output_ids = model.generate(input_ids).to(device)
# #print(f"output_ids: {output_ids}")

# # train...
# # decoder_input_ids=decoder_input_ids,
# loss = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask, decoder_attention_mask=decoder_attention_mask).loss
# loss.to(device)
# print(f"{loss.grad_fn}")
# print(f"loss is on device: {loss.device}")
# #loss.backward()

TORCH_DEBUG=1

model.requires_grad_(True)

# Train the model
trainer.train()

model.to(device)

# test the model with single sequence

#input_prompt = "C A R L F D P F V N D Y S P G T G Y G W L D P W G Q G T P V T V S A "
input_prompt = "S T G V A F M E I N G L R S D D T A T Y F C A I N R V G D R G S N P S Y F Q D W G Q G T R V T V S S "
print(f"input_prompt: {input_prompt}")

inputs = tokenizer(input_prompt, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
input_ids = inputs.input_ids.to(device)
attention_mask = inputs.attention_mask.to(device)

print(f"attention_mask: {attention_mask}")

#input_ids = tokenizer.encode(input_prompt, return_tensors="pt").to(device)
print(f"input_ids: {input_ids}")

# Generate text using the model
generated_seq = model.generate(input_ids=input_ids, 
                               attention_mask=attention_mask, 
                               max_length=100, 
                               output_scores=True, 
                               return_dict_in_generate=True)

# Turn output scores to probabilities
# generated_seq_probs = torch.nn.functional.softmax(generated_seq['scores'][0], dim=-1)

# Print the generated sequences and probabilities
print(f"encoded heavy sequence: {generated_seq}")

# Access the first sequence in the generated sequences
sequence = generated_seq["sequences"][0]

# Convert the generated IDs back to text
generated_text = tokenizer.decode(sequence, skip_special_tokens=True)

print("generated heavy sequence: ", generated_text)


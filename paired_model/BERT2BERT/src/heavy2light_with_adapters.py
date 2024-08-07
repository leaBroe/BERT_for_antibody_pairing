# environment: adap_2
# use pip install git+https://github.com/adapter-hub/adapters.git for installing adapters!
# commands for environment:
# pip install git+https://github.com/adapter-hub/adapters.git
# pip install wandb
# pip install pandas
# pip install datasets
# conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
# conda install chardet
# pip install accelerate -U
# no need to install transformers since adapters will automatically install the correct version needed

from transformers import BertGenerationEncoder, BertGenerationDecoder, EncoderDecoderModel, Seq2SeqTrainingArguments, BertTokenizer, Seq2SeqTrainer, AutoModel, AutoModelForCausalLM, DataCollatorForSeq2Seq, GenerationConfig, DataCollatorWithPadding, AutoTokenizer
from adapters import BnConfig, Seq2SeqAdapterTrainer, AdapterTrainer, BertAdapterModel, init
import wandb
import torch
import pandas as pd
from datasets import Dataset
import os
import datasets
import re


# print device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


############################################ BERT2BERT with adapters ############################################

# Load the encoder and decoder from Hugging Face

############################################ light heavy model / light light model ############################################
# decoder: light model smaller config and encoder: heavy model smaller config
# decoder path epoch 40: /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-56556520
# encoder smaller model epoch 19: /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-117674391

small_heavy_encoder = "/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-117674391"
small_light_decoder =  "/ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-56556520"

model = EncoderDecoderModel.from_encoder_decoder_pretrained(small_heavy_encoder, small_light_decoder , add_cross_attention=True)
init(model)

############################################ big heavy model / big light model ############################################
# decoder: light model big config and encoder: heavy model big config
# decoder path epoch 23: /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_3_roberta_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-32519999
# encoder bigger model epoch 9: /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/FULL_config_3_roberta_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-55740501

# big_heavy_encoder = "/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/FULL_config_3_roberta_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-55740501"
# big_light_decoder =  "/ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_3_roberta_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-32519999"

# model = EncoderDecoderModel.from_encoder_decoder_pretrained(big_heavy_encoder, big_light_decoder , add_cross_attention=True)
# init(model)



#encoder = AutoModel.from_pretrained("Exscientia/IgBert")
#decoder = AutoModelForCausalLM.from_pretrained("Exscientia/IgBert", add_cross_attention = True, is_decoder=True)

# #encoder = BertGenerationEncoder.from_pretrained("Exscientia/IgBert")
# #encoder = BertAdapterModel.from_pretrained("Exscientia/IgBert").base_model

# #decoder = BertGenerationDecoder.from_pretrained("Exscientia/IgBert", add_cross_attention=True, is_decoder=True)
# #decoder = BertAdapterModel.from_pretrained("Exscientia/IgBert", add_cross_attention = True, is_decoder=True).base_model

# init(encoder)
# init(decoder)

# ############################################ add adapters ############################################

config = BnConfig(mh_adapter=True, output_adapter=True, reduction_factor=16, non_linearity="relu")

model.add_adapter("heavy2light_adapter", config=config)
model.set_active_adapters("heavy2light_adapter")
model.train_adapter("heavy2light_adapter")

model.named_parameters

# Loop through all parameters, print and enable gradient computation for 'crossattention' parameters
for name, param in model.named_parameters():
    if re.match(".*crossattention.*", name):
        print(f"Parameter Name: {name}")
        print(f"   Requires Grad: {param.requires_grad}")
        print(f"   Parameter Shape: {param.size()}")
        print(f"   Parameter Device: {param.device}")
        param.requires_grad = True
        print(f"   Requires Grad: {param.requires_grad}")


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

print(f"number of trainable parameters: {count_trainable_params(model)}")
print_trainable_parameters(model)

tokenizer = AutoTokenizer.from_pretrained('/ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-56556520')


batch_size = 64
num_train_epochs = 50
learning_rate = 1e-4
weight_decay = 0.1
temperature = 0.5
num_beams = 5


# Set up the run name
run_name=f"full_Diverse_beam_search_{num_beams}_decoding_temp_{temperature}_max_length_150_early_stopping_true_batch_size_{batch_size}_epochs_{num_train_epochs}_lr_{learning_rate}_wd_{weight_decay}"

output_dir = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/{run_name}"
logging_dir = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/{run_name}_logging"


# Set up the Seq2Seq model configuration
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.pad_token_id = tokenizer.pad_token_id
model.config.vocab_size = model.config.encoder.vocab_size

generation_config = GenerationConfig(
    num_return_sequences=1,
    max_length=512,
    min_length=50,
    early_stopping = True,
    
    #length_penalty = -2.0,
    
    num_beams = num_beams, # before: 3
    num_beam_groups=5,
    diversity_penalty=1.0,

    # # sampling
    # do_sample=True,
    # penalty_alpha=0.6,
    # top_k=4,
    
    # no_repeat_ngram_size = 2,

    # # distribution adjustment
    # temperature=temperature, # before: 0.001 or 0.5
    # repetition_penalty=1,
    # encoder_repetition_penalty=1.5,

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


#generation_config.save_pretrained("generation_config", f"Diverse_beam_search_decoding_beams_{num_beams}.json")

generation_config_name = f"Diverse_beam_search_decoding"
# before generation_config_7.json
generation_config = GenerationConfig.from_pretrained("generation_config", f"{generation_config_name}.json")

############################################ Training Arguments ############################################


training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    logging_dir=logging_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch", # set to epoch for full data
    logging_strategy="epoch",
    #logging_steps=200, # set to 200 for full data
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=weight_decay,
    num_train_epochs=num_train_epochs,
    predict_with_generate=True,
    report_to="wandb",
    run_name=run_name,
    generation_config=generation_config,
    #eval_steps=1, # for full data, set to 2000
    #save_steps=100, # comment out for full data
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Create directories if they do not exist
os.makedirs(training_args.output_dir)
os.makedirs(training_args.logging_dir)

# Log in to Weights & Biases
#wandb.login()


wandb.init(project="heavy2light_translation", name=run_name)

############################################ load the data ############################################


def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(line.strip())

    sequences = []
    for entry in data:
        split_entry = entry.split('[SEP]') # otherwise [SEP] with spaces
        if (len(split_entry) == 2):
            sequences.append(split_entry)
        else:
            print(f"Skipping invalid entry: {entry}")

    df = pd.DataFrame(sequences, columns=['heavy', 'light'])
    return df


# SMALL training and validation data
#train_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/data/paired_full_seqs_sep_train_no_ids_small_SPACE_separated.txt'
#val_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/data/paired_full_seqs_sep_val_no_ids_small_SPACE_separated.txt'
#test_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_test_no_ids_space_separated_SMALL.txt'

# FULL dataset with input heavyseq[SEP]lightseq with each AA SPACE SEPARATED!!
train_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_train_no_ids_space_separated.txt"
val_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_val_no_ids_space_separated.txt"

# MEDIUM dataset with input heavyseq[SEP]lightseq with each AA SPACE SEPARATED!!
#train_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/data/medium_sized_train_data_seq2seq.txt"
#val_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/data/medium_sized_val_data_seq2seq.txt"

# small train and val set without spaces
#train_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_train_no_ids_small.txt"
#val_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_val_no_ids_small.txt"

train_df = load_data(train_file_path)
val_df = load_data(val_file_path)
#test_df = load_data(test_file_path)


encoder_max_length = 200
decoder_max_length = 200

def process_data_to_model_inputs(batch):
    # tokenize the inputs and labels
    inputs = tokenizer(batch["heavy"], padding="max_length", truncation=True, max_length=encoder_max_length)
    outputs = tokenizer(batch["light"], padding="max_length", truncation=True, max_length=decoder_max_length)

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
#test_dataset = Dataset.from_pandas(test_df[['heavy', 'light']])


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

# test_data = test_dataset.map(
#     process_data_to_model_inputs,   
#     batched=True,
#     batch_size=batch_size,
# )   

# # "decoder_input_ids",
# test_data.set_format(
#     type="torch", columns=["input_ids", "attention_mask", "decoder_attention_mask", "labels"],
# )




# print heavy and light seq from the first example in the training data (train_dataset)
print(f"first example heavy and light seq {train_dataset[0]}, {train_dataset[1]}")

# Print a few examples from the tokenized dataset
for example in train_data.select(range(1)):
    print(example)


# Initialize the trainer
trainer = Seq2SeqAdapterTrainer(
    model=model,
    tokenizer=tokenizer,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
    adapter_names=["heavy2light_adapter"],
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

#TORCH_DEBUG=1

#model.requires_grad_(True)

#print(f"trainer.get_train_dataloader().collate_fn: {trainer.get_train_dataloader().collate_fn}")

# Train the model
trainer.train()
#trainer.evaluate()

model.to(device)

model_output_path = os.path.join("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/", run_name)
adapter_output_path = f"{model_output_path}/final_adapter"

os.makedirs(adapter_output_path)

# Save model
model.save_pretrained(model_output_path)
# Save adapter
model.save_adapter(adapter_output_path, "heavy2light_adapter")
#decoder.save_adapter(output_path, "decoder_adapter")

# Load your test data
test_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_test_no_ids_space_separated_SMALL.txt'

# small test file with no spaces
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_test_no_ids_small.txt"

test_df = load_data(test_file_path)


# extract the light sequences from test_df
heavy_sequences = test_df["heavy"]
true_light_sequences = test_df["light"]

#print("light_sequences: ", light_sequences)
#print(f"length of light sequences {len(light_sequences)}")

generated_light_seqs = []

# Iterate through each sequence in the test dataset
for i in range(50):
    inputs = tokenizer(heavy_sequences[i], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)

    generated_seq = model.generate(input_ids=input_ids, 
                               attention_mask=attention_mask, 
                               max_length=150, # small small: 150, big big: 130
                               output_scores=True, 
                               return_dict_in_generate=True,
                                   generation_config=generation_config)
    
    # Access the first element in the generated sequence
    sequence = generated_seq["sequences"][0]

    # Convert the generated IDs back to text
    generated_text = tokenizer.decode(sequence, skip_special_tokens=True)
    true_light_seq = true_light_sequences[i]

    print("decoded light sequence: ", generated_text)
    print("true light sequence: ", true_light_seq)

    generated_light_seqs.append(generated_text)
    
    generated_text = generated_text.replace(" ", "")
    true_light_seq = true_light_seq.replace(" ", "")
    
    # Determine the length of the shorter sequence
    min_length = min(len(generated_text), len(true_light_seq))
    print(f"min_length:, {min_length}")
    
    # Calculate the number of matches
    matches = sum(res1 == res2 for res1, res2 in zip(generated_text, true_light_seq))
    print(f"matches:, {matches}")

    
    # Calculate the similarity percentage
    similarity_percentage = (matches / min_length) * 100
    
    print(f"similarity percentage: {similarity_percentage}")


print("generated_light_seqs:")
# print each generated sequence on new line
for seq in generated_light_seqs:
    print(seq)

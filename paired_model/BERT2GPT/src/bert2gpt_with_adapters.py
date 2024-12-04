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

from transformers import EncoderDecoderModel, Seq2SeqTrainingArguments, GenerationConfig, AutoTokenizer, PreTrainedTokenizerBase
from adapters import BnConfig, Seq2SeqAdapterTrainer, init
from typing import Dict, List, Union
import wandb
import torch
import pandas as pd
from datasets import Dataset
import os
import re


# print device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


############################################ BERT2GPT with adapters ############################################

# Load the encoder and decoder from Hugging Face

############################################ light heavy model / light light model ############################################
# decoder: light model smaller config and encoder: heavy model smaller config
# decoder path epoch 40: /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-56556520
# encoder smaller model epoch 19: /ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-117674391

#small_heavy_encoder="/storage/homefs/lb24i892/bert2gpt_translation/bert_encoder_heavy/checkpoint-117674391"
small_heavy_encoder = "/ibmm_data2/oas_database/paired_lea_tmp/heavy_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-117674391"
#small_light_decoder =  "/ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-56556520"

#light_gpt_decoder = "/storage/homefs/lb24i892/gpt_light_model_unpaired/model_outputs/small_gpt2_test_light_seqs_unp/checkpoint-74"
light_gpt_decoder ="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/gpt_decoder/checkpoint-74"

model = EncoderDecoderModel.from_encoder_decoder_pretrained(small_heavy_encoder, light_gpt_decoder , add_cross_attention=True)
init(model)

# Count the layers (modules)
layer_count = sum(1 for _ in model.modules())
print(f"Total number of layers (including activations, etc.): {layer_count}")

# Count layers with trainable parameters
trainable_layer_count = sum(1 for _ in model.children() if list(_.parameters()))
print(f"Number of trainable layers: {trainable_layer_count}")

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

#tokenizer = AutoTokenizer.from_pretrained('/ibmm_data2/oas_database/paired_lea_tmp/light_model/src/redo_ch/FULL_config_4_smaller_model_run_lr5e-5_500epochs_max_seq_length_512/checkpoint-56556520')
#tokenizer = AutoTokenizer.from_pretrained("/storage/homefs/lb24i892/bert2gpt_translation/bert_encoder_heavy/checkpoint-117674391")

# Load BERT tokenizer for encoder
bert_tokenizer = AutoTokenizer.from_pretrained(small_heavy_encoder)

# Load GPT tokenizer for decoder
gpt_tokenizer = AutoTokenizer.from_pretrained(light_gpt_decoder)

batch_size = 64
num_train_epochs = 30
learning_rate = 1e-4
weight_decay = 0.1
#temperature = 0.1
num_beams = 5
top_p = 0.9
penalty_alpha = 0.8
top_k = 2

repetition_penalty = 1.2
dola_layers = "high"
#dola_layers=[1,340]
max_length = 120

flag = "PLAbDab"
dataset = "healthy_human"
dataset_size = "full"
decoding = "DoLa"
translation_model="bert2gpt"
#decoding="nucleus"
#decoding="contrastive"

# Set up the run name
#run_name=f"{dataset_size}_{flag}_{dataset}_{dola_layers}_{decoding}_max_length_{max_length}_rep_penalty_{repetition_penalty}_num_epochs_{num_train_epochs}"
run_name=f"{translation_model}_{decoding}_layers_{dola_layers}_rep_penal_{repetition_penalty}_{dataset_size}_{flag}_{dataset}_max_length_{max_length}_num_epochs_{num_train_epochs}_3"


print(f"Training model with run_name: {run_name}")

#output_dir = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/{run_name}"
#output_dir = f"/storage/homefs/lb24i892/bert2gpt_translation/model_outputs/{run_name}"
output_dir = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/model_checkpoints/{run_name}"
#logging_dir = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/{run_name}_logging"
#logging_dir = f"/storage/homefs/lb24i892/bert2gpt_translation/model_outputs/{run_name}_logging"
logging_dir = f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/model_checkpoints/{run_name}_logging"


# # Set up the Seq2Seq model configuration
# model.config.decoder_start_token_id = tokenizer.cls_token_id
# model.config.eos_token_id = tokenizer.sep_token_id
# model.config.pad_token_id = tokenizer.pad_token_id
# model.config.vocab_size = model.config.encoder.vocab_size

# Set up the Seq2Seq model configuration
model.config.decoder_start_token_id = gpt_tokenizer.bos_token_id
model.config.eos_token_id = gpt_tokenizer.eos_token_id
model.config.pad_token_id = gpt_tokenizer.pad_token_id

# Ensure the decoder's vocab size is correctly set
model.config.decoder.vocab_size = model.config.decoder.vocab_size


generation_config = GenerationConfig(
    num_return_sequences=1,
    max_length=max_length,
    min_length=50,
    # Distribution and repetition control
    # temperature=0.7,  # Set as needed
    # dola_layers=dola_layers,  # Define this variable appropriately
    # repetition_penalty=repetition_penalty,  # Define this variable appropriately

    # Token IDs from GPT tokenizer
    pad_token_id=gpt_tokenizer.pad_token_id,
    eos_token_id=gpt_tokenizer.eos_token_id,
    decoder_start_token_id=gpt_tokenizer.bos_token_id,

    # Output and caching options
    use_cache=True,
    output_scores=True,
    output_hidden_states=True,
    return_dict_in_generate=True
)

generation_config.save_pretrained("generation_config", f"gpt_generation_config.json")

generation_config_name = f"gpt_generation_config"
#generation_config_name = f"Diverse_beam_search_decoding"
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



class CustomDataCollatorForSeq2Seq:
    def __init__(
        self,
        tokenizer_encoder: PreTrainedTokenizerBase,
        tokenizer_decoder: PreTrainedTokenizerBase,
        model=None,
        padding: Union[bool, str] = True,
        max_length: int = None,
        label_pad_token_id: int = -100,
    ):
        self.tokenizer_encoder = tokenizer_encoder
        self.tokenizer_decoder = tokenizer_decoder
        self.model = model
        self.padding = padding
        self.max_length = max_length
        self.label_pad_token_id = label_pad_token_id

    def __call__(self, batch: List[Dict[str, Union[str, torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # Extract source and target texts
        source_texts = [example['source_text'] for example in batch]
        target_texts = [example['target_text'] for example in batch]

        # Tokenize source texts (encoder inputs)
        encoder_inputs = self.tokenizer_encoder(
            source_texts,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Tokenize target texts (decoder inputs and labels)
        decoder_inputs = self.tokenizer_decoder(
            target_texts,
            padding=self.padding,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Prepare labels
        labels = decoder_inputs['input_ids'].clone()
        labels[labels == self.tokenizer_decoder.pad_token_id] = self.label_pad_token_id  # Mask padding tokens

        # Create the batch dictionary
        batch = {
            'input_ids': encoder_inputs['input_ids'],
            'attention_mask': encoder_inputs['attention_mask'],
            'labels': labels
        }

        return batch
    

# Instantiate the custom data collator
data_collator = CustomDataCollatorForSeq2Seq(
    tokenizer_encoder=bert_tokenizer,
    tokenizer_decoder=gpt_tokenizer,
    model=model,  # Pass your model if it requires decoder_input_ids
    padding='longest',  # Options: True, False, 'longest', 'max_length'
    max_length=512,  # Set your desired max_length
    label_pad_token_id=-100  # Tokens with this ID are ignored in loss computation
)

# Create directories if they do not exist
os.makedirs(training_args.output_dir)
os.makedirs(training_args.logging_dir)

# Log in to Weights & Biases
#wandb.login()


wandb.init(project="bert2gpt_translation", name=run_name)

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

# on vader
# SMALL dataset with input heavyseq[SEP]lightseq with each AA SPACE SEPARATED with dataset: human healthy, no vaccine, no disease and PLAbDab unique paired seqs
train_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/train_val_test_datasets/plabdab_human_healthy_no_vac_allocated_train_no_identifiers_spaces_small.txt"

# on ubelix
#train_file_path="/storage/homefs/lb24i892/bert2gpt_translation/plabdab_human_healthy_no_vac_allocated_train_no_identifiers_spaces_small.txt"

# on vader
val_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/train_val_test_datasets/plabdab_human_healthy_no_vac_allocated_val_no_identifiers_spaces_small.txt"

# on ubelix
#val_file_path="/storage/homefs/lb24i892/bert2gpt_translation/plabdab_human_healthy_no_vac_allocated_val_no_identifiers_spaces_small.txt"


train_df = load_data(train_file_path)
val_df = load_data(val_file_path)
#test_df = load_data(test_file_path)


encoder_max_length = 200
decoder_max_length = 200

def process_data_to_model_inputs(batch):
    # Tokenize the encoder inputs using the BERT tokenizer
    encoder_inputs = bert_tokenizer(
        batch["heavy"],
        padding="max_length",
        truncation=True,
        max_length=encoder_max_length,
        return_tensors="pt"
    )

    # Tokenize the decoder inputs (labels) using the GPT tokenizer
    decoder_inputs = gpt_tokenizer(
        batch["light"],
        padding="max_length",
        truncation=True,
        max_length=decoder_max_length,
        return_tensors="pt"
    )

    # Assign encoder inputs
    batch["input_ids"] = encoder_inputs["input_ids"]
    batch["attention_mask"] = encoder_inputs["attention_mask"]

    # Assign decoder inputs
    batch["decoder_input_ids"] = decoder_inputs["input_ids"]
    batch["decoder_attention_mask"] = decoder_inputs["attention_mask"]


    # Prepare labels
    labels = decoder_inputs["input_ids"].clone()
    labels[labels == gpt_tokenizer.pad_token_id] = -100  # Mask padding tokens in labels
    batch["labels"] = labels



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
    tokenizer=gpt_tokenizer,
    args=training_args,
    train_dataset=train_data,
    eval_dataset=val_data,
    data_collator=data_collator,
    adapter_names=["heavy2light_adapter"],
)


# Train the model
trainer.train()
#trainer.evaluate()

model.to(device)

#model_output_path = os.path.join("/storage/homefs/lb24i892/bert2gpt_translation/model_outputs/", run_name)
model_output_path = os.path.join("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2GPT/model_checkpoints/", run_name)
adapter_output_path = f"{model_output_path}/final_adapter"

os.makedirs(adapter_output_path)

# Save model
model.save_pretrained(model_output_path)
# Save adapter
model.save_adapter(adapter_output_path, "heavy2light_adapter")
#decoder.save_adapter(output_path, "decoder_adapter")

# Load your test data
#test_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_test_no_ids_space_separated_SMALL.txt'

# on vader: small test dataset all human paired, no duplicates + PLAbDab unique paired sequences
test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_all_disease_plabdab/train_val_test_datasets/human_healthy_all_diseases_plabdab_test_no_identifiers_spaces_small.txt"
# on ubelix:
#test_file_path="/storage/homefs/lb24i892/bert2gpt_translation/plabdab_human_healthy_no_vac_allocated_test_no_identifiers_spaces_small.txt"

# small test file with no spaces
#test_file_path = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_test_no_ids_small.txt"

test_df = load_data(test_file_path)


# extract the light sequences from test_df
heavy_sequences = test_df["heavy"]
true_light_sequences = test_df["light"]

#print("light_sequences: ", light_sequences)
#print(f"length of light sequences {len(light_sequences)}")

generated_light_seqs = []

# average similarity percentage
total_similarity_percentage = []

# Iterate through each sequence in the test dataset
for i in range(50):
    inputs = bert_tokenizer(heavy_sequences[i], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
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
    generated_text = gpt_tokenizer.decode(sequence, skip_special_tokens=True)
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

    # add similarity percentage to list
    total_similarity_percentage.append(similarity_percentage)


# print average similarity percentage
similarity_percentage = sum(total_similarity_percentage) / len(total_similarity_percentage)
print(f"average similarity percentage: {similarity_percentage}")

print("generated_light_seqs:")
# print each generated sequence on new line
for seq in generated_light_seqs:
    print(seq)

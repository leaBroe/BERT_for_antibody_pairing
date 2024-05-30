import pandas as pd
from transformers import EncoderDecoderModel, BertTokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments, BertModel, BertTokenizer
from datasets import Dataset
from torch.utils.data import DataLoader
import torch

# used env: class_env

#input data:
#heavy [SEP] light with single space separation
#Q V Q L Q E S G P G L V K P S E T L S L T C N V S G Y S I S S G Y Y W G W I R Q P P G K G L E W I G I I Y Q N G H S F Y N P S L K S R A A L S V A A S K N Q F S L N L R S V T A A D T A V Y F C A R V A S N A P T D W G Q G T L V T V S S [SEP] Q S A L T Q P P S A S G S L G Q S V T I S C T G S S S D V G G Y A Y V S W Y Q Q H P G K A P K V V I Y E V T K R P S G V P E R F S G S K S G N T A S L T V S G L Q A E D E A D Y Y C I S Y A G A N K L G V F G G G T K L T V L

# print device

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

# Load the tokenizer and model from local directories
#tokenizer = BertTokenizerFast.from_pretrained("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/IgBERT_models_HF/Exscientia_IgBert_tokenizer")
tokenizer = BertTokenizer.from_pretrained("Exscientia/IgBert")


# Load encoder and decoder models separately
#encoder = BertModel.from_pretrained("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/IgBERT_models_HF/Exscientia_IgBert_model")
#decoder = BertModel.from_pretrained("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/IgBERT_models_HF/Exscientia_IgBert_model")

encoder = BertModel.from_pretrained("Exscientia/IgBert")
decoder = BertModel.from_pretrained("Exscientia/IgBert")

# Create the EncoderDecoderModel
model = EncoderDecoderModel(encoder=encoder, decoder=decoder)
#model = EncoderDecoderModel.from_encoder_decoder_pretrained("Exscientia/IgBert", "Exscientia/IgBert")

max_length = None

def tokenize_function(examples):
    return tokenizer(examples['heavy'], examples['light'], padding="longest", truncation=False, max_length=128, return_tensors="pt") # return_tensors="pt": ensures that the output is in PyTorch tensor format, which is required for training

# Convert the dataframes to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df[['heavy', 'light']])
val_dataset = Dataset.from_pandas(val_df[['heavy', 'light']])



tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True) # batched=True: indicates that the function should process batches of examples at a time rather than one example at a time. This can be more efficient because it allows vectorized operations and reduces the overhead of repeated function calls.

# Convert the tokenized datasets to PyTorch datasets
tokenized_train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])
tokenized_val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_type_ids'])


# Print a few examples from the tokenized dataset
for example in tokenized_train_dataset.select(range(1)):
    print(example)

"""
Output:{
    'heavy': 'Q V Q L Q E S G P R L V K P S E T L S L T C A V S G G S I S S K N W W S W L R Q S P E K G L E W I G E V Y E T G T A N H N P S L T R R L A L S V D K S R N Q F H L N L S S V T A A D T G V Y F C A R G I V D R R P L Y F D N W G Q G I L V T V S S',
    'light': 'D I Q V T Q S P S S L S A S V G D R V T I T C R A S Q N I N T N L N W Y Q Q K A G R A P K V L I H G A S T L Q S G V P V R F S G S G S G T E F T L T I N N M E P E D V A T Y Y C Q Q S H N S R T F G Q G T R V E M K',
    'input_ids': [2, 18, 8, 18, 5, 18, 9, 10, 7, 16, 13, 5, 8, 12, 16, 10, 9, 15, 5, 10, 5, 15, 23, 6, 8, 10, 7, 7, 10, 11, 10, 10, 12, 17, 24, 24, 10, 24, 5, 13, 18, 10, 16, 9, 12, 7, 5, 9, 24, 11, 7, 9, 8, 20, 9, 15, 7, 15, 6, 17, 22, 17, 16, 10, 5, 15, 13, 13, 5, 6, 5, 10, 8, 14, 12, 10, 13, 17, 18, 19, 22, 5, 17, 5, 10, 10, 8, 15, 6, 6, 14, 15, 7, 8, 20, 19, 23, 6, 13, 7, 11, 8, 14, 13, 13, 16, 5, 20, 19, 14, 17, 24, 7, 18, 7, 11, 5, 8, 15, 8, 10, 10, 3, 14, 11, 18, 8, 15, 18, 10, 16, 10, 10, 5, 10, 6, 10, 8, 7, 14, 13, 8, 15, 11, 15, 23, 13, 6, 10, 18, 17, 11, 17, 15, 17, 5, 17, 24, 20, 18, 18, 12, 6, 7, 13, 6, 16, 12, 8, 5, 11, 22, 7, 6, 10, 15, 5, 18, 10, 7, 8, 16, 8, 13, 19, 10, 7, 10, 7, 10, 7, 15, 9, 19, 15, 5, 15, 11, 17, 17, 21, 9, 16, 9, 14, 8, 6, 15, 20, 20, 23, 18, 18, 10, 22, 17, 10, 13, 15, 19, 7, 18, 7, 15, 13, 8, 9, 21, 12, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
}
"""

# Explanation:
# The 'heavy' and 'light' fields contain the original text sequences.
# The 'input_ids' field shows the tokenized version of these sequences, where each word or token has been converted into its corresponding ID from the tokenizer's vocabulary.
# The 'token_type_ids' field shows that the tokens from 'heavy' are marked with 0, and the tokens from 'light' are marked with 1.
# The 'attention_mask' indicates which tokens are actual data (1) and which are padding (0).

# Set up the Seq2Seq model
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.max_length = 128
model.config.no_repeat_ngram_size = 3
model.config.early_stopping = False

batch_size = 4

# Set up training arguments and train the model
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    tokenizer=tokenizer
)

trainer.train()


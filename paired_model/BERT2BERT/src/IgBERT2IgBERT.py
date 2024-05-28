import pandas as pd
from transformers import EncoderDecoderModel, BertTokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments, BertModel
from datasets import Dataset

#input data:
#heavy [SEP] light with single space separation
#Q V Q L Q E S G P G L V K P S E T L S L T C N V S G Y S I S S G Y Y W G W I R Q P P G K G L E W I G I I Y Q N G H S F Y N P S L K S R A A L S V A A S K N Q F S L N L R S V T A A D T A V Y F C A R V A S N A P T D W G Q G T L V T V S S [SEP] Q S A L T Q P P S A S G S L G Q S V T I S C T G S S S D V G G Y A Y V S W Y Q Q H P G K A P K V V I Y E V T K R P S G V P E R F S G S K S G N T A S L T V S G L Q A E D E A D Y Y C I S Y A G A N K L G V F G G G T K L T V L

#Function to read and process data from a text file
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
tokenizer = BertTokenizerFast.from_pretrained("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/IgBERT_models_HF/Exscientia_IgBert_tokenizer")

# Load encoder and decoder models separately
encoder = BertModel.from_pretrained("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/IgBERT_models_HF/Exscientia_IgBert_model")
decoder = BertModel.from_pretrained("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/IgBERT_models_HF/Exscientia_IgBert_model")

# Create the EncoderDecoderModel
model = EncoderDecoderModel(encoder=encoder, decoder=decoder)

def tokenize_function(examples):
    return tokenizer(examples['heavy'], examples['light'], padding="max_length", truncation=True, max_length=128)

# Convert the dataframes to Hugging Face datasets
train_dataset = Dataset.from_pandas(train_df[['heavy', 'light']])
val_dataset = Dataset.from_pandas(val_df[['heavy', 'light']])

tokenized_train_dataset = train_dataset.map(tokenize_function, batched=True)
tokenized_val_dataset = val_dataset.map(tokenize_function, batched=True)

# Set up the Seq2Seq model
model.config.decoder_start_token_id = tokenizer.cls_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.max_length = 128
model.config.no_repeat_ngram_size = 3
model.config.early_stopping = True

# Set up training arguments and train the model
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
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


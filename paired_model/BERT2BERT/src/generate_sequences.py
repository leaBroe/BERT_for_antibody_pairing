from transformers import EncoderDecoderModel, AutoTokenizer, GenerationConfig
import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")

# Load the model and tokenizer
model_name = "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/FULL_data_max_length_150_early_stopping_false_heavy2light_with_adapters_batch_size_64_epochs_50_lr_0.0001_weight_decay_0.1"
tokenizer = AutoTokenizer.from_pretrained("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/FULL_data_max_length_150_early_stopping_false_heavy2light_with_adapters_batch_size_64_epochs_50_lr_0.0001_weight_decay_0.1/checkpoint-420050")
model = EncoderDecoderModel.from_pretrained(model_name)
model.to(device)

generation_config_name = "generation_config_4"
generation_config = GenerationConfig.from_pretrained("generation_config", f"{generation_config_name}.json")


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


# Load your test data
test_file_path = '/ibmm_data2/oas_database/paired_lea_tmp/paired_model/train_test_val_datasets/heavy_sep_light_seq/paired_full_seqs_sep_test_no_ids_space_separated_SMALL.txt'
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
                               max_length=150, 
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
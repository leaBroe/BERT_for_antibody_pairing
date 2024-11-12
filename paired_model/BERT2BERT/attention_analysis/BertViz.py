from transformers import AutoTokenizer, AutoModel
import torch
from bertviz import model_view

from extract_attention_scores_old import initialize_model_and_tokenizer

# tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
# model = AutoModel.from_pretrained("Helsinki-NLP/opus-mt-en-de", output_attentions=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# heavy2light 60 epochs diverse beam search beam = 5
run_name="full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1"
model_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1"
tokenizer_path=f"{model_path}/checkpoint-504060"
adapter_path=f"{model_path}/final_adapter"
generation_config_path=model_path
adapter_name="heavy2light_adapter"

heavy_input = "Q V Q Q P G A E L V R S G A S V K M S C K A S G Y T F T S Y N M H W V K Q T P G Q G L E W I G Y I Y P G N G G T I Y N Q K F K G K A T L T A D T S S S T A N M Q I S S L T S E D S A V Y F C A R G D Y R N D P F D F W G Q G T T L T V S S"
light_output = "D V Q I I Q T T A S L S A S V G E T V T I T C R A S E H I Y S Y L A W Y Q Q K Q G K S P Q L L V Y S A K T L A E G V P S R F S G S G S G T Q F S L K I N S L Q P E D F G S Y Y C Q H H Y D T P R T F G G G T K L E I R"

model, tokenizer, generation_config = initialize_model_and_tokenizer(model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name)

model.to(device)
print(f"model is on device: {model.device}")

inputs = tokenizer(heavy_input, return_tensors="pt", add_special_tokens=True).to(device)
encoder_input_ids = inputs.input_ids.to(device)
print(f"Device now: {device}")

print(f"encoder_input_ids are on device: {encoder_input_ids.device}")

with tokenizer.as_target_tokenizer():
    decoder_input_ids = tokenizer(light_output, return_tensors="pt", add_special_tokens=True).input_ids.to(device)


# Generate sequences
generated_outputs = model.generate(
    encoder_input_ids,
    attention_mask=inputs['attention_mask'],
    generation_config=generation_config,
    return_dict_in_generate=True
)

generated_ids = generated_outputs.sequences  # Extract the sequences tensor

# Convert generated IDs to tokens
generated_ids = generated_ids[0]  # Assuming batch size is 1
generated_tokens = tokenizer.convert_ids_to_tokens(generated_ids)

# Pass inputs and generated_ids to the model to get attentions
outputs = model(
    input_ids=inputs['input_ids'],
    attention_mask=inputs['attention_mask'],
    decoder_input_ids=generated_ids.unsqueeze(0),  # Add batch dimension
    output_attentions=True,
    return_dict=True
)
    


#outputs = model(input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids)

encoder_text = tokenizer.convert_ids_to_tokens(encoder_input_ids[0])
#decoder_text = generated_tokens

model_view(
    encoder_attention=outputs.encoder_attentions,
    decoder_attention=outputs.decoder_attentions,
    cross_attention=outputs.cross_attentions,
    encoder_tokens=encoder_text,
    decoder_tokens=generated_tokens
)


from transformers import AutoTokenizer, AutoModel
import torch
from extract_attention_scores_old import initialize_model_and_tokenizer

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

##### Extract embeddings from the encoder #####

# Tokenize the heavy input
inputs = tokenizer(heavy_input, return_tensors="pt", add_special_tokens=True).to(device)

# Pass inputs through the encoder
with torch.no_grad():
    encoder_outputs = model.encoder(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        output_hidden_states=True,
        return_dict=True
    )

# Extract embeddings (e.g., from the last hidden state)
encoder_hidden_states = encoder_outputs.hidden_states[-1] # Shape: (batch_size, seq_length, hidden_size)

# Aggregate embeddings (e.g., mean pooling)
encoder_embedding = encoder_hidden_states.mean(dim=1)  # Shape: (batch_size, hidden_size)


##### Extract embeddings from the decoder #####

with tokenizer.as_target_tokenizer():
    decoder_inputs = tokenizer(light_output, return_tensors="pt", add_special_tokens=True).to(device)

# Pass inputs through the decoder
with torch.no_grad():
    decoder_outputs = model.decoder(
        input_ids=decoder_inputs['input_ids'],
        attention_mask=decoder_inputs['attention_mask'],
        encoder_hidden_states=encoder_outputs.last_hidden_state,
        encoder_attention_mask=inputs['attention_mask'],
        output_hidden_states=True,
        return_dict=True
    )

# Extract decoder embeddings
decoder_hidden_states = decoder_outputs.hidden_states[-1]
decoder_embedding = decoder_hidden_states.mean(dim=1)



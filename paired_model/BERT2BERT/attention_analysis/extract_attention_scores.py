
# General Libraries
import re
import pandas as pd
from tqdm import tqdm
import numpy as np
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json
import gc

# Notebook Libraries
import math
from transformers import BertTokenizer, BertModel, EncoderDecoderModel, AutoTokenizer, GenerationConfig
import adapters
from adapters import BertAdapterModel, init
import torch
import torch.nn as nn
from transformers.modeling_utils import PreTrainedModel , PretrainedConfig

from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def initialize_model_and_tokenizer(model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name):
    """
    Initialize the model, tokenizer, and generation configuration.
    
    Args:
        model_path (str): Path to the model.
        tokenizer_path (str): Path to the tokenizer.
        adapter_path (str): Path to the adapter.
        generation_config_path (str): Path to the generation configuration.
        device (torch.device): Device to run the model on.
    
    Returns:
        model (EncoderDecoderModel): Initialized model.
        tokenizer (AutoTokenizer): Initialized tokenizer.
        generation_config (GenerationConfig): Generation configuration.
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    model = EncoderDecoderModel.from_pretrained(model_path, output_attentions=True) # output_attentions=True to get attention scores
    model.to(device)
    init(model)
    print(f"model is on device: {model.device}")
    #model.to(device)
    model.load_adapter(adapter_path)
    model.set_active_adapters(adapter_name)
    generation_config = GenerationConfig.from_pretrained(generation_config_path)
    
    return model, tokenizer, generation_config



# heavy2light 60 epochs diverse beam search beam = 5
run_name="full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1"
model_path="/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1"
tokenizer_path=f"{model_path}/checkpoint-504060"
adapter_path=f"{model_path}/final_adapter"
generation_config_path=model_path
adapter_name="heavy2light_adapter"

model, tokenizer, generation_config = initialize_model_and_tokenizer(model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name)

# update generation config to return attentions (if needed)
generation_config.output_attentions = True


def format_attention(attention, layers=None, heads=None):
    # Select specified layers if any
    if layers:
        attention = [attention[layer_index] for layer_index in layers]
    
    squeezed = []
    for layer_attention in attention:
        # Expected shape: batch_size x num_heads x 1 x seq_len
        if len(layer_attention.shape) != 4 or layer_attention.shape[2] != 1:
            raise ValueError(
                "The attention tensor does not have the expected dimensions. "
                "Expected shape [batch_size, num_heads, 1, seq_len]."
            )
        
        # Remove the singleton dimension at position 2
        layer_attention = layer_attention.squeeze(2)  # Shape now: batch_size x num_heads x seq_len
        
        # Select specified heads if any
        if heads:
            layer_attention = layer_attention[:, heads, :]
        
        # If multiple items in the batch, average across batch dimension
        layer_attention = layer_attention.mean(dim=0)  # Averaging across batch dimension
        squeezed.append(layer_attention)
    
    # Stack across layers; shape: num_layers x num_heads x seq_len
    return torch.stack(squeezed)

input_text = "QSALTQPASVSGSPGQSITISCTGTSSDVGGYNYVSWYQQHPGKAPKLMIYDVSNRPSGVSNRFSGSKSGNTASLTISGLQAEDEADYYCSSYTSSSTLVFGGGTKLTVL"

#inputs = tokenizer(input_text, return_tensors='pt') ##COME ERA
inputs = tokenizer(input_text, return_tensors='pt').to(device)
#attention_mask=inputs['attention_mask']
model.to(device)
outputs = model.generate(inputs['input_ids'], output_attentions = True, return_dict_in_generate=True) 

print(type(outputs))
print(outputs.keys())

attention = outputs['decoder_attentions'][-1] # shape: (num_heads, seq_len, seq_len)


for i, attention in enumerate(attention):
    print(f"Shape of attention tensor {i}: {attention.shape}")

inputs=inputs['input_ids'][0] #tokens id in the tokenizer vocab, same len as the input_test before adding whitespace and special char
tokens = tokenizer.convert_ids_to_tokens(inputs)  # Convert input ids to token strings

last_layer_attention = format_attention(attention, layers=[-1], heads=[0])  # Shape: (num_layers, seq_len, seq_len)
# # Average across heads to get a single attention matrix
# # Shape after mean: (seq_len, seq_len)
# attention_scores_avg = torch.mean(attention, dim=0).detach().numpy()

# # Example: Getting attention scores for each input token (sequence positions)
# for i, attention_scores in enumerate(attention_scores_avg):
#     print(f"Token {i}: Attention scores to all other tokens -> {attention_scores}")







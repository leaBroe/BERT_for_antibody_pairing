
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

# get information about the model
#print(model.config)


# update generation config to return attentions (if needed)
generation_config.output_attentions = True

def format_attention(attention, layers=None, heads=None):
    if layers:
        attention = [attention[layer_index] for layer_index in layers]
    squeezed = []
    for layer_attention in attention:
        # 1 x num_heads x seq_len x seq_len
        if len(layer_attention.shape) != 4:
            raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set output_attentions=True when initializing your model.")
        layer_attention = layer_attention.squeeze(0)
        if heads:
            layer_attention = layer_attention[heads]
        squeezed.append(layer_attention)
    print(squeezed[0].shape)
    # num_layers x num_heads x seq_len x seq_len
    return torch.stack(squeezed)



input_text = "QSALTQPASVSGSPGQSITISCTGTSSDVGGYNYVSWYQQHPGKAPKLMIYDVSNRPSGVSNRFSGSKSGNTASLTISGLQAEDEADYYCSSYTSSSTLVFGGGTKLTVL"
#input_text = "Q A G L T Q P P S V S K G L R Q T A T L T C T G N S N N V G N Q G A A W L Q Q H Q G H P P K L L S Y R N N N R P S G I S E R L S A S R S G N T A S L T I T G L Q P E D E A D Y Y C S A W D S S L S A W V F G G G T K L T V L"

# #inputs = tokenizer(input_text, return_tensors='pt') ##COME ERA
# inputs = tokenizer(input_text, return_tensors='pt').to(device)
# #attention_mask=inputs['attention_mask']
# model.to(device)
# attention_mask = inputs['attention_mask'].to(device)
# outputs = model.generate(inputs['input_ids'], output_attentions = True, return_dict_in_generate=True, attention_mask=attention_mask, generation_config=generation_config)

# print(type(outputs))
# print(outputs.keys())

# attention = outputs['decoder_attentions'][-1][0] #attention from the last layer of the decoder

# print(type(outputs['decoder_attentions'][-1]))
# print(outputs['decoder_attentions'][-1])

# inputs=inputs['input_ids'][0] #tokens id in the tokenizer vocab, same len as the input_test before adding whitespace and special char
# tokens = tokenizer.convert_ids_to_tokens(inputs)  # Convert input ids to token strings

# last_head_attentions=attention[:, -1, :, :] #attention from the last head of the decoder

# #last_attentions=format_attention(attention, layers=[-1], heads=[-1]) #extract attention from last layer from last head (29,15)

# print(last_head_attentions.shape)

# for i in list(last_head_attentions[0][0]):
#     print(i)

# # Average across heads to get a single attention matrix
# # Shape after mean: (seq_len, seq_len)
# attention_scores_avg = torch.mean(attention, dim=0).detach().numpy()

# # Example: Getting attention scores for each input token (sequence positions)
# for i, attention_scores in enumerate(attention_scores_avg):
#     print(f"Token {i}: Attention scores to all other tokens -> {attention_scores}")


def attention_score_to_cls_token_and_to_all(input_text,model,device):

    ''' Retrieve attention from model outputs attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True
    is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer)
    of shape (batch_size, num_heads, sequence_length, sequence_length).

    Outputs:
    df_all_vs_all: pandas.Dataframe containing attention score for all tokens versus all tokens
    att_to_cls: pandas.Series with attention score of all tokens related to the CLS token
    df_att_to_cls_exp: att_to_cls in pandas.Dataframe format with score in exponential format

    '''
    model = model.to(device)
    tokenizer = BertTokenizer.from_pretrained('Rostlab/prot_bert_bfd')
    #inputs = tokenizer(input_text, return_tensors='pt') ##COME ERA
    inputs = tokenizer(input_text, return_tensors='pt').to(device)
    attention_mask=inputs['attention_mask']
    outputs = model(inputs['input_ids'],attention_mask) #len 30 as the model layers #outpus.attentions
    attention = outputs[-1] #outpus has 2 dimensions, the second one are the attentions outputs.attentions
    inputs=inputs['input_ids'][0] #tokens id in the tokenizer vocab, same len as the input_test before adding whitespace and special char
    tokens = tokenizer.convert_ids_to_tokens(inputs)  # Convert input ids to token strings
    last_attentions=format_attention(attention, layers=[-1], heads=[-1]) #extract attention from last layer from last head (29,15)
    #(last_attentions[0][0]) #extracting from stacked list
    att_score=[]
    for i in list(last_attentions[0][0]): #extracting every list of attention
        #att_score.append((i).detach().numpy()) ### COME ERA
        att_score.append((i).detach().cpu().numpy())
        
    #len(att_score) # same len as number of tokens (len text + special char)
    x = np.stack(att_score, axis=0 )
    m = np.asmatrix(x) # attention score as matrix
    names = [_ for _ in tokens]
    df_all_vs_all = pd.DataFrame(m, index=names, columns=names) #attention score matrix all tokens vs all tokens
    #df_all_vs_all # df with attentio score all tokens vs all tokens
    att_to_cls = df_all_vs_all.loc['[CLS]'] #attention score only vs cls token
    #att_to_cls
    attention_to_cls_exp=[]

    att_to_cls.apply(lambda x: float(x))
    for i in att_to_cls:
        attention_to_cls_exp.append(math.exp(float(i)))

    token_att_num=tuple(zip(names,attention_to_cls_exp)) #pairs : token - a.score

    df_att_to_cls_exp=pd.DataFrame(token_att_num,columns=['token','attention'])
    #df_att_to_cls_exp['index'] = range(1, len(df_att_to_cls_exp) + 1) # starting count from 1 in the index
    df_att_to_cls_exp = df_att_to_cls_exp[df_att_to_cls_exp.token != '[CLS]']
    df_att_to_cls_exp = df_att_to_cls_exp[df_att_to_cls_exp.token != '[SEP]']
    #df_att_to_cls_exp

    torch.cuda.empty_cache()
    gc.collect()
    return df_all_vs_all,att_to_cls,df_att_to_cls_exp


test = attention_score_to_cls_token_and_to_all(input_text,model,device)

print(test[0])

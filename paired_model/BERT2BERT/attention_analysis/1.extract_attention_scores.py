import gc
import math
import json
import torch
import logging
import pandas as pd
import numpy as np
import torch.nn as nn
from transformers import (EncoderDecoderModel, 
                          AutoTokenizer, GenerationConfig)
from transformers.modeling_utils import PreTrainedModel, PretrainedConfig
from adapters import BertAdapterModel, init
from tqdm import tqdm
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AttentionAnalyzer:
    def __init__(self, model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name):
        self.device = device
        self.model, self.tokenizer, self.generation_config = self.initialize_model_and_tokenizer(
            model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name)
        
    def initialize_model_and_tokenizer(self, model_path, tokenizer_path, adapter_path, generation_config_path, device, adapter_name):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = EncoderDecoderModel.from_pretrained(model_path, output_attentions=True)
        model.to(device)
        init(model)
        model.load_adapter(adapter_path)
        model.set_active_adapters(adapter_name)
        generation_config = GenerationConfig.from_pretrained(generation_config_path)

        # Log the device on which the model is located
        logger.info(f"Model is on device: {model.device}")

        return model, tokenizer, generation_config
    
    def format_attention(self, attention, layers=None, heads=None):
        if layers:
            attention = [attention[layer_index] for layer_index in layers]
        
        squeezed = []
        for layer_attention in attention:
            # 1 x num_heads x seq_len x seq_len
            if len(layer_attention.shape) != 4:
                raise ValueError("The attention tensor does not have the correct number of dimensions. Make sure you set output_attentions=True when initializing your model.")
            
            # Squeeze the batch dimension
            layer_attention = layer_attention.squeeze(0)
            
            # Check the shape after squeezing
            print(f"Shape after squeezing: {layer_attention.shape}")
            assert len(layer_attention.shape) == 3, f"Expected 3D tensor, got {layer_attention.shape}"
            
            if heads:
                layer_attention = layer_attention[heads]
            
            # Check the shape after selecting heads
            print(f"Shape after selecting heads: {layer_attention.shape}")
            squeezed.append(layer_attention)
        
        # num_layers x num_heads x seq_len x seq_len
        stacked_attention = torch.stack(squeezed)
        
        # Check the final shape
        print(f"Final stacked shape: {stacked_attention.shape}")
        assert len(stacked_attention.shape) == 4, f"Expected 4D tensor, got {stacked_attention.shape}"
        
        return stacked_attention
        
    def attention_score_to_cls_token_and_to_all(self, input_text, model, tokenizer, device):
        ''' 
        Retrieve attention from model outputs attentions (tuple(torch.FloatTensor), optional, returned when output_attentions=True
        is passed or when config.output_attentions=True) — Tuple of torch.FloatTensor (one for each layer)
        of shape (batch_size, num_heads, sequence_length, sequence_length).

        Outputs:
        df_all_vs_all: pandas.Dataframe containing attention score for all tokens versus all tokens
        att_to_cls: pandas.Series with attention score of all tokens related to the CLS token
        df_att_to_cls_exp: att_to_cls in pandas.Dataframe format with score in exponential format
        '''
        model.to(device)
        tokenizer = tokenizer
        inputs = tokenizer(input_text, return_tensors='pt').to(device)
        attention_mask = inputs['attention_mask']
        outputs = model.generate(inputs['input_ids'], output_attentions=True, return_dict_in_generate=True, attention_mask=attention_mask, generation_config=generation_config)

        # print generated sequence
        print(tokenizer.decode(outputs[0][0], skip_special_tokens=True))

        # Extract encoder and decoder attention scores
        encoder_attentions = outputs.encoder_attentions  # tuple of tensors, one per layer
        decoder_attentions = outputs.decoder_attentions  # tuple of tensors, one per layer
        cross_attentions = outputs.cross_attentions      # tuple of tensors, one per layer (encoder-decoder attention)

        # Example: Process one layer of encoder attention
        # encoder_attentions[0] would be of shape (batch_size, num_heads, seq_len, seq_len)
        print("Last Encoder Layer Attention shape:", encoder_attentions[-1].shape)
        print("Last Decoder Layer Attention shape:", decoder_attentions[-1][0].shape)
        print("Last Cross Layer Attention shape:", cross_attentions[-1][0].shape)

        # Extract the attention scores from the last layer of the decoder 
        #attention = outputs["decoder_attentions"][-1][0]  # Access the first element of the tuple
        
        # take the mean of the attention scores across all heads
        #attention = torch.mean(attention, dim=1)

        # Convert input IDs to tokens
        inputs = inputs['input_ids'][0]
        tokens = tokenizer.convert_ids_to_tokens(inputs)
        
        # Extract attention from the last layer and last head
        last_attentions = self.format_attention(encoder_attentions, layers=[-1], heads=[-1])

        print(last_attentions.shape)

        # Average over heads to get a single attention matrix
        att_score = last_attentions.mean(dim=1).squeeze(0).detach().cpu().numpy()

        print(att_score.shape)
        
        # att_score=[]
        # for i in list(last_attentions[0][0]): #extracting every list of attention
        #     att_score.append((i).detach().cpu().numpy())

        # # access first element of att_score
        # att_score = att_score
        # att_score = np.array(att_score)

        # Convert to matrix
        m = np.asmatrix(att_score)
        
        # Create a DataFrame for all tokens versus all tokens
        names = [_ for _ in tokens]
        df_all_vs_all = pd.DataFrame(m, index=names, columns=names)
        
        # Extract attention scores related to the CLS token
        att_to_cls = df_all_vs_all.loc['[CLS]']
        
        # Convert attention scores to exponential format
        attention_to_cls_exp = [math.exp(float(i)) for i in att_to_cls]
        
        # Create a DataFrame for attention scores related to the CLS token
        token_att_num = tuple(zip(names, attention_to_cls_exp))
        df_att_to_cls_exp = pd.DataFrame(token_att_num, columns=['token', 'attention'])
        df_att_to_cls_exp = df_att_to_cls_exp[df_att_to_cls_exp.token != '[CLS]']
        df_att_to_cls_exp = df_att_to_cls_exp[df_att_to_cls_exp.token != '[SEP]']
        
        # Clear cache and return results
        torch.cuda.empty_cache()
        gc.collect()
        return df_all_vs_all, att_to_cls, df_att_to_cls_exp



if __name__ == "__main__":
    # Define configuration parameters
    config = {
        # heavy2light 60 epochs diverse beam search beam = 5
        "run_name": "full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1",
        "model_path": "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1",
        "tokenizer_path": "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1/checkpoint-504060",
        "adapter_path": "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1/final_adapter",
        "generation_config_path": "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1",
        "adapter_name": "heavy2light_adapter"
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    analyzer = AttentionAnalyzer(
        model_path=config["model_path"],
        tokenizer_path=config["tokenizer_path"],
        adapter_path=config["adapter_path"],
        generation_config_path=config["generation_config_path"],
        device=device,
        adapter_name=config["adapter_name"]
    )

    input_text = "S Y E L T Q P P S V S V S P G Q T A S I T C S G D K L G D K Y A C W Y Q Q K P G Q S P V L V I Y Q D S K R P S G I P E R F S G S N S G N T A T L T I S G T Q A M D E A D Y Y C Q A W D S S T V V F G G G T K L T V L"
    df_all_vs_all, att_to_cls, df_att_to_cls_exp = analyzer.attention_score_to_cls_token_and_to_all(input_text)

    # Save the results
    df_all_vs_all.to_csv(f"attention_scores_{config['run_name']}.csv")
    df_att_to_cls_exp.to_csv(f"attention_scores_to_cls_exp_{config['run_name']}.csv")
    att_to_cls.to_csv(f"attention_scores_to_cls_{config['run_name']}.csv")




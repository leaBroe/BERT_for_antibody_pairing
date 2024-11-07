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
        logger.info("Starting attention score computation.")
        self.model.to(self.device)
        inputs = self.tokenizer(input_text, return_tensors='pt').to(self.device)
        
        # Generate sequences
        generated_outputs = self.model.generate(
            inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            generation_config=self.generation_config,
            return_dict_in_generate=True
        )
        generated_ids = generated_outputs.sequences  # Extract the sequences tensor

        # Convert generated IDs to tokens
        generated_ids = generated_ids[0]  # Assuming batch size is 1
        generated_tokens = self.tokenizer.convert_ids_to_tokens(generated_ids)

        # Pass inputs and generated_ids to the model to get attentions
        outputs = self.model(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            decoder_input_ids=generated_ids.unsqueeze(0),  # Add batch dimension
            output_attentions=True,
            return_dict=True
        )

        # Extract the decoder attentions
        decoder_attentions = outputs.decoder_attentions  # List of tensors

        # Extract attention from the last layer
        last_decoder_attention = decoder_attentions[-1]  # Shape: (batch_size, num_heads, tgt_seq_len, tgt_seq_len)

        # Average over heads if necessary
        # Here, we average over heads to get a single attention matrix
        attention_matrix = last_decoder_attention.mean(dim=1).squeeze(0).detach().cpu().numpy()  # Shape: (tgt_seq_len, tgt_seq_len)

        # Create DataFrame with generated tokens as labels
        names = generated_tokens
        df_all_vs_all = pd.DataFrame(attention_matrix, index=names, columns=names)

        # Extract attention scores related to the [CLS] token
        if '[CLS]' in names:
            att_to_cls = df_all_vs_all.loc['[CLS]']
        else:
            logger.warning("'[CLS]' token not found in generated tokens.")
            att_to_cls = None

        # Convert attention scores to exponential format
        attention_to_cls_exp = [math.exp(float(i)) for i in att_to_cls] if att_to_cls is not None else []

        # Create a DataFrame for attention scores related to the [CLS] token
        if att_to_cls is not None:
            token_att_num = list(zip(names, attention_to_cls_exp))
            df_att_to_cls_exp = pd.DataFrame(token_att_num, columns=['token', 'attention'])
            df_att_to_cls_exp = df_att_to_cls_exp[~df_att_to_cls_exp['token'].isin(['[CLS]', '[SEP]'])]
        else:
            df_att_to_cls_exp = pd.DataFrame(columns=['token', 'attention'])

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
    df_all_vs_all, att_to_cls, df_att_to_cls_exp = analyzer.attention_score_to_cls_token_and_to_all(input_text, analyzer.model, analyzer.tokenizer, analyzer.device)

    # Save the results
    df_all_vs_all.to_csv(f"decoder_attention_scores_{config['run_name']}.csv")
    df_att_to_cls_exp.to_csv(f"decoder_attention_scores_to_cls_exp_{config['run_name']}.csv")
    att_to_cls.to_csv(f"decoder_attention_scores_to_cls_{config['run_name']}.csv")




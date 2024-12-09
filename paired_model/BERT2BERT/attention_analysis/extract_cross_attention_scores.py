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

pd.options.display.precision = 6  

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
        
    def cross_attention_scores(self, input_text, model, tokenizer, device):
        ''' 
        Extract cross-attention scores for generated tokens with respect to source tokens.

        Outputs:
        - df_cross_att: pandas.DataFrame with attention scores of generated tokens (rows) to input tokens (columns)
        '''
        logger.info("Starting cross-attention score computation.")
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

        # Extract input tokens (source tokens)
        input_tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # Extract cross-attentions from the last layer
        cross_attentions = outputs.cross_attentions  # List of tensors, one per layer
        last_cross_attention = cross_attentions[-1]  # Shape: (batch_size, num_heads, tgt_seq_len, src_seq_len)

        multiplier = 10000
        last_cross_attention = last_cross_attention * multiplier

        # Aggregate over attention heads (e.g., average or sum)
        avg_cross_attention = last_cross_attention.mean(dim=1)  # Shape: (batch_size, tgt_seq_len, src_seq_len)

        # Convert attention to numpy
        att_matrix = avg_cross_attention[0].detach().cpu().numpy()  # Shape: (tgt_seq_len, src_seq_len)

        # Create a DataFrame for visualization or further processing
        df_cross_att = pd.DataFrame(att_matrix, index=generated_tokens, columns=input_tokens)

        #Each target token has an attention distribution over all source tokens. Aggregate this distribution into a single value for each target token using the mean or max
        aggregated_scores = att_matrix.max(axis=1)  # Shape: (tgt_seq_len,)

        torch.cuda.empty_cache()
        gc.collect()

        return df_cross_att, aggregated_scores


if __name__ == "__main__":
    # Define configuration parameters
    config = {
        # best performing model so far
        "run_name": "PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1",
        "model_path": "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1",
        "tokenizer_path": "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1/checkpoint-95615",
        "adapter_path": "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1/final_adapter",
        "generation_config_path": "/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1",
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

    input_text = "E V Q L V E S G G D L V R P G G S L R L S C A A S G F P F S R A W M T W V R Q A P G K G L D W V A R I K S K A A D G S A D Y A A A V V G R F V I S R D D A T G T V Y L Q M N S L R S E D T A M Y H C A T D I G L T L V P A T G Y W G Q G V L V T V S S"
    df_cross_att, aggregated_scores = analyzer.cross_attention_scores(input_text, analyzer.model, analyzer.tokenizer, analyzer.device)

    # Save the results
    #df_cross_att.to_csv(f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/attention_analysis/attention_score_outputs/cross_attention_scores_{config['run_name']}.csv", float_format='%.6f')
    np.savetxt(f"/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/attention_analysis/attention_score_outputs/aggregated_cross_attention_max_scores_{config['run_name']}.txt", aggregated_scores, fmt='%.6f')

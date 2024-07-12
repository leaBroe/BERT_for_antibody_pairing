# Environments Usage Guide

This guide describes the various environments available and the specific use-cases for each. The necessity for different environments stems from compatibility issues among packages when employed in specific tasks, such as classification or Next Sentence Prediction (NSP). Moreover, each adapter’s version mandates a corresponding version of transformers. Consequently, to ensure optimal functionality and avoid conflicts, a separate environment is dedicated exclusively to adapters.

## Environment Descriptions

### `adapter_env`
**Usage:** Utilize this environment for running scripts that requires adapters. 

### `classification_env`
**Usage:** This environment is configured for classification tasks (using the ProtBERT bfd model). 

### `mlm_env`
**Usage:** Employ this environment for training the RoBERTa Masked Language Model (MLM) from scratch.  

### `nsp_env`
**Usage:** Environment or Next Sentence Prediction (NSP) and MLM fine-tuning with the ProtBERT bfd model. 

### `OAS_paired_env`
**Usage:** This environment was used for data preprocessing and clustering tasks. 

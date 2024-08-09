# Heavy Model Training for Antibody Pairing using RoBERTa and MLM

This directory contains scripts for training a heavy chain model from scratch, using the RoBERTa architecture and the Masked Language Modeling (MLM) approach. This model later serves as the encoder in the heavy2light model, which is part of our antibody pairing project.

## Overview

### RoBERTa Model
RoBERTa (A Robustly Optimized BERT Pretraining Approach) is a transformer-based model that builds upon BERT by optimizing its training process. It leverages a larger batch size, more data, and dynamic masking to improve performance on NLP tasks. In our work, we adapted RoBERTa to handle heavy chain sequences from the OAS database.

### Masked Language Modeling (MLM)
MLM is a pretraining technique where a percentage of input tokens are masked at random, and the model learns to predict these masked tokens. This approach allows the model to learn contextual representations of the heavy chain sequences, crucial for downstream tasks like antibody pairing.

### Encoder-Decoder Model
An Encoder-Decoder model is a type of neural network architecture where the encoder processes the input sequence and encodes it into a fixed-size context vector. The decoder then generates the output sequence based on this context vector. In our work, the trained heavy model (encoder) and light model (decoder) are used to predict light chain sequences given heavy chain sequences.

## Database

The heavy model was trained using sequences from the Observed Antibody Space (OAS) database. The OAS database aggregates Ig-seq outputs from over 80 studies, encompassing more than 2 billion unpaired and 2 million paired antibody sequences. For training purposes, we selected heavy chain sequences from unsorted B cells, plasma B cells, and memory B cells, focusing on data from healthy individuals without disease or vaccination history to ensure a diverse and unbiased dataset for model training.

## Data Clustering

To enhance the generalizability of our model, we implemented a clustering strategy as described by [Olsen et al.](https://academic.oup.com/bioinformaticsadvances/article/2/1/vbac046/6609807?login=true) The complementarity-determining region 3 (CDRH3) sequences were first clustered at 100% identity, and centroids were extracted. These centroids were then clustered at 70% identity, and the resulting sequences were used for model training. Further clustering at 50% identity determined the allocation of sequences into training, validation, and test datasets. This method minimizes redundancy and ensures that highly similar sequences are not split between datasets, reducing the risk of overfitting.

## Training Process

The heavy model was trained from scratch using the `run_mlm.py` script from the Hugging Face Transformers library. 

## Usage

**Training the Model:**  
   Use the provided scripts to initiate the training process with the pre-configured RoBERTa model and MLM strategy.

## Repository Structure

- **src:** Includes the `run_mlm.py` script and other utilities for model training and evaluation.
- **mmseqs:** Script for mmseqs2 for data clustering

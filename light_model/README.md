# Light Model Training for Antibody Pairing using RoBERTa and MLM

This directory contains scripts and resources for training a light chain model from scratch, using the RoBERTa architecture and the Masked Language Modeling (MLM) approach, similar to the heavy model. The light model later serves as the decoder in the heavy2light model, which is the main part of our antibody pairing project.

## Database

The light model was trained using unpaired light sequences from the Observed Antibody Space (OAS) database. Similarly to the heavy model, we selected light chain sequences from unsorted B cells, plasma B cells, and memory B cells, focusing on data from healthy individuals without a history of disease or vaccination.

## Data Clustering

The data was clustered the same ways as for the [heavy model](https://github.com/leaBroe/BERT_for_antibody_pairing/tree/master/heavy_model).

## Training Process

The light model was also trained from scratch using the `run_mlm.py` script from the Hugging Face Transformers library. 

## Repository Structure

- **[src/redo_ch](https://github.com/leaBroe/BERT_for_antibody_pairing/tree/master/light_model/src/redo_ch):** Includes the `run_mlm.py` script and other utilities for model training and evaluation.

# Scripts to run models using ProtBERT bfd

This folder uses the pretrained protein model ProtBERT bfd from [Rostlab](https://huggingface.co/Rostlab/prot_bert_bfd) firstly introduced in the paper [ProtTrans: Towards Cracking the Language of Life’s Code Through Self-Supervised Learning](https://www.biorxiv.org/content/10.1101/2020.07.12.199554v3).  

## Explanation of the files

[train_mlm_nsp_prot_bert.py](https://github.com/leaBroe/BERT_for_antibody_pairing/blob/master/paired_model/protBERT/train_mlm_nsp_prot_bert.py): Main file to perform NSP and MLM finetuning using a custom training loop.  

[nsp_mlm_protbert_with_adapters.py](https://github.com/leaBroe/BERT_for_antibody_pairing/blob/master/paired_model/protBERT/nsp_mlm_protbert_with_adapters.py): NSP and MLM finetuning using adapters, not finished / working yet.  

[preprocess_input_data.py](https://github.com/leaBroe/BERT_for_antibody_pairing/blob/master/paired_model/protBERT/preprocess_input_data.py): Since the tokenizer of ProtBERT bfd (and also IgBERT) expects a single space between each amino acid, this script preprocesses the input data by adding a space between each letter.  

[sweeps](https://github.com/leaBroe/BERT_for_antibody_pairing/tree/master/paired_model/protBERT/sweeps):  The scripts in this folder are used for hyperparameter tuning using sweeps from [Weights & Biases](https://wandb.ai/home). For more information regarding sweeps, see the [notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb#scrollTo=LWuspUMlnG2p) or the [walkthrough](https://docs.wandb.ai/guides/sweeps/walkthrough).  




# Scripts to run models using IgBERT

This folder uses the pretrained antibody model IgBERT from [Exscientia](https://huggingface.co/Exscientia/IgBert) firstly introduced in the paper [Large scale paired antibody language models](https://arxiv.org/abs/2403.17889).  

## Explanation of the files

[sweeps](https://github.com/leaBroe/BERT_for_antibody_pairing/tree/master/paired_model/IgBERT/sweeps): The scripts in this folder are used for hyperparameter tuning using sweeps from [Weights & Biases](https://wandb.ai/home). For more information regarding sweeps, see the [notebook](https://colab.research.google.com/github/wandb/examples/blob/master/colabs/pytorch/Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb#scrollTo=LWuspUMlnG2p) or the [walkthrough](https://docs.wandb.ai/guides/sweeps/walkthrough).  


[text_classification_transformers](https://github.com/leaBroe/BERT_for_antibody_pairing/tree/master/paired_model/IgBERT/text_classification_transformers): Text classification script from transformers, but in our case not directly applicable since it concatenates both heavy and light sequence to a single sequence for classification (not suitable for multi column classification tasks).  

[tutorial](https://github.com/leaBroe/BERT_for_antibody_pairing/tree/master/paired_model/IgBERT/tutorial): Tutorial for text classification using IMDB data. 

[create_and_synthesize_dataset_for_classification.py](https://github.com/leaBroe/BERT_for_antibody_pairing/blob/master/paired_model/IgBERT/create_and_synthesize_dataset_for_classification.py): For the classification task to work, we need paired and unpaired (synthetic) data. This script takes a dataset with sequences of the format heavy[SEP]light as input, splits the data on the [SEP] token, adds a 1 as label (paired) and generates random pairings of heavy and light chains and adds a 0 for unpaired. In this way, the sequences are not shared between the training, validation and test files (run the script for each training, test and val file separately). 

[light_heavy_classification.py](https://github.com/leaBroe/BERT_for_antibody_pairing/blob/master/paired_model/IgBERT/light_heavy_classification.py): Main file that performes the classifciation using BertForSequenceClassification and adapters.  
the input data is a csv file with columns 'heavy', 'light', 'label' and single space separated AAs (for the tokenizer of IgBERT)  
1 for paired, 0 for not paired  
example:  
heavy,light,label  
E V Q L V E S G G G L V Q P G G S L R L S C A A S G F T F S S Y D M H W V R Q A T G K G L E W V S A I G T A G D T Y Y P G S G K G R F T I S R E N A K N S L Y L Q M N S L R A G D T A V Y Y C A R A R P V G Y C S G G L G C G A F D I W G Q G T M V T V S S , S Y E L T Q P P S V S V S P G Q T A R I T C S G D A L P K Q Y A Y W Y Q H K P G Q A P V L V I Y K D S E R P S G I P E R F S G S S S G T T V T L T I S G V Q A E D E A D Y Y C Q S A D S S G T Y V V F G G G T K L T V L ,1  
Q V Q L Q E S G P G L V K P S E T L S L T C A V S G Y S I S S G Y Y W G W I R Q P P G K G L E W I G S I Y H S G S T Y Y N P S L K S R V T I S V D T S K N Q F S L K L S S V T A A D T A V Y Y C A R Y C G G D C Y Y V P D Y W G Q G T L V T V S S , S Y E L T Q P P S V S V S P G Q T A S I T C S G D K L G D K Y A C W Y Q Q K P G Q S P V L V I Y Q D S K R P S G I P E R F S G S N S G N T A T L T I S G T Q A M D E A D Y Y C Q A W D S S T E V V F G G G T K L T V L ,0  

[paired_full_seqs_sep_train_with_unpaired_small_space_separated_rm.csv](https://github.com/leaBroe/BERT_for_antibody_pairing/blob/master/paired_model/IgBERT/paired_full_seqs_sep_train_with_unpaired_small_space_separated_rm.csv) and [paired_full_seqs_sep_val_with_unpaired_small_space_separated_rm.csv](https://github.com/leaBroe/BERT_for_antibody_pairing/blob/master/paired_model/IgBERT/paired_full_seqs_sep_val_with_unpaired_small_space_separated_rm.csv): Small example data for training and validation sets.  


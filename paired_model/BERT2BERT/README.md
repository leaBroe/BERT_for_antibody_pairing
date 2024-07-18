# EncoderDecoder Models using either IgBERT or own Heavy/Light Models

This folder contains scripts to train an EncoderDecoder model with or without adapters using either IgBERT from [Exscientia](https://huggingface.co/Exscientia/IgBert) or our own models: The [heavy model](https://github.com/leaBroe/BERT_for_antibody_pairing/tree/master/heavy_model) as encoder and the [light model](https://github.com/leaBroe/BERT_for_antibody_pairing/tree/master/light_model) as decoder.  

[data](https://github.com/leaBroe/BERT_for_antibody_pairing/tree/master/paired_model/BERT2BERT/data): Example input data of the form "heavy sequence [SEP] light sequence"

[tutorial](https://github.com/leaBroe/BERT_for_antibody_pairing/tree/master/paired_model/BERT2BERT/tutorial): Tutorial for a bert2bert model from [here](https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Leveraging_Pre_trained_Checkpoints_for_Encoder_Decoder_Models.ipynb#scrollTo=mIBy7uK3Od4B), first introduced in the paper [Leveraging Pre-trained Checkpoints for Sequence Generation Tasks](https://arxiv.org/abs/1907.12461).  


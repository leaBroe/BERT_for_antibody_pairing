# used environemnt: bert2bert_env
# Leveraging Pre-trained Language Model Checkpoints for Encoder-Decoder Models
# Paper: https://doi.org/10.48550/arXiv.1907.12461
# Tutorial: https://colab.research.google.com/github/patrickvonplaten/notebooks/blob/master/Leveraging_Pre_trained_Checkpoints_for_Encoder_Decoder_Models.ipynb#scrollTo=mIBy7uK3Od4B

# Sequence-to-sequence tasks are defined as a mapping from an input sequence  𝐗1:𝑛  to an output sequence  𝐘1:𝑚  of a-priori unknown output length 𝑚. 
# Hence, a sequence-to-sequence model should define the conditional probability distribution of the output sequence  𝐘1:𝑚  conditioned on the input sequence  𝐗1:𝑛 :
# P_{𝜃_{model}}(𝐘1:𝑚|𝐗1:𝑛). 
# Without loss of generality, an input word sequence of 𝑛 words is hereby represented by the vector sequnece  𝐗1: 𝑛 = 𝐱1,…,𝐱𝑛  and an output sequence of 𝑚 words as  𝐘1:𝑚=𝐲1,…,𝐲𝑚 .

from transformers import EncoderDecoderModel, BertTokenizerFast, Seq2SeqTrainer, Seq2SeqTrainingArguments
import datasets
import torch
import pdb

import pandas as pd
#from IPython.display import display, HTML
from datasets import ClassLabel

train_data = datasets.load_dataset("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/tutorial/cnn_dailymail_2", "3.0.0", split="train[:5%]")
train_data.info.description

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

# map article and summary len to dict as well as if sample is longer than 512 tokens
def map_to_length(x):
  x["article_len"] = len(tokenizer(x["article"]).input_ids)
  x["article_longer_512"] = int(x["article_len"] > 512)
  x["summary_len"] = len(tokenizer(x["highlights"]).input_ids)
  x["summary_longer_64"] = int(x["summary_len"] > 64)
  x["summary_longer_128"] = int(x["summary_len"] > 128)
  return x



sample_size = 10000
data_stats = train_data.select(range(sample_size)).map(map_to_length, num_proc=4)

def compute_and_print_stats(x):
  if len(x["article_len"]) == sample_size:
    print(
        "Article Mean: {}, %-Articles > 512:{}, Summary Mean:{}, %-Summary > 64:{}, %-Summary > 128:{}".format(
            sum(x["article_len"]) / sample_size,
            sum(x["article_longer_512"]) / sample_size, 
            sum(x["summary_len"]) / sample_size,
            sum(x["summary_longer_64"]) / sample_size,
            sum(x["summary_longer_128"]) / sample_size,
        )
    )

output = data_stats.map(
  compute_and_print_stats, 
  batched=True,
  batch_size=-1,
)

encoder_max_length=512
decoder_max_length=128

def process_data_to_model_inputs(batch):
  # tokenize the inputs and labels
  inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=encoder_max_length)
  outputs = tokenizer(batch["highlights"], padding="max_length", truncation=True, max_length=decoder_max_length)

  batch["input_ids"] = inputs.input_ids
  batch["attention_mask"] = inputs.attention_mask
  batch["decoder_input_ids"] = outputs.input_ids
  batch["decoder_attention_mask"] = outputs.attention_mask
  batch["labels"] = outputs.input_ids.copy()

  # because BERT automatically shifts the labels, the labels correspond exactly to `decoder_input_ids`. 
  # We have to make sure that the PAD token is ignored
  batch["labels"] = [[-100 if token == tokenizer.pad_token_id else token for token in labels] for labels in batch["labels"]]

  return batch


train_data = train_data.select(range(32))

# batch_size = 16
batch_size=4

train_data = train_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["article", "highlights", "id"]
)

train_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

val_data = datasets.load_dataset("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/tutorial/cnn_dailymail_2", "3.0.0", split="validation[:1%]")

val_data = val_data.select(range(8))

val_data = val_data.map(
    process_data_to_model_inputs, 
    batched=True, 
    batch_size=batch_size, 
    remove_columns=["article", "highlights", "id"]
)

val_data.set_format(
    type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
)

# warm-start both the encoder and decoder with the "bert-base-cased" checkpoint.
bert2bert = EncoderDecoderModel.from_encoder_decoder_pretrained("bert-base-uncased", "bert-base-uncased")

print(bert2bert)
print(bert2bert.config)

#bert2bert.save_pretrained("bert2bert")
#bert2bert = EncoderDecoderModel.from_pretrained("bert2bert")

bert2bert.config.decoder_start_token_id = tokenizer.cls_token_id
bert2bert.config.eos_token_id = tokenizer.sep_token_id
bert2bert.config.pad_token_id = tokenizer.pad_token_id
bert2bert.config.vocab_size = bert2bert.config.encoder.vocab_size

bert2bert.config.max_length = 142
bert2bert.config.min_length = 56
bert2bert.config.no_repeat_ngram_size = 3
bert2bert.config.early_stopping = True
bert2bert.config.length_penalty = 2.0
bert2bert.config.num_beams = 4


training_args = Seq2SeqTrainingArguments(
    predict_with_generate=True,
    evaluation_strategy="steps",
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    fp16=True, 
    output_dir="./",
    logging_steps=2,
    save_steps=10,
    eval_steps=4,
    # logging_steps=1000,
    # save_steps=500,
    # eval_steps=7500,
    # warmup_steps=2000,
    # save_total_limit=3,
)



rouge = datasets.load_metric("rouge")



def compute_metrics(pred):
    labels_ids = pred.label_ids
    pred_ids = pred.predictions

    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    labels_ids[labels_ids == -100] = tokenizer.pad_token_id
    label_str = tokenizer.batch_decode(labels_ids, skip_special_tokens=True)

    rouge_output = rouge.compute(predictions=pred_str, references=label_str, rouge_types=["rouge2"])["rouge2"].mid

    return {
        "rouge2_precision": round(rouge_output.precision, 4),
        "rouge2_recall": round(rouge_output.recall, 4),
        "rouge2_fmeasure": round(rouge_output.fmeasure, 4),
    }



# instantiate trainer
trainer = Seq2SeqTrainer(
    model=bert2bert,
    tokenizer=tokenizer,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_data,
    eval_dataset=val_data,
)
trainer.train()


test_data = datasets.load_dataset("/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/tutorial/cnn_dailymail_2", "3.0.0", split="test[:1%]")


def generate_summary(batch):
    # cut off at BERT max length 512
    inputs = tokenizer(batch["article"], padding="max_length", truncation=True, max_length=512, return_tensors="pt")
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")

    outputs = bert2bert.generate(input_ids, attention_mask=attention_mask)

    output_str = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    batch["pred_summary"] = output_str

    return batch


batch_size = 16  # change to 64 for full evaluation

results = test_data.map(generate_summary, batched=True, batch_size=batch_size, remove_columns=["article"])

rouge.compute(predictions=results["pred_summary"], references=results["highlights"], rouge_types=["rouge2"])["rouge2"].mid




























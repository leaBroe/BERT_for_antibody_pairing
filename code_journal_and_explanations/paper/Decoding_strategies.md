# Generation Strategies

Generation Strategies on hugging face

[Text generation strategies](https://huggingface.co/docs/transformers/en/generation_strategies)

arguments:

[Generation](https://huggingface.co/docs/transformers/v4.46.2/en/main_classes/text_generation#transformers.GenerationConfig)

nucleus sampling and others:

[Decoding Strategies that You Need to Know for Response Generation](https://towardsdatascience.com/decoding-strategies-that-you-need-to-know-for-response-generation-ba95ee0faadc)

Full test set evaluation

run name:

PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1

```bash
> print(mean_blosum)
[1] 344.8011
> print(mean_similarity)
[1] 66.56336
> print(median_blosum)
[1] 356
> print(median_similarity)
[1] 65.76577
```

```bash
{
  "decoder_start_token_id": 2,
  "diversity_penalty": 1.0,
  "early_stopping": true,
  "eos_token_id": 3,
  "max_length": 512,
  "min_length": 50,
  "num_beam_groups": 5,
  "num_beams": 5,
  "output_hidden_states": true,
  "output_logits": true,
  "output_scores": true,
  "pad_token_id": 0,
  "return_dict_in_generate": true,
  "transformers_version": "4.40.2",
  "vocab_size": 25
}

```

# DoLa Decoding

[DoLa Explanation](https://www.notion.so/DoLa-Explanation-13e4a8ac4e208085ab25ea1033ce46ae?pvs=21)

# 18/11/2024

DoLa decoding model running:

```bash
paired_model/BERT2BERT/decoding_strategies/DoLa/logs/full_PLAbDab_healthy_human_[1,340]_DoLa_max_length_120_rep_penalty_1.2_num_epochs_30.o
```

Nucleus sampling model running:

```bash
paired_model/BERT2BERT/decoding_strategies/nucleus_sampling/logs/nucleus_0.9_temp_0.1_full_PLAbDab_healthy_human_max_length_120_num_epochs_30.txt
```

## Contrastive Search

[Generating Human-level Text with Contrastive Search in Transformers 🤗](https://huggingface.co/blog/introducing-csearch)

So far penalty_alpha=0.6 and top_k=4 used

```bash
{
  "decoder_start_token_id": 2,
  "eos_token_id": 3,
  "max_length": 512,
  "min_length": 50,
  "output_hidden_states": true,
  "output_logits": true,
  "output_scores": true,
  "pad_token_id": 0,
  "penalty_alpha": 0.6,
  "return_dict_in_generate": true,
  "top_k": 4,
  "transformers_version": "4.40.2",
  "vocab_size": 25
}
```

Contrastive search model with top_k = 2 and penalty_alpha = 0.8 is running

# 25/11/2024

## **Conditional Masked Language Decoding**

[Mask-Predict: Parallel Decoding of Conditional Masked Language Models](https://ar5iv.labs.arxiv.org/html/1904.09324)

https://github.com/facebookresearch/Mask-Predict/blob/main/train.py

```bash
python generate_cmlm.py ${output_dir}/data-bin --path ${model_dir}/checkpoint_best_average.pt --task translation_self --remove-bpe --max-sentences 20 --decoding-iterations 10 --decoding-strategy mask_predict
```


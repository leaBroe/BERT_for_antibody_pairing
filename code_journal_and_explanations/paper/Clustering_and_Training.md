# Clustering

## 01/11/2024

### Clustering approach:

**Paired sequences**

Only human with Disease: None and Vaccination: None

Cluster full paired sequence at 70% identity?

do same clustering approach also with original dataset (all human, besides everything *)

human, no vaccination, no disease, everything else: *

![Screenshot 2024-11-01 at 15.48.53.png](https://prod-files-secure.s3.us-west-2.amazonaws.com/882c1b95-476d-4a7f-9994-d02922cb2063/702c95d0-49d1-4d63-a7aa-3ab62508f84a/Screenshot_2024-11-01_at_15.48.53.png)

568’139 sequences in extracted data (path:)

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/extracted_subset_full.csv
```

```bash
awk -F ',' '!seen[$NF]++' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/extracted_subset_relevant_columns.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/extracted_subset_relevant_columns_no_dupl.csv
```

number of sequences before clustering but after duplicate removal:

509’520 seqs

create fasta file:

```bash
awk -F, '{print ">" $1 "\n" $10}' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/extracted_subset_relevant_columns_no_dupl.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/extracted_subset_relevant_columns_no_dupl.fasta
```

503'462 seqs (99 pident)

# 04/11/2024

## Results clustering (wc -l on idseq file)

number of sequences before clustering but after duplicate removal (on full paired seq): **509’520** seqs 

| Pident | # Seqs |
| --- | --- |
| 30 | 35’664 |
| 40 | 42’815 |
| 50 | 65’193 |
| 60 | 92’335 |
| 80 | 357’381 |
| 90 | 425’253 |
| 95 | 503’362 |
| 99 | 503’463 |

## Final clustering approach:

Remove the duplicates from the full paired sequence, cluster them at 30 pident and use this for the allocation but keep all sequences without duplicates (509’520 seqs).

## Allocation to train, test and validation sets

```bash
/ibmm_data2/oas_database/paired_lea_tmp/light_model/src/train_test_val_split.py
```

```bash
python /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/train_test_val_split.py --tsv_dataset /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/linclust/30-60_pident/extracted_subset_relevant_columns_no_dupl_30_clu.tsv --rep_fasta_file /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/linclust/80-99_pident/extracted_subset_relevant_columns_no_dupl.fasta --prefix human_healthy_no_vac_allocated_
```

train, test and validation datasets in path:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/train_test_val_datasets
```

```bash
awk -F, '{print ">" $1 "\n" $11}' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/extracted_subset_relevant_columns_no_dupl.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/extracted_subset_relevant_columns_no_dupl_sep.fasta
```

```bash
python /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/train_test_val_split.py --tsv_dataset /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/linclust/30-60_pident/extracted_subset_relevant_columns_no_dupl_30_clu.tsv --rep_fasta_file /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/extracted_subset_relevant_columns_no_dupl_sep.fasta --prefix human_healthy_no_vac_allocated_sep
```

# PLAbDab Sequences

## 06/11/2014

```bash
awk -F, '!seen[$NF]++' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/plabdab_paired_sequences_full_seqs_no_dupl_with_human_healthy_no_vac_from_oas.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/plabdab_paired_sequences_full_seqs_no_dupl_with_human_healthy_no_vac_from_oas_check.csv
```

```bash
awk -F, '{print ">" $1 "\n" $NF}' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/plabdab_paired_sequences_full_seqs_no_dupl_with_human_healthy_no_vac_from_oas.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/plabdab_paired_sequences_full_seqs_no_dupl_with_human_healthy_no_vac_from_oas.fasta
```

```bash
CUDA_VISIBLE_DEVICES=7 ./mmseqs_linclust.sh plabdab_paired_sequences_full_seqs_no_dupl_with_human_healthy_no_vac_from_oas
```

Dataset with human healthy paired seqs (no disease, no vac) + PLAbDab unique seqs 

Did 30 pident clustering for allocation to train, test and val datasets:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/linclust/plabdab_paired_sequences_full_seqs_no_dupl_with_human_healthy_no_vac_from_oas_30_clu.tsv
```

```bash
python /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/train_test_val_split.py --tsv_dataset /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/linclust/conc_oas_healthy_human_plabdab_sep_seqs_only_30_clu.tsv --rep_fasta_file /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/linclust/conc_oas_healthy_human_plabdab_sep_seqs_only.fasta --prefix plabdab_human_healthy_no_vac_allocated
```

509’520

```bash
cat /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/extracted_subset_relevant_columns_no_dupl.csv /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/plabdab_paired_sequences_full_seqs_no_dupl.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/plabdab_paired_sequences_full_seqs_no_dupl_with_human_healthy_no_vac_from_oas_2.csv
```

extract the last column of csv file 

```bash
awk -F, '{print $NF}' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/plabdab_paired_sequences_full_seqs_no_dupl.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/plabdab_paired_sequences_full_seqs_only_no_dupl.csv
```

```bash
cat /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/extracted_subset_relevant_columns_no_dupl_sep_no_identifiers.fasta /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/plabdab_paired_sequences_full_seqs_only_no_dupl.csv > conc_oas_healthy_human_plabdab_sep_seqs_only.txt
```

remove duplicates

```bash
sort /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/conc_oas_healthy_human_plabdab_sep_seqs_only.txt | uniq > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/conc_oas_healthy_human_plabdab_sep_seqs_only_no_dupl.txt
```

final concatenated dataset with OAS and plabdab sequences (healthy human, no vac):

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/conc_oas_healthy_human_plabdab_sep_seqs_only.txt
```

# 11/11/2024

So far:

New dataset 1: Only human healthy, no disease, no vac → 509’520 seqs (no dupl)

New dataset 2: Only human healthy, no disease, no vac + PLAbDab paired seqs → 588’389 seqs (no dupl)

→ models with both datasets trained, evaluation ongoing

New dataset 3: Human healthy AND disease, 30 pident clustering but keeping all seqs after duplicate removal (same as prevous dataset but without the clustering) + PLAbDab paired seqs (this would be the biggest dataset available) → 1’075’104 seqs (no dupl)

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/data_oas/paired_rowid_cdrh3_full_sep.csv
```

```bash
awk -F, '{print $1 "," $NF}' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/PLAbDab_db/plabdab_paired_sequences_full_seqs_no_dupl.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_all_disease_plabdab/plabdab_id_full_sep.csv
```

```bash
awk -F, '!seen[$NF]++' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_all_disease_plabdab/conc_human_all_paired_plabdab.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_all_disease_plabdab/conc_human_all_paired_plabdab_no_dupl.csv
```

final dataset:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_all_disease_plabdab/conc_human_all_paired_plabdab_no_dupl.csv
```

```bash
awk -F, '{print ">" $1 "\n" $NF}' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_all_disease_plabdab/conc_human_all_paired_plabdab_no_dupl.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_all_disease_plabdab/conc_human_all_paired_plabdab_no_dupl.fasta
```

```bash
python /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/train_test_val_split.py --tsv_dataset /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_all_disease_plabdab/linclust/conc_human_all_paired_plabdab_no_dupl_30_clu.tsv --rep_fasta_file /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_all_disease_plabdab/linclust/conc_human_all_paired_plabdab_no_dupl.fasta --prefix human_healthy_all_diseases_plabdab
```

New dataset 4:  human healthy, SARS-COV2 + PLAbDab paired seqs → 1’292’245 seqs

```bash
# SBATCH --gres=gpu:h100:1
# SBATCH --job-name=extr_subset
# SBATCH --output=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/logs/extr_subset_%j.o
# SBATCH --error=/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_no_vac/logs/extr_subset_%j.e

```

```bash
awk -F, '{print $1 "," $NF}' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_and_covid/extracted_data_from_oas/extracted_subset_healthy_covid_relevant_cols.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_and_covid/extracted_data_from_oas/extracted_subset_healthy_covid_full_sep.csv
```

```bash
awk -F, '!seen[$NF]++' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_and_covid/healthy_covid_and_plabdab_concat.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_and_covid/healthy_covid_and_plabdab_concat_no_dupl.csv
```

```bash
awk -F, '{print ">" $1 "\n" $NF}' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_and_covid/healthy_covid_and_plabdab_concat_no_dupl.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_and_covid/healthy_covid_and_plabdab_concat_no_dupl.fasta
```

# 12/11/2024

New dataset 4

```bash
python /ibmm_data2/oas_database/paired_lea_tmp/light_model/src/train_test_val_split.py --tsv_dataset /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_and_covid/linclust/healthy_covid_and_plabdab_concat_no_dupl_30_clu.tsv --rep_fasta_file /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/new_data/human_healthy_and_covid/linclust/healthy_covid_and_plabdab_concat_no_dupl.fasta --prefix human_healthy_covid_allocated_
```

Dataset 4 is ready for training 

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

run name:

PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1

```bash
> print(mean_blosum)
[1] 345.3567
> print(mean_similarity)
[1] 66.51719
> print(median_blosum)
[1] 355
> print(median_similarity)
[1] 65.74074
```

run name: PLAbDab_all_human_paired_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.0001_wd_0.1

```bash
> print(mean_blosum)
[1] 331.9578
> print(mean_similarity)
[1] 64.92727
> print(median_blosum)
[1] 332
> print(median_similarity)
[1] 62.83186
```

# 20/11/2024

Error in full evaluation script for dataset 4 (only this dataset was affected!) → corrected it & is now running

full evaluation with run name: 

**DoLa_layers_high_rep_penal_1.2_full_PLAbDab_healthy_human_max_length_120_num_epochs_50**

is now in queue (184931)

# 21/11/2024-25/11/2024

Now running on H100 (184932)

full evaluation run name:

**nucleus_0.9_temp_0.1_full_PLAbDab_healthy_human_max_length_120_num_epochs_30**

```bash
Average BLOSUM Score: 351.5054810584816
Average Similarity Percentage: 66.37958238123686%
Average Perplexity: 3.0216522203708753
```

run name: PLAbDab_healthy_human_covid_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.0001_wd_0.1

```bash
Average BLOSUM Score: 348.2300114529809
Average Similarity Percentage: 65.78935186902771%
Average Perplexity: 2.702276081367077
```

run name: DoLa_layers_high_rep_penal_1.2_full_PLAbDab_healthy_human_max_length_120_num_epochs_50

```bash
Average BLOSUM Score: 350.74000237937423
Average Similarity Percentage: 66.20492208679194%
Average Perplexity: 2.763099335019939
```

All new decoding strategies either fully evaluated or running

| run name | Dataset | Decoding | BLOSUM Mean | Similarity Mean [%] | Perplexity mean | path to model |
| --- | --- | --- | --- | --- | --- | --- |
| PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1 |  Only human healthy, no disease, no vac + PLAbDab paired seqs  | Diverse Beam Search  "num_beam_groups": 5,
"num_beams": 5,   "diversity_penalty": 1.0,
 | 344.8011 | **66.56336** | 2.8187 | /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1 |
| PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1 |  Only human healthy, no disease, no vac + PLAbDab paired seqs  | Diverse Beam Search  "num_beam_groups": 5,
"num_beams": 5,   "diversity_penalty": 1.0, | 345.3567 | **66.51719** | 2.14915 | /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.001_wd_0.1 |
| PLAbDab_all_human_paired_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.0001_wd_0.1 | Human healthy AND disease + PLAbDab paired seqs  | Diverse Beam Search  "num_beam_groups": 5,
"num_beams": 5,   "diversity_penalty": 1.0, | 331.9578 | **64.92727** | 2.78301 | /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/PLAbDab_all_human_paired_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.0001_wd_0.1 |
| **nucleus_0.9_temp_0.1_full_PLAbDab_healthy_human_max_length_120_num_epochs_30** |  Only human healthy, no disease, no vac + PLAbDab paired seqs  | Nucleus top_p=0.9, temp=0.1 | 351.506 | **66.3796** | 3.02165 | /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/nucleus_0.9_temp_0.1_full_PLAbDab_healthy_human_max_length_120_num_epochs_30 |
| PLAbDab_healthy_human_covid_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.0001_wd_0.1 | human healthy, SARS-COV2 + PLAbDab paired seqs | Diverse Beam Search  "num_beam_groups": 5,
"num_beams": 5,   "diversity_penalty": 1.0, | 348.2300 | **65.78935** | 2.7022 | /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/PLAbDab_healthy_human_covid_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_60_lr_0.0001_wd_0.1 |
| DoLa_layers_high_rep_penal_1.2_full_PLAbDab_healthy_human_max_length_120_num_epochs_50 |  Only human healthy, no disease, no vac + PLAbDab paired seqs  | DoLa, “high”, repetition penalty=1.2 | 350.7400 | **66.2049** | 2.76309 | /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/DoLa_layers_high_rep_penal_1.2_full_PLAbDab_healthy_human_max_length_120_num_epochs_50 |
| contrastive_k_2_pen_0.8_temp_0.1_full_PLAbDab_healthy_human_max_length_120_num_epochs_50 |  Only human healthy, no disease, no vac + PLAbDab paired seqs | Contrastive Search, k=2, repetition penalty = 0.8 |  |  |  | /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/heavy2light_model_checkpoints/contrastive_k_2_pen_0.8_temp_0.1_full_PLAbDab_healthy_human_max_length_120_num_epochs_50 |
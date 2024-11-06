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
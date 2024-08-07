# PCA / t-SNE / UMAP Species and Diseases
## Data preprocessing:

1. Use 

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/src/extract_columns_from_sqlite_db.sh 
```

to extract the relevant columns from OAS db

resulting output file:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_subgroups_analysis/paired_oas_db_full_extraction.csv
```

1. use

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/match_sequences.jl
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/run_match_sequences.sh
```

to extract the test set sequences from the full paired db

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_subgroups_analysis/full_test_data_extraction_species_diseases.txt
lines: 74218
should be: 67209
```

remove duplicates from heavy[SEP]light column (last column → $NF)

```python
awk -F ',' '!seen[$NF]++' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_subgroups_analysis/full_test_data_extraction_species_diseases.txt > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_subgroups_analysis/full_test_data_extraction_species_diseases_no_dupl.txt
```

output file with 67209 sequences:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_subgroups_analysis/full_test_data_extraction_species_diseases_no_dupl.txt
```

```python
header (manually added):
Species, Disease, BType, Isotype_light, sequence_alignment_aa_light, sequence_alignment_light, sequence_alignment_aa_heavy, sequence_alignment_heavy, sequence_alignment_heavy_sep_light

```

use 

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/species_diseases_subgroups_analysis/add_spaces_between_AAs.py
```

to add spaces between the sequences

use plot_umap_pca_tSNE_diseases_species.py for plotting
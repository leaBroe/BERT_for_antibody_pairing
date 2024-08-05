### Data Processing steps for e.g. creating the U-MAP / PCA / t-SNE:

At the end we want to have a csv file of the form:

```python
sequence_alignment_heavy_sep_light,locus
Q V Q L Q E S G P G L V K P S E T L S L T C T V S G G S I S G F Y W S W I R Q S P G K G L E W I A Y I Y F S G S T N Y N P S L K S R V T L S V D T S K N Q F S L K L S S V T A A D S A V Y Y C A R D V G P Y N S I S P G R Y Y F D Y W G P G T L V T V S S [SEP] Q S A L T Q P A S V S G S P G Q S I T I S C T G T S S D V G N Y N L V S W Y Q H H P G K A P K L M I Y E V S K R P S G I S N R F S G S K S G N T A S L T I S G L Q A D D E A D Y Y C C S Y A G S R I L Y V F G S G T K V T V L ,IGL
```

Where IGL = Lambda light chain and IGK = Kappa light chain

We get the IGL/IGK label by using PyIR (column name: “locus”)

## Workflow for the full test dataset with 67’209 sequences:

We need a fasta file with the dna sequence for PyIR

Full extracted sequences from OAS.db:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/full_test_data/full_paired_data_for_analysis.csv
```

all relevant test data sequences from OAS.db with all columns:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/full_test_data/full_test_data_extraction_with_header.txt
```

Use the file:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/convert_to_fasta.jl
```

to convert the file from the OAS.db into a fasta file of the form:

```python
>1
CAGTCTGCCCTGACTCAGCCAGCCTCCGTGTCTGGGTCTCCTGGACAGTCGATCACCATCTCCTGCACTGGAACCAGCAGTGATGTTGGGAATTATAACCTTGTCTCCTGGTACCAACACCACCCAGGCAAAGCCCCCAAACTCATGATTTATGAGGTCAGTAAGCGGCCCTCAGGGATTTCTAATCGCTTCTCTGGCTCCAAGTCTGGCAACACGGCCTCCCTGACAATCTCTGGGCTCCAGGCTGACGACGAGGCTGATTATTACTGCTGCTCATATGCAGGTAGTAGAATCCTTTATGTCTTCGGATCTGGGACCAAGGTCACCGTCCTAG
>2
GAAATTGTGTTGACGCAGTCGCCAGGCACCCTGTCTTTGTCTACAGGGGAAAGAGCCACCCTCTCTTGCAGGGCCGGTCAGACTGTTGACGGCAACTCCTTAGCCTGGTACCAGCACAAACCTGGCCAGGCTCCCAGGCTCCTCATCTTTCGTGCATCTCGTAGGGCCGCTGACATCCCAGACAGGTTCACTGGCAGTGGGTCTGGGACCGACTTCACTCTCACCATTAGCAGACTGGAGGTTGAAGATTTCGCAGTTTATTACTGTCAGCAGTATGGTGCCTCACCAAAAACGTTCGGCCAAGGGACCAAGGTGGAA
>3
CAGTCTGCCCTGACTCAGCCTGCCTCCGTGTCTGGGTCTCCTGGACAGTCGATCACCATCTCCTGCACTGGAAGCAGCAGTGATGTTGGGAGTTATAACCTTGTCTCTTGGTACCAACAGCACCCAGGCAAAGCCCCCAAACTCATGATTTATGAGGTCAGTAAGCGGCCCTCAGGGGTTTCTAATCGCTTCTCTGGCTCCAAGTCTGGCAACACGGCCTCCCTGACAATCTCTGGGCTCCAGGCTGAGGACGAGGCTCAATATTACTGCTGCTCATATGGAGGTAGGAATTTTCATGTGCTATTCGGCGGAGGGACCGAGCTGACCGTCCTAG
>4
CAGTCTGCCCTGACTCAGCCTCCCTCCGCGTCCGGGTCTCTTGGACAGTCAGTCACCATCTCCTGCACTGGAAGTAGTAGTGACGTTGGTGGGTATGCCTATGTCTCCTGGTATCAACAACACCCAGGCAAAGCCCCCAAAGTCGTAATTTATGAGGTCACTAAGCGGCCCTCAGGGGTCCCTGAACGGTTCTCTGGCTCCAAGTCTGGCAACACGGCCTCCCTGACCGTCTCTGGGCTCCAGGCTGAAGATGAGGCTGATTATTACTGCATCTCATATGCCGGCGCCAACAAATTAGGGGTATTCGGCGGAGGGACCAAGCTGACCGTCCTAG
```

where the dna sequence is the dna sequence of the light chain AA sequence

full test data fasta file:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/full_test_data/full_paired_data_sequence_alignment_light.fasta
```

Then  use this fasta file to perform PyIR. For this, use the script:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/PyIR.py
```

67,209 sequences processed in 42.41 seconds, 1,584 sequences / s

```python
gunzip -d /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/full_test_data/full_paired_data_sequence_alignment_light.tsv.gz
```

Then, we need to extract the relevant columns for the u-map

```python
columns_to_extract = [
    "sequence_id",
    "sequence",
    "sequence_alignment_aa",
    "locus"
]
```

using the script:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/extract_relevant_cols_from_tsv.py
```

output file:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/extracted_data_from_PyIR/umap_extraction_from_pyir_full_data.csv
```

Then use the script:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_preprocessing_for_umap.py
```

to merge the csv file with the locus labels and the csv with the  heavy[SEP]light column, 

important: both csv files need to have the same column name: “sequence_alignment_aa” to merge them on.

Then remove the duplicates with:

```python
awk -F ',' '!seen[$1]++' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_for_umap_pca/full_data_heavyseplight_locus.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_for_umap_pca/full_data_heavyseplight_locus_no_dupl.csv
```

final output file with 67’209 sequences:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_for_umap_pca/full_data_heavyseplight_locus_no_dupl.csv
```

To include spaces between the AAs use the script:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/protBERT/preprocess_input_data.py
```

and to remove the spaces in the last column use:

```python
awk 'BEGIN { FS=OFS="," } { gsub(/ /, "", $NF); print }' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_for_umap_pca/full_data_heavyseplight_locus_no_dupl_spaces.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_for_umap_pca/full_data_heavyseplight_locus_no_dupl_spaces_rm.csv
```

So we end up with the final input file of the form:

```python
sequence_alignment_heavy_sep_light,locus
Q V Q L Q E S G P G L V R P S E T L S L E C S V S G S S L S N D Y Y W G W I R Q P P G K G L Q W I G N I Y H S G T T Y Y N P S L K S R L T M S V D T S R N H F S L Q L D S V T A A D T A V Y Y C A R L I Y T G Y G K R C F D Y W G Q G A L V T V S S [SEP] D I Q M T Q S P P F V S A S V G D S V T I T C R A S Q G I T D W L A W Y Q H K Q G K A P K L L I F A A S T L Q S G V P S R F S G T G S G T D F T L T I T R L Q P E D S A T Y Y C Q Q G Y T F P G G F T F G P G T K V D V K ,IGK
Q V Q L V E S G G G V V Q P G R S L R L S C A A S G F T F S S Y G M H W V R Q A P G K G L E W V G V I W Y D G S K K Y Y S D S V K G R F T I S R D S P N N M L Y L Q M N S L R A E D T A V Y F C A R D D D G S N Q Y G I F E Y W G Q G T V V T V S S [SEP] Q S A L T Q P V S V S G S P G Q S I A I S C T G T S S D V G G Y N S V S W F Q Q H P G K A P K L M I Y D V S N R P S G V S N R F S G S K S G N T A S L T I S G L Q A E D E A D Y Y C S S Y T S S S T R L F G G G T K L T V L ,IGL
```

location:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/data_for_umap_pca/full_data_heavyseplight_locus_no_dupl_spaces_rm.csv
```

We then can use the file 

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/src/plot_umap.py
```

for the U-map.

# B-Type analysis of generated sequences (similarity) / Comparison of Memory and Naive B Cells

extracted information about the B-Types:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/Btypes_full_paired_data_for_analysis.csv
```

Used the file:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/match_sequences.jl
```

output file:

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/Btypes_full_paired_test_data.csv
```

Remove duplicates:

```python
awk -F ',' '!seen[$3]++' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/B_cell_analysis_output.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/B_cell_analysis_output_no_dupl.csv
```

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/Btypes_full_paired_test_data_no_dupl.csv
```

model output heavy2light

```python
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/full_eval_heavy2light_with_adapters125463.o
```

```python
awk -F ',' '!seen[$2]++' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/B_cell_analysis_output.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/B-cell_analysis/B_cell_analysis_output_no_dupl.csv
```

### Results B-Cell analysis of heavy2light model on full test set MEAN:

```python
                                            Perplexity  BLOSUM Score  Similarity Percentage
BType                                                                                      
ASC                                           2.258298     59.026316              25.719750
CD27-memory-and-Plasmablast/Plasma-B-Cells    2.221776     92.713127              30.184347
Memory-B-Cells                                2.416487     93.632222              30.312491
Naive-B-Cells                                 1.857270    101.022147              31.820085
Plasma-B-Cells                                2.333874     94.121564              30.370104
Plasmablast                                   2.310689     88.673410              29.606787
Plasmablast/Plasma-B-Cells                    2.460713     70.237770              27.013723
RV+B-Cells                                    2.087540     48.058824              24.554099
Unsorted-B-Cells                              2.108372     94.692478              30.629889
double-nagative-B-cells                       2.300443     91.584906              30.065477
```

### Results Kappa Lambda analysis MEAN (full test set)

```python
       BLOSUM Score  Similarity Percentage  Perplexity
locus                                                 
IGH      -68.000000               8.488388   26.063198
IGK      167.590051              40.481777    2.042250
IGL        7.685513              18.843335    2.356489
```

### Results Kappa Lambda analysis MEDIAN (full test set)

```python
       BLOSUM Score  Similarity Percentage  Perplexity
locus                                                 
IGH           -70.0               8.035714   25.548422
IGK           110.0              31.775701    1.880061
IGL           -49.0              11.214953    2.166538
```

### Sequences in test set:

IGK / Kappa: 36’729 sequences

IGL / Lambda: 30’475 sequences

IGH: only 5 sequences → exclude them

```python
awk -F ',' '!seen[$1]++' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/kappa_lambda_analysis/umap_k_l_subtypes/full_data_heavyseplight_locus_v_family.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/kappa_lambda_analysis/umap_k_l_subtypes/full_data_heavyseplight_locus_v_family_no_dupl.csv
```

```python
awk 'BEGIN { FS=OFS="," } { gsub(/ /, "", $2); print }' /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/kappa_lambda_analysis/umap_k_l_subtypes/full_data_heavyseplight_locus_v_family_no_dupl_spaces_rm.csv > /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/kappa_lambda_analysis/umap_k_l_subtypes/full_data_heavyseplight_locus_v_family_no_dupl_spaces_rm2.csv
```
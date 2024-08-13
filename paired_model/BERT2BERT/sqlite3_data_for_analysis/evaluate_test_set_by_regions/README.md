# Light Chain Regions Analysis (see IgBERT paper for reference)

[IgBERT paper](https://arxiv.org/abs/2403.17889)

I’m using the model with run name: full_diverse_beam_search_2_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1

[run name = name on wandb](https://wandb.ai/ibmm-unibe-ch/heavy2light_translation?nw=nwuserlea_broe)

reference name: h2l_div_beam_search_2_epoch_10_lr_1e-4_wd_0.1 (for directory names)

because this has the best overall metrics (BLOSUM and Similarity %):

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/logs/full_eval_full_diverse_beam_search_2_temp_0.5_max_length_150_early_stopping_true_batch_size_64_epochs_10_lr_0.0001_wd_0.1_127798.o

# on full test set (67209 seqs):
Average BLOSUM Score: 108.25002603817941
Average Similarity Percentage: 33.05800137172086%
Mean Perplexity: 2.1675915718078613
```

I used the file

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions.py
```

to first convert the generated and true sequences form the output file of the full evaluation into a DNA fasta file of the form:

```bash
>True_Seq_1
CAATCTGCTTTAACTCAACCTGTTTCTGTTTCTGGTTCTCCTGGTCAATCTATTGCTATTTCTTGTACTGGTACTTCTTCTGATGTTGGTGGTTATAATTCTGTTTCTTGGTTTCAACAACATCCTGGTAAAGCTCCTAAATTAATGATTTATGATGTTTCTAATCGTCCTTCTGGTGTTTCTAATCGTTTTTCTGGTTCTAAATCTGGTAATACTGCTTCTTTAACTATTTCTGGTTTACAAGCTGAAGATGAAGCTGATTATTATTGTTCTTCTTATACTTCTTCTTCTACTCGTTTATTTGGTGGTGGTACTAAATTAACTGTTTTA
>Generated_Seq_1
GATATTCAAATGACTCAATCTCCTTCTTCTTTATCTGCTTCTGTTGGTGATCGTGTTACTATTACTTGTCGTGCTTCTCAATCTATTTCTTCTTATTTAAATTGGTATCAACAAAAACCTGGTAAAGCTCCTAAATTATTAATTTATGCTGCTTCTTCTTTACAATCTGGTGTTCCTTCTCGTTTTTCTGGTTCTGGTTCTGGTACTGATTTTACTTTAACTATTTCTTCTTTACAACCTGAAGATTTTGCTACTTATTATTGTCAACAATCTTATTCTACTCCTCCTACTTTTGGTCAAGGTACTAAAGTTGAAATTAAA
>True_Seq_2
GAAATTGTTTTAACTCAATCTCCTGGTACTTTATCTTTATCTCCTGGTGAAGCTGCTACTTTATCTTGTAAATCTTCTCAACCTGTTGCTACTACTTATTTAGCTTGGTATCAACAAAAACGTGGTCAACCTCCTCGTTTATTAATTTATGGTACTTCTAATCGTGCTGTTGGTATTCCTGATCGTTTTACTGGTTCTGGTTCTCGTACTGATTTTACTTTAACTATTGATCGTTTAGAAGCTGAAGATTTTGGTTTATATTTTTGTCAACAATATGCTACTTCTCCTTATACTTTTGGTCAAGGTACTAATTTAGAA
>Generated_Seq_2
GAAATTGTTTTAACTCAATCTCCTGCTACTTTATCTTTATCTCCTGGTGAACGTGCTACTTTATCTTGTCGTGCTTCTCAATCTGTTTCTTCTTATTTAGCTTGGTATCAACAAAAACCTGGTCAAGCTCCTCGTTTATTAATTTATGATGCTTCTAATCGTGCTACTGGTATTCCTGCTCGTTTTTCTGGTTCTGGTTCTGGTACTGATTTTACTTTAACTATTTCTTCTTTAGAACCTGAAGATTTTGCTGTTTATTATTGTCAACAACGTTCTAATTGGCCTCCTTTAACTTTTGGTGGTGGTACTAAAGTTGAAATTAAA
```

I checked the validity of the generated DNA sequences with the website: https://web.expasy.org/translate/

```bash
#true seq 1 with expasy translate tool (first line) and DNA sequence generated from script (2nd line):
QSALTQPVSVSGSPGQSIAISCTGTSSDVGGYNSVSWFQQHPGKAPKLMIYDVSNRPSGVSNRFSGSKSGNTASLTISGLQAEDEADYYCSSYTSSSTRLFGGGTKLTVL
QSALTQPVSVSGSPGQSIAISCTGTSSDVGGYNSVSWFQQHPGKAPKLMIYDVSNRPSGVSNRFSGSKSGNTASLTISGLQAEDEADYYCSSYTSSSTRLFGGGTKLTVL

#generated seq 1 with expasy translate tool (first line) and DNA sequence generated from script (2nd line):
DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPTFGQGTKVEIK
DIQMTQSPSSLSASVGDRVTITCRASQSISSYLNWYQQKPGKAPKLLIYAASSLQSGVPSRFSGSGSGTDFTLTISSLQPEDFATYYCQQSYSTPPTFGQGTKVEIK
```

Then, the same script can be used to perform PyIR on the fasta file, which gives you a tsv file with the full PyIR output.

```bash
134,418 sequences successfully split into 1325 pieces
Starting process pool using 64 processors
100%|███████████████████████████████████████████████████████████| 134418/134418 [00:52<00:00, 2539.55seq/s]
134,418 sequences processed in 53.39 seconds, 2,517 sequences / s
```

```bash
134418 / 2 = 67209 -> because each sequence is double (true seq and generated seq)
```

use gunizp -d to unzip the output tsv.gz file

the resulting tsv file has the columns:

```bash
sequence_id     sequence        locus   stop_codon      vj_in_frame     v_frameshift    productive      rev_comp   complete_vdj    v_call  d_call  j_call  sequence_alignment      germline_alignment      sequence_alignment_aa      germline_alignment_aa   v_alignment_start       v_alignment_end d_alignment_start       d_alignment_end    j_alignment_start       j_alignment_end v_sequence_alignment    v_sequence_alignment_aa v_germline_alignment       v_germline_alignment_aa d_sequence_alignment    d_sequence_alignment_aa d_germline_alignment       d_germline_alignment_aa j_sequence_alignment    j_sequence_alignment_aa j_germline_alignment       j_germline_alignment_aa fwr1    fwr1_aa cdr1    cdr1_aa fwr2    fwr2_aa cdr2    cdr2_aa fwr3    fwr3_aa    fwr4    fwr4_aa cdr3    cdr3_aa junction        junction_length junction_aa     junction_aa_lengthv_score  d_score j_score v_cigar d_cigar j_cigar v_support       d_support       j_support       v_identityd_identity       j_identity      v_sequence_start        v_sequence_end  v_germline_start        v_germline_end     d_sequence_start        d_sequence_end  d_germline_start        d_germline_end  j_sequence_start  j_sequence_end   j_germline_start        j_germline_end  fwr1_start      fwr1_end        cdr1_start      cdr1_end   fwr2_start      fwr2_end        cdr2_start      cdr2_end        fwr3_start      fwr3_end        fwr4_start fwr4_end        cdr3_start      cdr3_end        np1     np1_length      np2     np2_length      v_family   d_family        j_family        cdr3_aa_length
```

I then used the script:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/extract_relevant_cols_from_tsv.py
```

to extract the needed columsn from the tsv and store it in a csv file.

output csv file:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/h2l_div_beam_search_2_epoch_10_lr_1e-4_wd_0.1/full_test_set_true_gen_seqs_all_relevant_cols.csv
```

For the actual analysis of the regions, i used an R script and ran it locally (not on the server). 

The R script can still be found at the server:

```bash
/ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/sqlite3_data_for_analysis/evaluate_test_set_by_regions/evaluate_by_regions.R
```

I used R locally because the data manipulation was easier.
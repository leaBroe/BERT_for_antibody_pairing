# AbNatiV-humanness score

[Assessing antibody and nanobody nativeness for hit selection and humanization with AbNatiV](https://www.nature.com/articles/s42256-023-00778-3)

[](https://gitlab.developers.cam.ac.uk/ch/sormanni/abnativ)

[Register :: Chemistry of Health - software](https://www-cohsoftware.ch.cam.ac.uk/index.php/register/register_success_validate/0)

[www-cohsoftware.ch.cam.ac.uk](https://www-cohsoftware.ch.cam.ac.uk/application/single_pages/abnativ/data/lea_b_546844921_profiles/single_seq_abnativ_profile.png)

```bash
# Align and Compute the AbNatiV VH-humanness scores (sequence and residue levels) for a set of sequences in a fasta file
# In directory test/test_scoring are saved test_vh_abnativ_seq_scores.csv and test_vh_abnativ_res_scores.csv
# Profile figures are saved in test/test_vh_profiles for each sequence
abnativ score -nat VH -i test/4_heavy_sequences.fa -odir test/test_results2 -oid test_vh -align -ncpu 4

# For one single sequence
abnativ score -nat VH -i EIQLVQSGPELKQPGETVRISCKASGYTFTNYGMNWVKQAPGKGLKWMGWINTYTGEPTYAADFKRRFTFSLETSASTAYLQISNLKNDDTATYFCAKYPHYYGSSHWYFDVWGAGTTVTVSS -odir test/test_results2 -oid test_single_vh -align -plot
```

```bash
abnativ score [-h] [-nat NATIVENESS_TYPE] [-mean] [-i INPUT_FILEPATH_OR_SEQ] [-odir OUTPUT_DIRECTORY] [-oid OUTPUT_ID] [-align] [-ncpu NCPU] [-isVHH] [-plot]

Use a trained AbNatiV model (default or custom) to score a set of input antibody sequences

optional arguments:
  -h, --help            show this help message and exit
  -nat NATIVENESS_TYPE, --nativeness_type NATIVENESS_TYPE
                        To load the AbNatiV default trained models type VH, VKappa, VLambda, or VHH, otherwise add directly the path to
                        your own AbNatiV trained checkpoint .ckpt (default: VH)
  -mean, --mean_score_only
                        Generate only a file with a score per sequence. If not, generate a second file with a nativeness score per
                        positin with a probability score for each aa at each position. (default: False)
  -i INPUT_FILEPATH_OR_SEQ, --input_filepath_or_seq INPUT_FILEPATH_OR_SEQ
                        Filepath to the fasta file .fa to score or directly a single string sequence (default: to_score.fa)
  -odir OUTPUT_DIRECTORY, --output_directory OUTPUT_DIRECTORY
                        Filepath of the folder where all files are saved (default: abnativ_scoring)
  -oid OUTPUT_ID, --output_id OUTPUT_ID
                        Prefix of all the saved filenames (e.g., name sequence) (default: antibody_vh)
  -align, --do_align    Do the alignment and the cleaning of the given sequences before scoring. This step can takes a lot of time if
                        the number of sequences is huge. (default: False)
  -ncpu NCPU, --ncpu NCPU
                        If ncpu>1 will parallelise the algnment process (default: 1)
  -isVHH, --is_VHH      Considers the VHH seed for the alignment. It is more suitable when aligning nanobody sequences (default: False)
  -v, --verbose         Print more details about every step. (default: False)
  -plot, --is_plotting_profiles
                        Plot profile for every input sequence and save them in {output_directory}/{output_id}_profiles. (default:
                        False)

```

```bash
abnativ score -nat VKappa -i /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1/igk_true_sequences.fasta -odir /ibmm_data2/oas_database/paired_lea_tmp/paired_model/BERT2BERT/humanness_score_AbnatiV/PLAbDab_human_healthy_full_diverse_beam_search_5_temp_0.2_max_length_150_early_stopping_true_batch_size_64_epochs_50_lr_0.0001_wd_0.1/abnativ_output -oid igk_true_sequences 
```
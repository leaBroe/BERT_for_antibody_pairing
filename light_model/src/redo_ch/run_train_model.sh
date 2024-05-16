#!/bin/bash

#SBATCH --gres=gpu:a100:1
#SBATCH --job-name=light_conf_3

#--config_name 'config3.json' \

#    --train_file /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/train_test_val_datasets/light_all_seqs_train_no_ids.txt \
#    --validation_file /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/train_test_val_datasets/light_all_seqs_val_no_ids.txt \

#    --train_file /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/training_set_test.txt \
#    --validation_file /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/test_set_test.txt \


eval "$(conda shell.bash hook)"
conda init bash
conda activate lea_env
/home/leab/anaconda3/envs/lea_env/bin/python run_mlm.py \
    --model_type 'roberta' \
    --tokenizer_name ./ProteinTokenizer \
    --train_file /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/train_test_val_datasets/light_all_seqs_train_no_ids.txt \
    --validation_file /ibmm_data2/oas_database/paired_lea_tmp/light_model/data/train_test_val_datasets/light_all_seqs_val_no_ids.txt \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 16 \
    --do_train \
    --do_eval \
    --do_predict \
    --evaluation_strategy 'epoch' \
    --save_strategy 'epoch' \
    --learning_rate 5e-4 \
    --weight_decay 0.1 \
    --num_train_epochs 500 \
    --lr_scheduler_type 'linear' \
    --log_level 'info' \
    --seed 42 \
    --data_seed 42 \
    --bf16 False \
    --project_name light_model_unpaired_all_seqs \
    --load_best_model_at_end True \
    --metric_for_best_model 'loss' \
    --line_by_line \
    --greater_is_better False \
    --config_name 'config3.json' \
    --report_to 'wandb' \
    --max_seq_length 512 \
    --output_dir ./FULL_config_3_roberta_run_lr5e-4_500epochs_max_seq_length_512

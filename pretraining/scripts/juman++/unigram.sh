#!/bin/bash

# specify job-related names
JOB_NAME=jumanpp_unigram
PRE_TOKENIZER=juman
SUBWORD_TOKENIZER=unigram

# run
python -m torch.distributed.launch \
    --master_port 1235 \
    --nproc_per_node 4 \
    --nnodes 1 \
src/pretrain.py \
    --tokenizer_path ../data/dict/${JOB_NAME}.json \
    --pretokenizer_type ${PRE_TOKENIZER} \
    --data_path ../data/pretraining/juman++/${SUBWORD_TOKENIZER} \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --adam_epsilon 1e-6 \
    --max_grad_norm 1.0 \
    --max_steps 500000 \
    --warmup_steps 10000 \
    --save_steps 100000 \
    --prediction_loss_only True \
    --seed=42 \
    --per_device_train_batch_size 32 \
    --logging_steps 50 \
    --report_to tensorboard \
    --output_dir ../weights/${JOB_NAME} \
    --overwrite_output_dir \
    --logging_dir ../log/pretraining/${JOB_NAME} \
    --disable_tqdm True \
    --fp16

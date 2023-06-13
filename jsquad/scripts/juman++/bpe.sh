JOB_NAME=jumanpp_bpe
PRE_TOKENIZER=jumanpp

python src/fine-tuning_squad.py \
    --tokenizer_path ../data/dict/${JOB_NAME}.json \
    --pretrain_path ../weights/${JOB_NAME}/checkpoint-500000 \
    --pretokenizer_type ${PRE_TOKENIZER} \
    --train_data_path ../data/JGLUE/datasets/jsquad-v1.1/train-v1.1.json \
    --valid_data_path ../data/JGLUE/datasets/jsquad-v1.1/valid-v1.1.json \
    --max_seq_length 384 \
    --vocab_size 30000 \
    --learning_rate 3e-5 \
    --num_train_epoch 5 \
    --warmup_ratio 0.1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --report_to tensorboard \
    --output_dir ../weights/jsquad/${JOB_NAME}/cross_validation_ \
    --logging_dir ../log/jsquad/${JOB_NAME} \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --load_best_model_at_end True \
    --metric_for_best_model f-1 \
    --greater_is_better True \
    --fp16 True \
    --disable_tqdm True \
    --include_inputs_for_metrics=True 

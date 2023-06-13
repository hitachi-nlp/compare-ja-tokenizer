JOB_NAME=vaporetto_bpe
PRE_TOKENIZER=vaporetto

python src/fine-tuning_marc-ja.py \
    --tokenizer_path ../data/dict/${JOB_NAME}.json \
    --pretrain_path ../weights/${JOB_NAME}/checkpoint-500000 \
    --pretokenizer_type ${PRE_TOKENIZER} \
    --vaporetto_model_path ../data/dict/bccwj-suw+unidic+tag/bccwj-suw+unidic+tag.model.zst \
    --train_data_path ../data/JGLUE/datasets/marc_ja-v1.0/train-v1.0.json \
    --val_data_path ../data/JGLUE/datasets/marc_ja-v1.0/valid-v1.0.json \
    --max_length 512 \
    --vocab_size 30000 \
    --learning_rate 3e-5 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --weight_decay 0.01 \
    --num_train_epoch 5 \
    --warmup_ratio 0.1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --report_to tensorboard \
    --output_dir ../weights/marc-ja/${JOB_NAME}/cross_validation_ \
    --logging_dir ../log/marc-ja/${JOB_NAME} \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --greater_is_better True \
    --fp16 True \
    --disable_tqdm True

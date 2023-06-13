JOB_NAME=jumanpp_unigram
PRE_TOKENIZER=jumanpp

python src/fine-tuning_ner.py \
    --tokenizer_path ../data/dict/${JOB_NAME}.json \
    --pretrain_path ../weights/${JOB_NAME}/checkpoint-500000 \
    --pretokenizer_type ${PRE_TOKENIZER} \
    --data_path ../data/ner-wikipedia-dataset/ner.json \
    --max_length 128 \
    --vocab_size 30000 \
    --learning_rate 3e-05 \
    --adam_beta1 0.9 \
    --adam_beta2 0.999 \
    --weight_decay 0.01 \
    --num_train_epochs 5 \
    --warmup_ratio 0.1 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --report_to tensorboard \
    --output_dir ../weights/ner/${JOB_NAME} \
    --logging_dir ../log/ner/${JOB_NAME} \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --load_best_model_at_end True \
    --metric_for_best_model accuracy \
    --greater_is_better True \
    --fp16 True \
    --disable_tqdm True 

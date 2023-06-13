JOB_NAME=mecab_bpe
PRE_TOKENIZER=mecab

python src/run_swag.py \
     --model_name_or_path ../weights/${JOB_NAME}/checkpoint-500000 \
     --tokenizer_path ../data/dict/${JOB_NAME}.json \
     --pretokenizer_type ${PRE_TOKENIZER} \
     --max_seq_length 64 \
     --per_device_train_batch_size 8 \
     --per_device_eval_batch_size 8 \
     --learning_rate 3e-05 \
     --adam_beta1 0.9 \
     --adam_beta2 0.999 \
     --weight_decay 0.01 \
     --evaluation_strategy epoch \
     --save_strategy epoch \
     --report_to tensorboard \
     --num_train_epoch 5 \
     --num_train_epochs 5 \
     --metric_for_best_model accuracy \
     --output_dir ../weights/jcommonsenseqa/${JOB_NAME}/cross_validation_ \
     --logging_dir ../log/jcommonsenseqa/${JOB_NAME} \
     --train_file ../data/JGLUE/datasets/jcommonsenseqa-v1.0/train-v1.0.json \
     --validation_file ../data/JGLUE/datasets/jcommonsenseqa-v1.0/valid-v1.0.json \
     --train_data_path ../data/JGLUE/datasets/jcommonsenseqa-v1.0/train-v1.0.json \
     --val_data_path ../data/JGLUE/datasets/jcommonsenseqa-v1.0/valid-v1.0.json \
     --use_fast_tokenizer False \
     --load_best_model_at_end True \
     --metric_for_best_model accuracy \
     --greater_is_better True \
     --evaluation_strategy epoch \
     --fp16 True \
     --warmup_ratio 0.1

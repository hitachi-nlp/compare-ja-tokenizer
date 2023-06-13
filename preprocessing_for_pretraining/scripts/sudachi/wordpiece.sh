#!/bin/bash
SUBWORD_TOKENIZER=wordpiece

python src/sudachi/preprocess.py \
    --wikipedia_data_path ../data/pretraining/sudachi/pretrain_sentence_wiki.txt \
    --cc100_data_path ../data/pretraining/sudachi/pretrain_sentence_cc100.txt \
    --tokenizer_path ../data/dict/sudachi_${SUBWORD_TOKENIZER}.json \
    --output_dir ../data/pretraining/sudachi/${SUBWORD_TOKENIZER}

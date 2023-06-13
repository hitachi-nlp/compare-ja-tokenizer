#!/bin/bash
SUBWORD_TOKENIZER=unigram

python src/nothing/preprocess.py \
    --wikipedia_data_path ../data/pretraining/pretrain_sentence_wiki.txt \
    --cc100_data_path ../data/pretraining/nothing/pretrain_sentence_cc100.txt \
    --tokenizer_path ../data/dict/nothing_${SUBWORD_TOKENIZER}.json \
    --output_dir ../data/pretraining/nothing/${SUBWORD_TOKENIZER}

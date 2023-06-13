#!/bin/bash
SUBWORD_TOKENIZER=unigram

python src/mecab/preprocess.py \
    --wikipedia_data_path ../data/pretraining/mecab/pretrain_sentence_wiki.txt \
    --cc100_data_path ../data/pretraining/mecab/pretrain_sentence_cc100.txt \
    --tokenizer_path ../data/dict/mecab_${SUBWORD_TOKENIZER}.json \
    --output_dir ../data/pretraining/mecab/${SUBWORD_TOKENIZER}

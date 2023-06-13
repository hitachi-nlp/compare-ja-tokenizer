#!/bin/bash
SUBWORD_TOKENIZER=unigram

python src/juman++/preprocess.py \
    --wikipedia_data_path ../data/pretraining/juman++/pretrain_sentence_wiki.txt \
    --cc100_data_path ../data/pretraining/juman++/pretrain_sentence_cc100.txt \
    --tokenizer_path ../data/dict/jumanpp_${SUBWORD_TOKENIZER}.json \
    --output_dir ../data/pretraining/juman++/${SUBWORD_TOKENIZER}

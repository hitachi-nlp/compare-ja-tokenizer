#!/bin/bash
SUBWORD_TOKENIZER=wordpiece

python src/vaporetto/preprocess.py \
    --wikipedia_data_path ../data/pretraining/vaporetto/pretrain_sentence_wiki.txt \
    --cc100_data_path ../data/pretraining/vaporetto/pretrain_sentence_cc100.txt \
    --tokenizer_path ../data/dict/vaporetto_${SUBWORD_TOKENIZER}.json \
    --output_dir ../data/pretraining/vaporetto/${SUBWORD_TOKENIZER}

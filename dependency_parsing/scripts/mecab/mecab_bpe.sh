#!/bin/bash

SEED=78
SUBWORD_TOKENIZER=bpe
PRETOKENIZER=mecab
export TOKENIZERS_PARALLELISM=true

python -u -m supar.cmds.biaffine_dep train -b -d 0 -c ./config/${PRETOKENIZER}/${PRETOKENIZER}_${SUBWORD_TOKENIZER}.ini -p ${PRETOKENIZER}_${SUBWORD_TOKENIZER}_${SEED} \
	--seed=${SEED} \
	--train ../data/ud-treebanks-v2.10/UD_Japanese-GSD/ja_gsd-ud-train.conllu \
	--dev ../data/ud-treebanks-v2.10/UD_Japanese-GSD/ja_gsd-ud-dev.conllu \
	--test ../data/ud-treebanks-v2.10/UD_Japanese-GSD/ja_gsd-ud-test.conllu \
	--encoder bert \
	--pretokenizer_type ${PRETOKENIZER} \
	--bert ../weights/${PRETOKENIZER}_${SUBWORD_TOKENIZER} \
	--bert_tokenizer_file ../data/dict/${PRETOKENIZER}_${SUBWORD_TOKENIZER}.json

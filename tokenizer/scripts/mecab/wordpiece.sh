SUBWORD_TOKENIZER=wordpiece

python src/mecab/train_with_mecab.py \
    --data_path ../data/tokenizer/tokenizer_data.txt \
    --output_path ../data/dict/mecab_${SUBWORD_TOKENIZER}.json \
    --subword_method ${SUBWORD_TOKENIZER} \
    --vocab_size 30000

SUBWORD_TOKENIZER=bpe

python src/nothing/train_with_nothing.py \
    --data_path ../data/tokenizer/tokenizer_data.txt \
    --output_path ../data/dict/nothing_${SUBWORD_TOKENIZER}.json \
    --subword_method ${SUBWORD_TOKENIZER} \
    --vocab_size 30000

SUBWORD_TOKENIZER=unigram

python src/sudachi/train_with_sudachi.py \
    --data_path ../data/tokenizer/sudachi/pretokenized_data.txt \
    --output_path ../data/dict/sudachi_${SUBWORD_TOKENIZER}.json \
    --subword_method ${SUBWORD_TOKENIZER} \
    --vocab_size 30000

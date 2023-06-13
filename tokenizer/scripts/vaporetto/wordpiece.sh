SUBWORD_TOKENIZER=wordpiece

python src/vaporetto/train_with_vaporetto.py \
    --data_path ../data/tokenizer/vaporetto/pretokenized_data.txt \
    --output_path ../data/dict/vaporetto_${SUBWORD_TOKENIZER}.json \
    --subword_method ${SUBWORD_TOKENIZER} \
    --vocab_size 30000

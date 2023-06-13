SUBWORD_TOKENIZER=unigram

python src/juman++/train_with_jumanpp.py \
    --data_path ../data/tokenizer/juman++/pretokenized_data.txt \
    --output_path ../data/dict/jumanpp_${SUBWORD_TOKENIZER}.json \
    --subword_method ${SUBWORD_TOKENIZER} \
    --vocab_size 30000

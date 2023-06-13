Preprocessing for tokenizer training
===
Here, we explain the procedure to preprocess the pretraining data for BERT models and training data for tokenizers.

## 1. Download
### Wikipedia
The following commands will generate the dataset under `./data/original/wikipedia_ja`.
```bash
cd compare-ja-tokenizer/preprocessing_for_tokenizers/
python src/generate_wiki_dataset.py
```

### CC-100
```bash
cd ../data/original
wget http://data.statmt.org/cc-100/ja.txt.xz
gunzip ja.txt.xz
cd ../../compare-ja-tokenizer/preprocessing_for_tokenizers/
```

## 2. Preprocess Wikipedia
The following command will generate the data as `../data/sentence/tokenizer_data_wiki.txt`. Note that this process takes a lot of time.
```bash
python src/make_sentence_data_wiki.py
```

## 3. Preprocess CC-100
The following command will generate the data as `../data/sentence/tokenizer_data_cc100.txt`. Note that this process takes a lot of time.
```bash
python src/make_sentence_data_cc100.py
```

## 4. Create a file for tokenizer training
```bash
# Pick up 10M sentences
shuf -n 10000000 ../data/sentence/tokenizer_data_wiki.txt > ../data/tokenizer/tokenizer_data.txt
```

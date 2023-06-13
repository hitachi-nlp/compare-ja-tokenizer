Preprocessing for pretraining
===

Here, we explain how to generate tokenized data for pretraining.

## 1. Generate pretraining raw text data
### Wikipedia
The following command will generate `pretrain_sentence_wiki.txt` under `../data/pretraining`.

```bash
cd ./compare-ja-tokenizer/preprocessing_for_pretraining
python src/make_data_wiki.py \
    --tokenizr_path ../data/dict/mecab_bpe.json \
    --pretokenizer_type mecab
```

### CC-100
The following command will generate `pretrain_sentence_cc100.txt` under `../data/pretraining`.

```bash
python src/make_data_cc100.py \
    --tokenizr_path ../data/dict/mecab_bpe.json \
    --pretokenizer_type mecab
```

## 2. Generate pretraining datasets
### 2.1 MeCab
*  Pre-tokenize texts (Takes time.)
```bash
# Wikipedia
cat ../data/pretraining/pretrain_sentence_wiki.txt | mecab --input-buffer-size=1048576 -Owakati > ../data/pretraining/mecab/pretrain_sentence_wiki.txt
# CC-100
cat ../data/pretraining/pretrain_sentence_cc100.txt | mecab -Owakati > ../data/pretraining/mecab/pretrain_sentence_cc100.txt
```

* Generate tokenized dataset
```bash
# BPE
bash ./scripts/mecab/bpe.sh

# WordPiece
bash ./scripts/mecab/wordpiece.sh

# Unigram
bash ./scripts/mecab/unigram.sh
```

### 2.2 Juman++
* Pre-tokenize texts (Takes time.)
```bash
# Wikipedia
jumanpp ../data/pretraining/pretrain_sentence_wiki.txt --segment > ../data/pretraining/juman++/pretrain_sentence_wiki.txt
# CC-100
jumanpp ../data/pretraining/pretrain_sentence_cc100.txt --segment > ../data/pretraining/juman++/pretrain_sentence_cc100.txt
```

* Generate tokenized dataset
```bash
# BPE
bash ./scripts/juman++/bpe.sh

# WordPiece
bash ./scripts/juman++/wordpiece.sh

# Unigram
bash ./scripts/juman++/unigram.sh
```

### 2.3 Sudachi
* Pre-tokenize texts (Takes time.)
```bash
# Wikipedia
python src/sudachi/pretokenize_sudachi.py \
    --input_path ../data/pretraining/pretrain_sentence_wiki.txt \
    --output_path ../data/pretraining/sudachi/pretrain_sentence_wiki.txt

# CC-100
python src/sudachi/pretokenize_sudachi.py \
    --input_path ../data/pretraining/pretrain_sentence_cc100.txt \
    --output_path ../data/pretraining/sudachi/pretrain_sentence_cc100.txt
```

* Generate tokenized dataset
```bash
# BPE
bash ./scripts/sudachi/bpe.sh

# WordPiece
bash ./scripts/sudachi/wordpiece.sh

# Unigram
bash ./scripts/sudachi/unigram.sh
```

### 2.4 Vaporetto
* Pre-tokenize texts (Takes time.)
```bash
# Wikipedia
cat ../data/pretraining/pretrain_sentence_wiki.txt | cargo run --release -p predict -- --model /path/to/bccwj-suw+unidic+tag.model.zst > ../data/pretraining/vaporetto/temp_pretrain_sentence_wiki.txt
sed -e 's/\\//g' ../data/pretraining/vaporetto/temp_pretrain_sentence_wiki.txt > ../data/pretraining/vaporetto/pretrain_sentence_wiki.txt

# CC-100
cat ../data/pretraining/pretrain_sentence_cc100.txt | cargo run --release -p predict -- --model /path/to/bccwj-suw+unidic+tag.model.zst > ../data/pretraining/vaporetto/temp_pretrain_sentence_cc100.txt
sed -e 's/\\//g' ../data/pretraining/vaporetto/temp_pretrain_sentence_cc100.txt > ../data/pretraining/vaporetto/pretrain_sentence_cc100.txt
```

* Generate tokenized dataset
```bash
# BPE
bash ./scripts/vaporetto/bpe.sh

# WordPiece
bash ./scripts/vaporetto/wordpiece.sh

# Unigram
bash ./scripts/vaporetto/unigram.sh
```

### 2.5 Nothing
```bash
# BPE
bash ./scripts/Nothing/bpe.sh

# WordPiece
bash ./scripts/Nothing/wordpiece.sh

# Unigram
bash ./scripts/Nothing/unigram.sh
```

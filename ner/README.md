Named Entity Recognition
===

Here, we explain how to fine-tune our BERT models on the Japanese NER dataset.

## 1. Download the dataset
```bash
cd ./compare-ja-tokenizer/data
git clone https://github.com/stockmarkteam/ner-wikipedia-dataset.git
cd ..
```

## 2. Fine-tuning

### 2.1 MeCab
```bash
cd ./ner

# BPE
bash ./scirpts/mecab/bpe.sh

# WordPiece
bash ./scirpts/mecab/wordpiece.sh

# Unigram
bash ./scirpts/mecab/unigram.sh
```

* The resulting weight files will be stored under `./weights/ner`. If you want to change the location, you must edit `output_dir` in each shell script.
* Log files will be saved under `./log/ner`. If you want to change the location, you must edit `--logging_dir` in each shell script.


### 2.2 Juman++
```bash
# BPE
bash ./scripts/juman++/bpe.sh

# WordPiece
bash ./scripts/juman++/wordpiece.sh

# Unigram
bash ./scripts/juman++/unigram.sh
```


### 2.3 Sudachi
```bash
# BPE
bash ./scirpts/sudachi/bpe.sh

# WordPiece
bash ./scirpts/sudachi/wordpiece.sh

# Unigram
bash ./scirpts/sudachi/unigram.sh
```


### 2.4 Vaporetto
```bash
# BPE
bash ./scirpts/vaporetto/bpe.sh

# WordPiece
bash ./scirpts/vaporetto/wordpiece.sh

# Unigram
bash ./scirpts/vaporetto/unigram.sh
```
* You need to specify the location of `bccwj-suw+unidic+tag.model.zst` as `vaporetto_model_path` if you do not download it under `../data/dict/bccwj-suw+unidic+tag/`.


### 2.5 Nothing
```bash
# BPE
bash ./scirpts/nothing/bpe.sh

# WordPiece
bash ./scirpts/nothing/wordpiece.sh

# Unigram
bash ./scirpts/nothing/unigram.sh
```

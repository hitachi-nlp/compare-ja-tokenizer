Pretraining
===

Here, we explain how to pretrain BERT models using the preprocessed data and tokenizer.

## 1. MeCab
```bash
cd ./compare-ja-tokenizer/pretraining

# BPE
bash ./scirpts/mecab/bpe.sh

# WordPiece
bash ./scirpts/mecab/wordpiece.sh

# Unigram
bash ./scirpts/mecab/unigram.sh
```

* The resulting weight files will be stored under `./weights`. If you want to change the location, you must edit `output_dir` in each shell script.
* Log files will be saved under `./log/pretraining`. If you want to change the location, you must edit `--logging_dir` in each shell script.


## 2. Juman++
```bash
# BPE
bash ./scripts/juman++/bpe.sh

# WordPiece
bash ./scripts/juman++/wordpiece.sh

# Unigram
bash ./scripts/juman++/unigram.sh
```

## 3. Sudachi
```bash
# BPE
bash ./scirpts/sudachi/bpe.sh

# WordPiece
bash ./scirpts/sudachi/wordpiece.sh

# Unigram
bash ./scirpts/sudachi/unigram.sh
```

## 4. Vaporetto
```bash
# BPE
bash ./scirpts/vaporetto/bpe.sh

# WordPiece
bash ./scirpts/vaporetto/wordpiece.sh

# Unigram
bash ./scirpts/vaporetto/unigram.sh
```

## 5. Nothing
```bash
# BPE
bash ./scirpts/nothing/bpe.sh

# WordPiece
bash ./scirpts/nothing/wordpiece.sh

# Unigram
bash ./scirpts/nothing/unigram.sh
```
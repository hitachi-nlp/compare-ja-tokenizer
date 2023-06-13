JSTS
===

Here, we explain how to fine-tune our BERT models on the JSTS dataset.

## 1. MeCab
```bash
cd ./compare-ja-tokenizer/jsts

# BPE
bash ./scirpts/mecab/bpe.sh

# WordPiece
bash ./scirpts/mecab/wordpiece.sh

# Unigram
bash ./scirpts/mecab/unigram.sh
```

* The resulting weight files will be stored under `./weights/jsts`. If you want to change the location, you must edit `output_dir` in each shell script.
* Log files will be saved under `./log/jsts`. If you want to change the location, you must edit `--logging_dir` in each shell script.


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
* You need to specify the location of `bccwj-suw+unidic+tag.model.zst` as `vaporetto_model_path` if you do not download it under `../data/dict/bccwj-suw+unidic+tag/`.


## 5. Nothing
```bash
# BPE
bash ./scirpts/nothing/bpe.sh

# WordPiece
bash ./scirpts/nothing/wordpiece.sh

# Unigram
bash ./scirpts/nothing/unigram.sh
```

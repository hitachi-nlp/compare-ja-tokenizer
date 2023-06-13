Training tokenizers
===

Here, we explain how to generate dictionary files for tokenizers.

## 1. Training tokenizers with MeCab
```bash
cd ./compare-ja-tokenizer/tokenizer

# BPE (Takes time.)
bash ./scripts/mecab/bpe.sh

# WordPiece (Takes time.)
bash ./scripts/mecab/wordpiece.sh

# Unigram (Takes time.)
bash ./scripts/mecab/unigram.sh
```

You might need to specify the location of `mecabrc` as follows if you have installed MeCab locally.
```bash
echo 'export MECABRC="[your path]/etc/mecabrc"' >> ~/.bashrc
```
> [your path] should be `$HOME/usr` if you follow our instructions.


## 2. Training tokenizers with Juman++
```bash
# Pre-tokenize (Takes time.)
jumanpp ../data/tokenizer/tokenizer_data.txt --segment > ../data/tokenizer/juman++/pretokenized_data.txt

# BPE
bash ./scripts/juman++/bpe.sh

# WordPiece
bash ./scripts/juman++/wordpiece.sh

# Unigram
bash ./scripts/juman++/unigram.sh
```


## 3. Training tokenizers with Sudachi
```bash
# Pre-tokenize (Takes time.)
python src/sudachi/pretokenize_sudachi.py \
    --input_path ../data/tokenizer/tokenizer_data.txt \
    --output_path  ../data/tokenizer/sudachi/pretokenized_data.txt

# BPE
bash ./scripts/sudachi/bpe.sh

# WordPiece
bash ./scripts/sudachi/wordpiece.sh

# Unigram
bash ./scripts/sudachi/unigram.sh
```


## 4. Training tokenizers with Vaporetto
```bash
# Pre-tokenize (Takes time.)
cat ../data/tokenizer/tokenizer_data.txt | cargo run --release -p predict -- --model /path/to/bccwj-suw+unidic+tag.model.zst > ../data/tokenizer/vaporetto/temp_pretokenized_data.txt
sed -e 's/\\//g' ../data/tokenizer/vaporetto/temp_pretokenized_data.txt > ../data/tokenizer/vaporetto/pretokenized_data.txt
```
> You must specify the path to `bccwj-suw+unidic+tag.model.zst`. In default, your model should be downloaded under the path: `../data/bccwj-suw+unidic+tag/`.

```bash
# BPE
bash ./scripts/vaporetto/bpe.sh

# WordPiece
bash ./scripts/vaporetto/wordpiece.sh

# Unigram
bash ./scripts/vaporetto/unigram.sh
```


## 5. Training tokenizers without morphological analyzers  
```bash
# BPE
bash ./scripts/nothing/bpe.sh

# WordPiece
bash ./scripts/nothing/wordpiece.sh

# Unigram
bash ./scripts/nothing/unigram.sh
```

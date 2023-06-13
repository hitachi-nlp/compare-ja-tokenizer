Dependency Parsing
=====

## 1. Download the dataset
Please download the UD dataset from https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-4758 and place and unfreeze it under `../data`.

## 2. Install the SuPar library
We use the **customized** [SuPar](https://github.com/yzhangcs/parser) library in order to use our own tokenizers.

```bash
cd ./compare-ja-tokenizer/dependency_parsing
pip install supar-1.1.4-py3-none-any.whl
```
> In case you encounter an error when installing the above package, please run `pip install setuptools --upgrade` and retry the installation.

## 3. Fine-tuning
### 3.1 MeCab
```bash
# BPE
bash ./scripts/mecab/mecab_bpe.sh

# WordPiece
bash ./scripts/mecab/mecab_wordpiece.sh

# Unigram
bash ./scripts/mecab/mecab_unigram.sh
```
* Log files will be generated under this directory named as `${morphological_analyzer_name}_${subword_tokenizer_name}_${seed}.log`.

### 3.2 Juman++
```bash
# BPE
bash ./scripts/juman++/jumanpp_bpe.sh

# WordPiece
bash ./scripts/juman++/jumanpp_wordpiece.sh

# Unigram
bash ./scripts/juman++/jumanpp_unigram.sh
```

### 3.3 Sudachi
```bash
# BPE
bash ./scripts/sudachi/sudachi_bpe.sh

# WordPiece
bash ./scripts/sudachi/sudachi_wordpiece.sh

# Unigram
bash ./scripts/sudachi/sudachi_unigram.sh
```

### 3.4 Vaporetto
```bash
# BPE
bash ./scripts/vaporetto/vaporetto_bpe.sh

# WordPiece
bash ./scripts/vaporetto/vaporetto_wordpiece.sh

# Unigram
bash ./scripts/vaporetto/vaporetto_unigram.sh
```
* You need to specify the location of `bccwj-suw+unidic+tag.model.zst` as `unidic_path` both in `scripts` and `config` if you do not download it under `../data/dict/bccwj-suw+unidic+tag/`.


### 3.5 Nothing
```bash
# BPE
bash ./scripts/nothing/nothing_bpe.sh

# WordPiece
bash ./scripts/nothing/nothing_wordpiece.sh

# Unigram
bash ./scripts/nothing/nothing_unigram.sh
```

## License
For the original implementation of the SuPar library:  
```
MIT License

Copyright (c) 2018-2023 Yu Zhang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

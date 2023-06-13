How do different tokenizers perform on downstream tasks in scriptio continua languages?: A case study in Japanese
===

This is the official implementation of the paper titled: "How do different tokenizers perform on downstream tasks in scriptio continua languages?: A case study in Japanese". To reproduce our results, please follow the following instructions.


## 1. Requirements  
* Python >= 3.9
* PyTorch 1.8.1  
* Transformers 4.24.0.dev0  


## 2. Installation  
### 2.1 PyTorch  
```bash
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
```

### 2.2 Transformers  
```bash
git clone https://github.com/huggingface/transformers.git
cd transformers
pip install -e .
cd ..
```

### 2.3 Other Python packages
```bash
pip install -r requirements.txt
```

### 2.4 Japanese Morphological Analyzers  
Here, we install required packages under `${HOME}/usr`, but you can choose your preferred location by modifying `--prefix`.

#### 2.4.1 MeCab
* Model
    ```bash
    git clone https://github.com/taku910/mecab.git
    cd mecab/mecab
    ./configure --prefix=${HOME}/usr --with-charset=UTF8
    make
    make install
    cd ../..
    ```

* Dictionary
    ```bash
    wget "https://drive.google.com/uc?export=download&id=0B4y35FiV1wh7MWVlSDBCSXZMTXM" -O mecab-ipadic-2.70-20070801.tar.gz
    tar xvzf mecab-ipadic-2.7.0-20070801.tar.gz
    cd mecab-ipadic-2.7.0-20070801
    ./configure --with-mecab-config=$HOME/usr/bin/mecab-config --with-charset=UTF8 --prefix=$HOME/usr
    make
    make install
    cd ..
    ```

#### 2.4.2 Juman++
```bash
wget "https://github.com/ku-nlp/jumanpp/releases/download/v2.0.0-rc3/jumanpp-2.0.0-rc3.tar.xz"
tar xvJf jumanpp-2.0.0-rc3.tar.xz
cd jumanpp-2.0.0-rc3
mkdir build && cd build
curl -LO https://github.com/catchorg/Catch2/releases/download/v2.13.8/catch.hpp
mv catch.hpp ../libs/
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$HOME/usr
make
make install
echo 'export PATH=$PATH:$HOME/usr' >> ~/.bashrc
echo 'export PATH=$PATH:$HOME/usr/bin' >> ~/.bashrc
cd ..
```

#### 2.4.3 Sudachi
```bash
pip install sudachipy
pip install sudachidict_core
```

#### 2.4.4 Vaporetto
See https://github.com/daac-tools/vaporetto for more details.

```bash
cd data/dict
wget https://github.com/daac-tools/vaporetto/releases/download/v0.5.0/bccwj-suw+unidic+tag.tar.xz
tar xf ./bccwj-suw+unidic+tag.tar.xz
cd ../..
```

## 3. Preprocessing data for tokenizer training
Please see [preprocessing_for_tokenizers](./preprocessing_for_tokenizers/).


## 4. Training tokenizers
Please see [tokenizer](./tokenizer/).


## 5. Preprocessing data for pretraining
Please see [preprocessing_for_pretraining](./preprocessing_for_pretraining/).


## 6. Pretraining
Please see [pretraining](./pretraining/).


## 7. Fine-tuning
### 7.1 JGLUE
First, please clone the JGLUE repository and download the JGLUE dataset under `./data`, following https://github.com/yahoojapan/JGLUE.

### 7.1.1 MARC-ja
Please see [marc-ja](./marc-ja/).

### 7.1.2 JSTS
Please see [jsts](./jsts/).

### 7.1.3 JNLI
Please see [jnli](./jnli/).

### 7.1.4 JSQuAD
Please see [jsquad](./jsquad/).

### 7.1.5 JCommonsenseQA
Please see [jcommonsenseqa](./jcommonsenseqa/).

### 7.2 NER
Please see [ner](./ner/).

### 7.3 UD
Please see [dependency_parsing](./dependency_parsing/).

## Pretrained Weights
The pretrained weights are available on the Hugging Face Hub.

| | BPE | Unigram | WordPiece |  
| :-: | :-: | :-: | :-: |
| MeCab | TBA  | TBA | TBA |
| Juman++ | TBA  | TBA | TBA |
| Sudachi | TBA  | TBA | TBA |
| Vaporetto | TBA  | TBA | TBA |
| Nothing | TBA  | TBA | TBA |


## Citation
```
TBA
```

## License
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a> unless specified.

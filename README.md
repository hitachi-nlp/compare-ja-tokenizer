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
| MeCab | [bert-base_mecab-bpe](https://huggingface.co/hitachi-nlp/bert-base_mecab-bpe)  | [bert-base_mecab-unigram](https://huggingface.co/hitachi-nlp/bert-base_mecab-unigram) | [bert-base_mecab-wordpiece](https://huggingface.co/hitachi-nlp/bert-base_mecab-wordpiece) |
| Juman++ | [bert-base_jumanpp-bpe](https://huggingface.co/hitachi-nlp/bert-base_jumanpp-bpe)  | [bert-base_jumanpp-unigram](https://huggingface.co/hitachi-nlp/bert-base_jumanpp-unigram) | [bert-base_jumanpp-wordpiece](https://huggingface.co/hitachi-nlp/bert-base_jumanpp-wordpiece) |
| Sudachi | [bert-base_sudachi-bpe](https://huggingface.co/hitachi-nlp/bert-base_sudachi-bpe)  | [bert-base_sudachi-unigram](https://huggingface.co/hitachi-nlp/bert-base_sudachi-unigram) | [bert-base_sudachi-wordpiece](https://huggingface.co/hitachi-nlp/bert-base_sudachi-wordpiece) |
| Vaporetto | [bert-base_vaporetto-bpe](https://huggingface.co/hitachi-nlp/bert-base_vaporetto-bpe)  | [bert-base_vaporetto-unigram](https://huggingface.co/hitachi-nlp/bert-base_vaporetto-unigram) | [bert-base_vaporetto-wordpiece](https://huggingface.co/hitachi-nlp/bert-base_vaporetto-wordpiece) |
| Nothing | [bert-base_nothing-bpe](https://huggingface.co/hitachi-nlp/bert-base_nothing-bpe)  | [bert-base_nothing-unigram](https://huggingface.co/hitachi-nlp/bert-base_nothing-unigram) | [bert-base_nothing-wordpiece](https://huggingface.co/hitachi-nlp/bert-base_nothing-wordpiece) |


## Dictionary files
The trained dictionary files are available from this repository.

| | BPE | Unigram | WordPiece |  
| :-: | :-: | :-: | :-: |
| MeCab | [mecab_bpe.json](./data/dict/mecab_bpe.json) | [mecab_unigram.json](./data/dict/mecab_unigram.json) | [mecab_wordpiece.json](./data/dict/mecab_wordpiece.json) |
| Juman++ | [jumanpp_bpe.json](./data/dict/jumanpp_bpe.json) | [jumanpp_unigram.json](./data/dict/jumanpp_unigram.json) | [jumanpp_wordpiece.json](./data/dict/jumanpp_wordpiece.json) |
| Sudachi | [sudachi_bpe.json](./data/dict/sudachi_bpe.json) | [sudachi_unigram.json](./data/dict/sudachi_unigram.json) | [sudachi_wordpiece.json](./data/dict/sudachi_wordpiece.json) |
| Vaporetto | [vaporetto_bpe.json](./data/dict/vaporetto_bpe.json) | [vaporetto_unigram.json](./data/dict/vaporetto_unigram.json) | [vaporetto_wordpiece.json](./data/dict/vaporetto_wordpiece.json) |
| Nothing | [nothing_bpe.json](./data/dict/nothing_bpe.json) | [nothing_unigram.json](./data/dict/nothing_unigram.json) | [nothing_wordpiece.json](./data/dict/nothing_wordpiece.json) |


### How to load our dictionary files
Because we use the customised tokenizers, we cannot use `AutoTokenizer.from_pretrained()` to load a dictionary file.  
To load the file and construct a tokenizer, please use the following script. You must call `build_tokenizer()` to generate a tokenizer.

```python
from typing import Optional

from tokenizers import Tokenizer
from tokenizers import NormalizedString, PreTokenizedString
from tokenizers.processors import BertProcessing
from transformers import PreTrainedTokenizerFast

from pyknp import Juman
from MeCab import Tagger
from sudachipy import tokenizer
from sudachipy import dictionary
import vaporetto

import mojimoji
import traceback
import textspan


class JumanPreTokenizer:
    def __init__(self):
        self.juman = Juman("jumanpp", multithreading=True)
    
    def tokenize(self, sequence: str) -> list[str]:
        text = mojimoji.han_to_zen(sequence).rstrip()
        try:
            result = self.juman.analysis(text)
        except:
            traceback.print_exc()
            text = ""
            result = self.juman.analysis(text)
        return [mrph.midasi for mrph in result.mrph_list()]
    
    def custom_split(self, i: int, normalized_string: NormalizedString) -> list[NormalizedString]:
        text = str(normalized_string)
        tokens = self.tokenize(text)
        tokens_spans = textspan.get_original_spans(tokens, text)
        return [normalized_string[st:ed] for cahr_spans in tokens_spans for st,ed in cahr_spans]
    
    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.custom_split)


class MecabPreTokenizer:
    def __init__(self, mecab_dict_path: Optional[str] = None):
        mecab_option = (f"-Owakati -d {mecab_dict_path}" if mecab_dict_path is not None else "-Owakati")
        self.mecab = Tagger(mecab_option)
    
    def tokenize(self, sequence: str) -> list[str]:
        #return self.mecab.parse(mojimoji.han_to_zen(sequence)).strip().split(" ")
        return self.mecab.parse(sequence).strip().split(" ")
    
    def custom_split(self, i: int, normalized_string: NormalizedString) -> list[NormalizedString]:
        text = str(normalized_string)
        tokens = self.tokenize(text)
        tokens_spans = textspan.get_original_spans(tokens, text)
        return [normalized_string[st:ed] for cahr_spans in tokens_spans for st,ed in cahr_spans]
    
    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.custom_split)


class SudachiPreTokenizer:
    def __init__(self, mecab_dict_path: Optional[str] = None):
        self.sudachi = dictionary.Dictionary().create()
    
    def tokenize(self, sequence: str) -> list[str]:
        #return self.mecab.parse(mojimoji.han_to_zen(sequence)).strip().split(" ")
        return [token.surface() for token in self.sudachi.tokenize(sequence)]
    
    def custom_split(self, i: int, normalized_string: NormalizedString) -> list[NormalizedString]:
        text = str(normalized_string)
        tokens = self.tokenize(text)
        tokens_spans = textspan.get_original_spans(tokens, text)
        return [normalized_string[st:ed] for cahr_spans in tokens_spans for st,ed in cahr_spans]
    
    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.custom_split)


class VaporettoPreTokenizer:
    def __init__(self, unidic_path: str):
        with open(unidic_path, 'rb') as fp:
            model = fp.read()
        self.tokenizer = vaporetto.Vaporetto(model, predict_tags=False)
    
    def tokenize(self, sequence: str) -> list[str]:
        tokens = self.tokenizer.tokenize(sequence)
        return [token.surface() for token in tokens]
    
    def custom_split(self, i: int, normalized_string: NormalizedString) -> list[NormalizedString]:
        text = str(normalized_string)
        tokens = self.tokenize(text)
        tokens_spans = textspan.get_original_spans(tokens, text)
        return [normalized_string[st:ed] for cahr_spans in tokens_spans for st,ed in cahr_spans]
    
    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.custom_split)


def build_tokenizer(
    dict_path: str, 
    pretokenizer_type: str = None,
    vaporetto_model_path: str = None
) -> PreTrainedTokenizerFast:
    # load a tokenizer
    tokenizer = Tokenizer.from_file(dict_path)
    # load a pre-tokenizer
    if pretokenizer_type == 'mecab':
        pre_tokenizer = MecabPreTokenizer()
    elif pretokenizer_type == 'jumanpp':
        pre_tokenizer = JumanPreTokenizer()
    elif pretokenizer_type == 'vaporetto':
        pre_tokenizer = VaporettoPreTokenizer(vaporetto_model_path)
    elif pretokenizer_type == 'sudachi':
        pre_tokenizer = SudachiPreTokenizer()
    elif pretokenizer_type == 'nothing':
        pre_tokenizer = None
    else:
        raise NotImplementedError()
    tokenizer.post_processor = BertProcessing(
        cls=("[CLS]", tokenizer.token_to_id('[CLS]')),
        sep=("[SEP]", tokenizer.token_to_id('[SEP]'))
    )
    # convert to PreTrainedTokenizerFast
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token='[UNK]',
        cls_token='[CLS]',
        sep_token='[SEP]',
        pad_token='[PAD]',
        mask_token='[MASK]'
    )
    # set a pre-tokenizer
    if pre_tokenizer is not None:
        tokenizer._tokenizer.pre_tokenizer = pre_tokenizer
    return tokenizer
```


## Citation
```
@inproceedings{fujii-etal-2023-how,
      title={How does the task complexity of masked pretraining objectives affect downstream performance?}, 
      author={Takuro Fujii and Koki Shibata and Atsuki Yamaguchi and Terufumi Morishita and Yasuhiro Sogawa},
      booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics: Student Research Workshop",
      month = July,
      year = "2023",
      address = "Toronto, Canada",
      publisher = "Association for Computational Linguistics",
}
```

## License
This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a> unless specified.

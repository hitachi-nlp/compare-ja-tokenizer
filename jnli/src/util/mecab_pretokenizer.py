from typing import Optional

import textspan
from MeCab import Tagger
from tokenizers import NormalizedString, PreTokenizedString


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

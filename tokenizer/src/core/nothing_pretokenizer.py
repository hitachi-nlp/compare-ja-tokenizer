from typing import Optional

import textspan
from tokenizers import NormalizedString, PreTokenizedString


class NothingPreTokenizer:
    def __init__(self):
        self.test = 1
    
    def tokenize(self, sequence: str) -> list[str]:
        return sequence.replace(" ","").split(" ")
    
    def custom_split(self, i: int, normalized_string: NormalizedString) -> list[NormalizedString]:
        text = str(normalized_string)
        tokens = self.tokenize(text)
        tokens_spans = textspan.get_original_spans(tokens, text)
        return [normalized_string[st:ed] for cahr_spans in tokens_spans for st,ed in cahr_spans]
    
    def pre_tokenize(self, pretok: PreTokenizedString):
        pretok.split(self.custom_split)

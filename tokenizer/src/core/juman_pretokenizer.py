from typing import Optional

import textspan
import mojimoji
from pyknp import KNP, Juman
from tokenizers import NormalizedString, PreTokenizedString

import traceback
import os
os.environ["TOKENIZERS_PARALLELISM"] = "True"

class JumanPreTokenizer:
    def __init__(self):
        self.juman = Juman("jumanpp", multithreading=True)
    
    def tokenize(self, sequence: str) -> list[str]:
        text = mojimoji.han_to_zen(sequence).rstrip()
        if len(text.encode('utf-8')) <= 4096:
            result = self.juman.analysis(text)
        elif len(text.encode('utf-8')) > 4096:
            # traceback.print_exc()
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
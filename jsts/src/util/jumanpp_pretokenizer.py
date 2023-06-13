from typing import Optional

from pyknp import Juman
import mojimoji

import traceback
import textspan
from tokenizers import NormalizedString, PreTokenizedString


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

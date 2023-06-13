from typing import Optional

import textspan
import vaporetto
from tokenizers import NormalizedString, PreTokenizedString


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

import json
from pyknp import Juman
import mojimoji
from typing import Optional
import textspan
import vaporetto
from tokenizers import NormalizedString, PreTokenizedString
import traceback
import pandas as pd
from tqdm import tqdm

class JumanPreTokenizer:
    def __init__(self):
        self.juman = Juman(command='jumanpp', multithreading=True)
    
    def tokenize(self, sequence: str) -> list[str]:
        text = mojimoji.han_to_zen(sequence).rstrip()
        try:
            result = self.juman.analysis(text)
        except:
            traceback.print_exc()
            text = ""
            result = self.juman.analysis(text)
        return [mrph.midasi for mrph in result.mrph_list()]

class VaporettoPreTokenizer:
    def __init__(self, unidic_path: str):
        with open(unidic_path, 'rb') as fp:
            model = fp.read()
        self.tokenizer = vaporetto.Vaporetto(model, predict_tags=False)
    
    def tokenize(self, sequence: str) -> list[str]:
        tokens = self.tokenizer.tokenize(sequence)
        return [token.surface() for token in tokens]


train_file = "/compare-ja-tokenizer/ner/data/ner.json"
with open(train_file, "r") as fp:
    json_train_data = pd.read_json(fp)
print(json_train_data)


juman_pretokenizer = JumanPreTokenizer()
vaporetto_pretokenizer = VaporettoPreTokenizer('compare-ja-tokenizer/dict/bccwj-suw+unidic+tag.model.zst')


print(json_train_data["entities"])
"""
for i, entities in tqdm(enumerate(json_train_data["entities"])):
    for j, entity in enumerate(json_train_data["entities"][i]):
        json_train_data["entities"][i][j]["name"] = " ".join(juman_pretokenizer.tokenize(entity["name"]))
"""
pre_tokenized = []
for text in tqdm(json_train_data["text"]):
    pre_tokenized.append(" ".join(juman_pretokenizer.tokenize(text)))

json_train_data["pre tokenized text"] = pre_tokenized

json_train_data.to_json("/compare-ja-tokenizer/ner/data/ner_whitespaced_juma.json", orient="records", force_ascii=False, indent=4)
print(json_train_data)

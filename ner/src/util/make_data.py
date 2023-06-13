from datasets import load_dataset
from IPython.core.display import display
import json
import unicodedata
import random
import torch

# preprocessing
def make_data(data_path):
    dataset = json.load(open(data_path,'r'))
    type_id_dict = {
        "人名": 1,
        "法人名": 2,
        "政治的組織名": 3,
        "その他の組織名": 4,
        "地名": 5,
        "施設名": 6,
        "製品名": 7,
        "イベント名": 8
    }

    for sample in dataset:
        sample['text'] = unicodedata.normalize('NFKC', sample['text'])
        for e in sample["entities"]:
            e['type_id'] = type_id_dict[e['type']]
            del e['type']

    random.shuffle(dataset)
    n = len(dataset)
    print(n)
    n = int(n*0.9)
    train_and_val_ds = dataset[:n]
    test_ds = dataset[n:]

    return train_and_val_ds, test_ds


def create_dataset(tokenizer, dataset, max_length):
    dataset_for_loader = []
    texts_list = []
    entities_list = []
    for sample in dataset:
        text = sample['text']
        entities = sample['entities']
        # pretokenized = sample["pre tokenized text"]
        encoding = tokenizer.encode_plus_tagged(
            text, entities, max_length=max_length
        )
        # encoding = tokenizer.encode_plus_tagged(
        #   text, entities, pretokenized, max_length=max_length
        # )
        encoding = { k: torch.tensor(v) for k, v in encoding.items() }
        dataset_for_loader.append(encoding)
        texts_list.append(text)
        entities_list.append(entities)
    return dataset_for_loader, texts_list, entities_list

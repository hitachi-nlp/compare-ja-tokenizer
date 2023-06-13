from datasets import load_dataset
from IPython.core.display import display

# preprocessing
def make_data(train_data_path: str,
              val_data_path: str, 
              tokenize_function):
    # load dataset
    dataset = load_dataset('json', data_files={
        'train': train_data_path,
        'valid': val_data_path,
    }).map(lambda example: {'label': 1 if example['label'] == 'positive' else 0})
    # confirm
    display(dataset)

    # tokenize
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # confirm
    display(tokenized_datasets)

    return tokenized_datasets['train'].shuffle(seed=42), tokenized_datasets['valid'].shuffle(seed=42)


# tokenize class
class TokenizerFuncClass():
    def __init__(self, tokenizer, max_length):
        self.tokenizer = tokenizer
        self.max_length = max_length
    def tokenize_function(self, examples):
        return self.tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=self.max_length)

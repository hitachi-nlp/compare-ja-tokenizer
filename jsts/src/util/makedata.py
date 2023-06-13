from datasets import load_dataset
from IPython.core.display import display

# preprocessing
def make_data(tokenize_function, train_path, valid_path):

    # load dataset
    dataset = load_dataset('json', data_files={
        'train': train_path,
        'valid': valid_path,
    })
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
        return self.tokenizer(examples['sentence1'], examples['sentence2'], truncation=True, max_length=self.max_length)
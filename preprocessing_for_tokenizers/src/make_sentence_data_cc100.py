from datasets import load_dataset
from tqdm import tqdm

def main():
    # load dataset
    len_lines = 20000000
    cc100_dataset = load_dataset("text", 
                                data_files='../data/original/ja.txt',
                                split='train',
                                streaming=True) # -> IterableDataset
    cc100_dataset = cc100_dataset.take(len_lines)

    # split by \n
    texts = []
    for sample in tqdm(iter(cc100_dataset)):
        texts.append(sample['text'])
    
    # remove illegal sentences
    filtered = []
    for i in tqdm(range(len(texts))):
        if len(texts[i]) >= 2:
            filtered.append(texts[i])
        else:
            pass
    
    with open("../data/sentence/tokenizer_data_cc100.txt", "w") as fp:
        for sentence in filtered:
            # remove indentations
            fp.write(sentence.lstrip(' ').lstrip('^ ').lstrip('a ').lstrip('b ').lstrip('c ').lstrip('d ').lstrip('e ').lstrip('f ')+'\n')


if __name__ == '__main__':
    main()
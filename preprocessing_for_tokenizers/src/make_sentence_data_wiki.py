from datasets import load_from_disk
from tqdm import tqdm

def main():
    # load data
    wikipedia_dataset = load_from_disk("../data/original/wikipedia_ja")
    wikipedia_dataset = wikipedia_dataset.shuffle(seed=42)
    wikipedia_dataset = wikipedia_dataset.remove_columns('title')

    # split by \n
    texts = ""
    for i in tqdm(range(int(wikipedia_dataset['train'].num_rows))):
        texts += wikipedia_dataset['train'][i]['text']
    texts = texts.split('\n')

    # remove sentences which contain less than 30 chars and no table
    filtered = []
    for i in tqdm(range(len(texts))):
        if len(texts[i]) >= 30 and 'Category' not in texts[i] and '||' not in texts[i]:
            filtered.append(texts[i])
        else:
            pass

    texts = "".join(filtered)
    sentences = texts.split('。')
    with open("../data/sentence/tokenizer_data_wiki.txt", "w") as fp:
        for sentence in sentences:
            if len(sentence) >= 30:
                # remove indents
                fp.write(sentence.lstrip(' ')+'。\n')

if __name__ == '__main__':
    main()
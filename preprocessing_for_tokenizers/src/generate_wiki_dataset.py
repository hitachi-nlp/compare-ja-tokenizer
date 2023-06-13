from datasets import load_dataset

def main():
    # convert tensorflow datasets into huggingface datasets
    wiki_dataset = load_dataset(
        "./src/wikipedia_ja.py"
    )
    # save to the local disk
    wiki_dataset.save_to_disk('./data/original/wikipedia_ja')

if __name__ == '__main__':
    main()
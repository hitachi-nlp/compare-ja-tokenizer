from datasets import concatenate_datasets, load_dataset
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import WhitespaceSplit
from tokenizers.processors import BertProcessing
from transformers import PreTrainedTokenizerFast


def main(args):
    # load dataset
    wikipedia_dataset = load_dataset(
        "text", 
        data_files=args.wikipedia_data_path,
        split='train'
    ) # -> Dataset
    cc100_dataset = load_dataset(
        "text", 
        data_files=args.cc100_data_path,
        split='train'
    ) # -> Dataset

    # concat datasets
    dataset = concatenate_datasets([wikipedia_dataset, cc100_dataset])
    dataset = dataset.shuffle(seed=42)
    
    # load tokenizer
    pre_tokenizer = WhitespaceSplit()
    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    tokenizer.post_processor = BertProcessing(
        cls=("[CLS]", tokenizer.token_to_id('[CLS]')),
        sep=("[SEP]", tokenizer.token_to_id('[SEP]'))
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token='[UNK]',
        cls_token='[CLS]',
        sep_token = '[SEP]',
        pad_token='[PAD]',
        mask_token='[MASK]'
    )
    tokenizer._tokenizer.pre_tokenizer = pre_tokenizer
    
    # tokenize samples
    dataset = dataset.map(
        lambda samples: tokenizer(samples["text"], 
                                  max_length=512, 
                                  padding="max_length", 
                                  truncation=True), 
        num_proc=10, batched=True
    )
    
    # save to disk
    dataset.save_to_disk(args.output_dir)


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description="Preprocessing samples using a pre-trained tokenizer with MeCab."
    )
    parser.add_argument(
        "--wikipedia_data_path", 
        help="(str) Path to a wikipedia text data path.", 
        required=True
    )
    parser.add_argument(
        "--cc100_data_path", 
        help="(str) Path to a cc100 text data path.", 
        required=True
    )
    parser.add_argument(
        "--tokenizer_path", 
        help="(str) Path to a tokenizer data path.", 
        required=True
    )
    parser.add_argument(
        "--output_dir", 
        help="(str) Path to a output data directory.", 
        required=True
    )
    args = parser.parse_args()
    main(args)

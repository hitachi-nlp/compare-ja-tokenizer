from tokenizers import Tokenizer
from tokenizers.models import BPE, Unigram, WordPiece
from tokenizers.pre_tokenizers import BertPreTokenizer, PreTokenizer, WhitespaceSplit
from tokenizers.trainers import BpeTrainer, UnigramTrainer, WordPieceTrainer

from core import SudachiPreTokenizer

special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]


def wordpiece(args):
    """Train a WordPiece-based tokenizer with Sudachi."""
    # init
    tokenizer = Tokenizer(WordPiece())
    # apply pretokenizer
    tokenizer.pre_tokenizer = WhitespaceSplit()
    # set up a trainer
    trainer = WordPieceTrainer(
        special_tokens=special_tokens, 
        vocab_size=args.vocab_size
    )
    # train
    tokenizer.train(
        files=[args.data_path], 
        trainer=trainer
    )
    # save
    # see https://github.com/huggingface/tokenizers/issues/613 
    # and https://tech.mntsq.co.jp/entry/2021/02/26/120013 for why we replace the pretokenizer.
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.save(args.output_path)
    return


def unigram(args):
    """Train a Unigram-based tokenizer with Sudachi."""
    # init
    tokenizer = Tokenizer(Unigram())
    # apply pretokenizer
    tokenizer.pre_tokenizer = WhitespaceSplit()
    # set up a trainer
    trainer = UnigramTrainer(
        special_tokens=special_tokens, 
        vocab_size=args.vocab_size,
        unk_token= '[UNK]'
    )
    # train
    tokenizer.train(
        files=[args.data_path], 
        trainer=trainer
    )
    # save
    # see https://github.com/huggingface/tokenizers/issues/613 
    # and https://tech.mntsq.co.jp/entry/2021/02/26/120013 for why we replace the pretokenizer.
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.save(args.output_path)
    return


def bpe(args):
    """Train a BPE-based tokenizer with Sudachi."""
    # init
    tokenizer = Tokenizer(BPE())
    # apply pretokenizer
    tokenizer.pre_tokenizer = WhitespaceSplit()
    # set up a trainer
    trainer = BpeTrainer(
        special_tokens=special_tokens, 
        vocab_size=args.vocab_size
    )
    # train
    tokenizer.train(
        files=[args.data_path], 
        trainer=trainer
    )

    # save
    # see https://github.com/huggingface/tokenizers/issues/613 
    # and https://tech.mntsq.co.jp/entry/2021/02/26/120013 for why we replace the pretokenizer.
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.save(args.output_path)
    return


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description="Training a tokenizer with Sudachi."
    )
    parser.add_argument(
        "--data_path", 
        help="(str) Path to a training text data path.", 
        required=True
    )
    parser.add_argument(
        "--output_path", 
        help="(str) Path to a json output path.", 
        required=True
    )
    parser.add_argument(
        "--subword_method", 
        choices=["wordpiece", "unigram", "bpe"],
        help="(str) Type of a subword tokenization method.", 
        required=True
    )
    parser.add_argument(
        "--vocab_size", 
        help="(int) The size of a vocabulary.", 
        default=30000,
        type=int
    )
    parser.add_argument(
        "--local_rank", 
        type=int, 
        default=0
    )
    args = parser.parse_args()
    
    if args.subword_method == 'wordpiece':
        wordpiece(args)
    elif args.subword_method == 'unigram':
        unigram(args)
    elif args.subword_method == 'bpe':
        bpe(args)
    else:
        raise NotImplementedError()

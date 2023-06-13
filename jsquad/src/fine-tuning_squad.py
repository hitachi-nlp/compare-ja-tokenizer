import os

from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import PreTokenizer, WhitespaceSplit
from tokenizers.processors import BertProcessing
from util import (AdditionalArguments, JumanPreTokenizer, MecabPreTokenizer,
                  SudachiPreTokenizer, TokenizerFuncClass,
                  VaporettoPreTokenizer, cross_validation, make_data)

from transformers import (AutoTokenizer, HfArgumentParser,
                          PreTrainedTokenizerFast, TrainingArguments, set_seed)
from transformers.data.data_collator import DataCollatorWithPadding


def main(args, training_args):
    # init
    set_seed(training_args.seed)
    if args.pretokenizer_type =='cl-tohoku/bert-base-japanese':
        tokenizer = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')

    else:
        # set up a tokenizer
        if args.pretokenizer_type == 'mecab':
            pre_tokenizer = MecabPreTokenizer()
        elif args.pretokenizer_type == 'jumanpp':
            pre_tokenizer = JumanPreTokenizer()
        elif args.pretokenizer_type == 'vaporetto':
            pre_tokenizer = VaporettoPreTokenizer(args.vaporetto_model_path)
        elif args.pretokenizer_type == 'sudachi':
            pre_tokenizer = SudachiPreTokenizer()
        elif args.pretokenizer_type == 'nothing':
            pre_tokenizer = None
        else:
            raise NotImplementedError()
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        tokenizer = Tokenizer.from_file(args.tokenizer_path)
        tokenizer.post_processor = BertProcessing(
            cls=("[CLS]", tokenizer.token_to_id('[CLS]')),
            sep=("[SEP]", tokenizer.token_to_id('[SEP]'))
        )
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            unk_token='[UNK]',
            cls_token='[CLS]',
            sep_token='[SEP]',
            pad_token='[PAD]',
            mask_token='[MASK]'
        )
        if pre_tokenizer is not None:
            tokenizer._tokenizer.pre_tokenizer = PreTokenizer.custom(pre_tokenizer)

    # prepare a dataset
    global tokenize_func
    tokenize_func = TokenizerFuncClass(tokenizer, args.max_seq_length)
    train_and_valid_ds, test_ds = make_data(tokenize_func, args)

    # data collator
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # train
    cross_validation(
        tokenize_func=tokenize_func,
        train_and_valid_ds = train_and_valid_ds,
        test_ds = test_ds,
        data_collator = data_collator,
        args=args,
        training_args=training_args
    )

if __name__ == '__main__':
    parser = HfArgumentParser((AdditionalArguments, TrainingArguments))
    args, training_args = parser.parse_args_into_dataclasses()
    
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info(f'Training args: {training_args}')
    logger.info(f'Additional self-defined args: {args}')
    
    main(args, training_args)
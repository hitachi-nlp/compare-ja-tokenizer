# module import
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import PreTokenizer, WhitespaceSplit
from tokenizers.processors import BertProcessing
from transformers import (BertConfig, 
                          BertForMaskedLM,
                          DataCollatorForLanguageModeling, 
                          HfArgumentParser,
                          PreTrainedTokenizerFast,
                          Trainer, TrainingArguments, 
                          set_seed)
from datasets import load_from_disk

from util import AdditionalArguments


def main(args, training_args):
    # init
    set_seed(training_args.seed)
    
    # set up a tokenizer
    if args.pretokenizer_type in ['mecab', 'sudachi', 'juman', 'vaporetto']:
        pre_tokenizer = WhitespaceSplit()
    elif args.pretokenizer_type == 'nothing':
        pre_tokenizer = None
    else:
        raise NotImplementedError()
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
        tokenizer._tokenizer.pre_tokenizer = pre_tokenizer
    
    # prepare a dataset
    dataset = load_from_disk(args.data_path)
    
    # set up a data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=0.15
    )
    
    # build a model
    # bert base
    # https://huggingface.co/cl-tohoku/bert-base-japanese/blob/main/config.json
    config = BertConfig(
        attention_probs_dropout_prob=0.1,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        hidden_size=768,
        initializer_range=0.02,
        intermediate_size=3072,
        layer_norm_eps=1e-12,
        max_position_embeddings=512,
        num_attention_heads=12,
        num_hidden_layers=12,
        vocab_size=args.vocab_size
    )
    model = BertForMaskedLM(config)
    
    # train
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset
    )
    trainer.train()
    trainer.save_model(training_args.output_dir)


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

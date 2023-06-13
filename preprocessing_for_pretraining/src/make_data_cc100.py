from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import PreTokenizer

from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.processors import BertProcessing
from transformers import PreTrainedTokenizerFast
from core import MecabPreTokenizer

from tqdm import tqdm


def main(args):
    # load tokenizer
    tokenizer = load_tokenizer(args)
    # load tokenizer data
    sentences = open('../data/sentence/tokenizer_data_cc100.txt', 'r').readlines()
    # get tokens per sentence
    sentence_token = {'sentence': [], 'num token': []}
    for sentence in tqdm(sentences):
        sentence = sentence.replace("\n",'')
        tokenized = tokenizer.tokenize(sentence)
        sentence_token['sentence'].append(sentence)
        sentence_token['num token'].append(len(tokenized))
    """
    sentence_token = {
        sentence: [[吾輩は猫である。], [名前はまだない。], ...],
        num token: [6 ,5, ...]
        }
    """
    # merge sentences to nearly 512
    num_tokens = sentence_token['num token']
    start_pos = {'start':[], 'end': [], 'len': []}
    start = 0
    cnt = 0
    max_length = 512 - 2    # [CLS] and [SEP] are considered
    for i in tqdm(range(len(num_tokens))):
        if cnt >= max_length:
            start_pos['start'].append(start)
            start_pos['end'].append(i-2)
            # length of one sentence is more than 512
            if cnt-(num_tokens[i-1]) == 0:
                start_pos['len'].append(len(sentence_token['sentence'][i-1]))
            # length of merged sentences is more than 512
            else:
                start_pos['len'].append(cnt-num_tokens[i-1])
            start = i-1
            cnt = num_tokens[i-1] + num_tokens[i]
        # length of merged sentences is less than 512
        else:
            cnt += num_tokens[i]
    """
    start_pos = {
        start: [0, 9, 14, 26, ...],
        end: [10, 15, 27, ...],
        len: [480, 499, 502, ...]
    }
    """
    # make pretrain data by merging sentences
    sentences_for_pretrain = []
    for start_idx, end_idx in tqdm(zip(start_pos['start'], start_pos['end'])):
        sentence = ''
        for j in range(start_idx, end_idx+1):
            sentence += sentence_token['sentence'][j]
        sentences_for_pretrain.append(sentence)
    with open(f"../data/pretraining/pretrain_sentence_cc100.txt", "w") as fp:
        for sentences in sentences_for_pretrain:
            fp.write(sentences+'\n')


def load_tokenizer(args):
    # set up a tokenizer
    pre_tokenizer = None
    if args.pretokenizer_type == 'mecab':
        pre_tokenizer = MecabPreTokenizer()
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
        sep_token = '[SEP]',
        pad_token='[PAD]',
        mask_token='[MASK]'
    )
    if pre_tokenizer is not None:
        tokenizer._tokenizer.pre_tokenizer = PreTokenizer.custom(pre_tokenizer)

    return tokenizer
        

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(
        description="Generating pretraining data with MeCab."
    )
    parser.add_argument(
        "--tokenizer_path", 
        help="(str) Path to a json tokenizer path.", 
        required=False
    )
    parser.add_argument(
        "--pretokenizer_type", 
        help="(str) Type of a tokenization method.", 
        required=True
    )
    args = parser.parse_args()

    main(args)
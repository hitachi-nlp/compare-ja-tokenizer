#!/usr/bin/env python
# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for multiple choice.
"""
# You can also adapt this script on your own multiple choice task. Pointers for this are left as comments.

import logging
import os
from dataclasses import dataclass, field
from itertools import chain
from typing import Optional, Union
from sklearn.model_selection import KFold

import pandas as pd
import numpy as np
import torch
from datasets import load_dataset
from IPython.core.display import display

import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForMultipleChoice,
    PreTrainedTokenizerFast,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.utils import PaddingStrategy, check_min_version

from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import PreTokenizer
from tokenizers.processors import BertProcessing

import shutil
from util import MecabPreTokenizer, JumanPreTokenizer, VaporettoPreTokenizer, SudachiPreTokenizer
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#os.environ["TRANSFORMERS_OFFLINE"] == "1"
# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.25.0.dev0")

logger = logging.getLogger(__name__)

@dataclass
class AdditionalArguments:
    """Define additional arguments that are not included in `TrainingArguments`."""
    tokenizer_path: Optional[str] = field(
        default="",
        metadata={"help": "A tokenizer path."}
    )
    
    pretokenizer_type: Optional[str] = field(
        default="",
        metadata={"help": ("A type of a pre-tokenizer. Choose one from "
                  "mecab, juman, sudachi, vaporetto, nothing, and cl-tohoku/bert-base-japanese.")}
    )

    train_data_path: Optional[str] = field(
        default="",
        metadata={"help": "A path to a target task training data."}
    )

    val_data_path: Optional[str] = field(
        default="",
        metadata={"help": "A path to a target task validation data."}
    )

    vaporetto_model_path: Optional[str] = field(
        default="",
        metadata={"help": "A tokenizer model path for vaporetto."}
    )
    

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a text file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    max_seq_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. If passed, sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to the maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )

    def __post_init__(self):
        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."


@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature.pop(label_name) for feature in features]
        batch_size = len(features)
        num_choices = len(features[0]["input_ids"])
        flattened_features = [
            [{k: v[i] for k, v in feature.items()} for i in range(num_choices)] for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # Un-flatten
        batch = {k: v.view(batch_size, num_choices, -1) for k, v in batch.items()}
        # Add back labels
        batch["labels"] = torch.tensor(labels, dtype=torch.int64)
        return batch


# preprocessing
def make_data(tokenize_function, train_data_path, val_data_path):

    # load dataset
    dataset = load_dataset('json', data_files={
        'train': train_data_path,
        'valid': val_data_path,
    })
    # confirm
    display(dataset)

    # tokenize
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    # confirm
    display(tokenized_datasets)

    return tokenized_datasets['train'].shuffle(seed=42), tokenized_datasets['valid'].shuffle(seed=42)


# Preprocessing the datasets.
class TokenizerFuncClass:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # When using your own dataset or a different dataset from swag, you will probably need to change this.
        self.ending_names = [f"choice{i}" for i in range(5)]
        self.context_name = "question"
        self.max_seq_length = 64

    def preprocess_function(self, examples):
        first_sentences = [[context] * 5 for context in examples[self.context_name]]
        second_sentences = [
            [f"{examples[end][i]}" for end in self.ending_names] for i in range(len(examples[self.context_name]))
        ]

        # Flatten out
        first_sentences = list(chain(*first_sentences))
        second_sentences = list(chain(*second_sentences))

        # Tokenize
        tokenized_examples = self.tokenizer(
            first_sentences,
            second_sentences,
            truncation=True,
            max_length=self.max_seq_length,
            padding="max_length",
        )
        print(tokenized_examples.keys())
        # Un-flatten
        return {k: [v[i : i + 5] for i in range(0, len(v), 5)] for k, v in tokenized_examples.items()}


def cross_validation(train_and_valid_ds, test_ds, data_collator, model_args, training_args, compute_metrics):
    test_accuracy = []
    kf = KFold(n_splits=5, shuffle=True, random_state=training_args.seed)
    num=0
    for train_idx, valid_idx in kf.split(train_and_valid_ds):
        num+=1
        model = AutoModelForMultipleChoice.from_pretrained(model_args.model_name_or_path)

        # split datasets into fold
        train_fold = train_and_valid_ds.select(train_idx)
        valid_fold = train_and_valid_ds.select(valid_idx)
        
        # train
        training_args.output_dir = training_args.output_dir.rstrip(str(num-1))
        training_args.output_dir = training_args.output_dir + str(num)
        
        # Initialize our Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_fold,
            eval_dataset=valid_fold,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )
        trainer.train()
        trainer.save_model(training_args.output_dir + '/best_model.bin')

        # error analisys
        outputs = trainer.predict(valid_fold)
        preds_df = pd.DataFrame({
            'pred': outputs.predictions.argmax(axis=-1),
            'label': outputs.label_ids
        })
        preds_df.to_csv(training_args.output_dir + '/result.csv')
        
        # make all fold validation table
        evaluation = trainer.evaluate(test_ds)
        test_accuracy.append([str(num)+'th', evaluation['eval_accuracy']*100])
    
    training_args.output_dir = training_args.output_dir.replace('/cross_validation_5', '')
    # add all fold validation table to average ± std
    columns = ['CV', 'test_acc']
    test_accuracy_np = np.array([x[1] for x in test_accuracy])
    test_accuracy.append(['final result', str(sum(test_accuracy_np)/len(test_accuracy_np))+'±'+str(np.std(test_accuracy_np))])
    df = pd.DataFrame(data=test_accuracy, columns=columns).set_index('CV')
    df.to_csv(training_args.output_dir + '/evaluation.csv')
    display(df)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    parser = HfArgumentParser((AdditionalArguments, ModelArguments, DataTrainingArguments, TrainingArguments))
    args, model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    transformers.logging.set_verbosity_error()
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).

    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    # set up a tokenizer
    if args.pretokenizer_type =='cl-tohoku/bert-base-japanese':
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

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
    
    #tokenizer = AutoTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
    tokenizerfuc = TokenizerFuncClass(tokenizer)

    train_dataset, eval_dataset = make_data(tokenizerfuc.preprocess_function, args.train_data_path, args.val_data_path)

    # Data collator
    data_collator = (
        default_data_collator
        if data_args.pad_to_max_length
        else DataCollatorForMultipleChoice(tokenizer=tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None)
    )

    # Metric
    def compute_metrics(eval_predictions):
        predictions, label_ids = eval_predictions
        preds = np.argmax(predictions, axis=1)
        return {"accuracy": (preds == label_ids).astype(np.float32).mean().item()}

    cross_validation(
        train_and_valid_ds=train_dataset,
        test_ds=eval_dataset,
        data_collator=data_collator,
        training_args=training_args,
        model_args=model_args,
        compute_metrics=compute_metrics
    )


if __name__ == "__main__":
    main()
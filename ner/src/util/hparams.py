from dataclasses import dataclass, field
from typing import Optional

@dataclass
class AdditionalArguments:
    """Define additional arguments that are not included in `TrainingArguments`."""
    tokenizer_path: Optional[str] = field(
        default="",
        metadata={"help": "A tokenizer path."}
    )

    vaporetto_model_path: Optional[str] = field(
        default="",
        metadata={"help": "A tokenizer model path for vaporetto."}
    )

    pretrain_path: Optional[str] = field(
        default="",
        metadata={"help": "Pretrain weights path"}
    )
    
    pretokenizer_type: Optional[str] = field(
        default="",
        metadata={"help": ("A type of a pre-tokenizer. Choose one from "
                  "mecab, juman, sudachi, vaporetto, nothing, and cl-tohoku/bert-base-japanese.")}
    )
    
    save_interval: Optional[float] = field(
        default=43200.0,
        metadata={"help": "An interval of saving weights in seconds."}
    )

    data_path: Optional[str] = field(
        default="",
        metadata={"help": "A path to a target task training data."}
    )
    
    max_length: Optional[int] = field(
        default=512,
        metadata={"help": "The maximum number of tokens per sample."}
    )

    vocab_size: Optional[int] = field(
        default=30000,
        metadata={"help": "The size of a vocabulary."}
    )

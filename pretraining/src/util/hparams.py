from dataclasses import dataclass, field
from typing import Optional


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
                  "mecab, juman, vaporetto, sudachi, and nothing.")}
    )
    
    data_path: Optional[str] = field(
        default="",
        metadata={"help": "Path to a pretraining data."}
    )

    vocab_size: Optional[int] = field(
        default=30000,
        metadata={"help": "The size of a vocabulary."}
    )

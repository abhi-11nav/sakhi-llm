import json
import logging
import os

import sentencepiece as spm
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)



class DataIterator:
    def __init__(self, jsonl_path, text_key="text", eos_token="<|eos|>"):
        self.jsonl_path = jsonl_path
        self.text_key = text_key
        self.eos_token = eos_token

    def write_to(self, output_path):
        with (
            open(self.jsonl_path, "r", encoding="utf-8") as f_in,
            open(output_path, "w", encoding="utf-8") as f_out,
        ):
            for line in f_in:
                try:
                    data = json.loads(line)
                    text = data.get(self.text_key, "").strip()
                    if text:
                        f_out.write(f"{text} {self.eos_token}\n")
                except json.JSONDecodeError:
                    continue


def train_sentencepiece_tokenizer(
    corpus_file: str,
    save_dir: str,
    vocab_size: int = 32000,
    character_coverage: float = 1.0,
    model_type: str = "bpe",  # could also be "unigram", "char", or "word"
    eos_token: str = "<|eos|>",
) -> str:
    os.makedirs(save_dir, exist_ok=True)
    raw_text_path = os.path.join(save_dir, "training.txt")

    logger.info("Extracting training data from JSONL...")
    DataIterator(corpus_file, eos_token=eos_token).write_to(raw_text_path)

    model_prefix = os.path.join(save_dir, "spm")
    logger.info("Training SentencePiece tokenizer...")
    spm.SentencePieceTrainer.train(
        input=raw_text_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        model_type=model_type,
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        user_defined_symbols=[eos_token],
    )

    logger.info("Loading and wrapping with PreTrainedTokenizerFast...")
    sp_model_path = model_prefix + ".model"
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=None,
        tokenizer_object=None,
        sp_model_file=sp_model_path,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<s>",
        eos_token=eos_token,
    )

    tokenizer.save_pretrained(save_dir)
    logger.info(f"Saved SentencePiece tokenizer to: {save_dir}")
    return save_dir

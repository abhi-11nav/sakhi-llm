import argparse
import json
import os

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from tokenizers.trainers import BpeTrainer
from transformers import PreTrainedTokenizerFast


class JsonlTextIterator:
    def __init__(self, jsonl_path, text_key="text", eos_token="<|eos|>"):
        self.jsonl_path = jsonl_path
        self.text_key = text_key
        self.eos_token = eos_token

    def __iter__(self):
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    text = data.get(self.text_key, "").strip()
                    if text:
                        yield text + " " + self.eos_token
                except json.JSONDecodeError:
                    continue


def train_tokenizer(
    corpus_file: str, save_dir: str, vocab_size: int, min_frequency: int
):
    os.makedirs(save_dir, exist_ok=True)

    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=True,
        special_tokens=[
            "[UNK]",
            "<|pad|>",
            "<|eos|>",
        ],
    )

    print(f"Training tokenizer on: {corpus_file} (streaming mode)")
    iterator = JsonlTextIterator(corpus_file)
    tokenizer.train_from_iterator(iterator, trainer=trainer)

    tokenizer_json_path = os.path.join(save_dir, "tokenizer.json")
    tokenizer.save(tokenizer_json_path)
    print(f"Saved raw tokenizer to: {tokenizer_json_path}")

    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_json_path)
    hf_tokenizer.add_special_tokens(
        {
            "pad_token": "<|pad|>",
            "eos_token": "<|eos|>",
            "unk_token": "[UNK]",
        }
    )

    hf_tokenizer.save_pretrained(save_dir)
    print(f"Saved HuggingFace-compatible tokenizer to: {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on large Telugu JSONL corpus using streaming"
    )
    parser.add_argument(
        "--corpus", type=str, required=True, help="Path to input .jsonl file"
    )
    parser.add_argument(
        "--save_dir", type=str, default="soki_tokenizer", help="Save directory"
    )
    parser.add_argument("--vocab_size", type=int, default=32000, help="Vocabulary size")
    parser.add_argument(
        "--min_frequency", type=int, default=2, help="Minimum frequency"
    )

    args = parser.parse_args()

    train_tokenizer(
        corpus_file=args.corpus,
        save_dir=args.save_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

import argparse
import json
import logging

from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def analyze_tokenizer(tokenizer_path: str, jsonl_path: str, max_lines: int = 10000):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    total_tokens = 0
    total_unk_tokens = 0
    total_sequences = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= max_lines:
                break
            try:
                data = json.loads(line)
                text = data.get("text", "").strip()
                if not text:
                    continue

                tokens = tokenizer.encode(text)
                total_tokens += len(tokens)
                total_sequences += 1

                unk_id = tokenizer.unk_token_id
                total_unk_tokens += tokens.count(unk_id)

            except json.JSONDecodeError:
                continue

    avg_length = total_tokens / total_sequences if total_sequences > 0 else 0
    unk_rate = (total_unk_tokens / total_tokens) * 100 if total_tokens > 0 else 0

    logger.info(f"Analyzed {total_sequences} sequences.")
    logger.info(f"Average sequence length: {avg_length:.2f} tokens")
    logger.info(f"[UNK] token rate: {unk_rate:.4f}%")

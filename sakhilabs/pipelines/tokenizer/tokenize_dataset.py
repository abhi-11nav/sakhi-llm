import json
import logging
from collections import Counter

from tqdm import tqdm
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)


def chunk_large_txt_stream(
    txt_path: str,
    tokenizer_path: str,
    output_path: str,
    chunk_length: int,
    read_block_size: int = 16384,
):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    tokenizer.add_special_tokens(
        {
            "pad_token": "<|pad|>",
            "eos_token": "<|eos|>",
            "unk_token": "[UNK]",
        }
    )

    token_buffer = []
    written = 0

    pbar = tqdm(desc="Processing", unit="chunk")

    with (
        open(txt_path, "r", encoding="utf-8") as infile,
        open(output_path, "w", encoding="utf-8") as outfile,
    ):
        while True:
            chunk = infile.read(read_block_size)
            if not chunk:
                break

            token_buffer.extend(tokenizer.encode(chunk))

            while len(token_buffer) >= chunk_length + 1:
                current_chunk = token_buffer[: chunk_length + 1]
                token_buffer = token_buffer[chunk_length:]

                json.dump(
                    {"input_ids": current_chunk[:-1], "labels": current_chunk[1:]},
                    outfile,
                )
                outfile.write("\n")
                written += 1
                pbar.update(1)
    pbar.close()
    logger.info(f"\nCompleted. Total chunks written: {written}")


def analyze_tokenized_dataset(jsonl_path, top_k=20):
    total_samples = 0
    total_tokens = 0
    token_freq = Counter()
    lengths = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Analyzing"):
            sample = json.loads(line)
            tokens = sample["input_ids"]
            token_freq.update(tokens)
            token_len = len(tokens)

            total_tokens += token_len
            lengths.append(token_len)
            total_samples += 1

    average_tokens = total_tokens / total_samples if total_samples else 0
    unique_tokens = len(token_freq)
    top_tokens = token_freq.most_common(top_k)
    min_len = min(lengths) if lengths else 0
    max_len = max(lengths) if lengths else 0

    logger.info(f"Total samples:         {total_samples:,}")
    logger.info(f"Total tokens:          {total_tokens:,}")
    logger.info(f"Average tokens/sample: {average_tokens:.2f}")
    logger.info(f"Unique tokens:         {unique_tokens:,}")
    logger.info(f"Min tokens/sample:     {min_len}")
    logger.info(f"Max tokens/sample:     {max_len}")
    logger.info(f"\n Top {top_k} most frequent tokens:")
    for token, freq in top_tokens:
        logger.info(f"Token ID {token}: {freq:,} times")

    return {
        "total_samples": total_samples,
        "total_tokens": total_tokens,
        "average_tokens_per_sample": average_tokens,
        "unique_tokens": unique_tokens,
        "top_tokens": top_tokens,
        "lengths": lengths,
    }


if __name__ == "__main__":
    chunk_large_txt_stream(
        txt_path="local-data/data/combined_text_with_eos.txt",
        tokenizer_path="local-data/tokenizers/tokenizer_64k/tokenizer.json",
        output_path="local-data/data/tokenized_chunks.jsonl",
        chunk_length=1024,
        read_block_size=16384,
    )

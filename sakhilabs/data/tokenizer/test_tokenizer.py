import argparse
import json

from transformers import PreTrainedTokenizerFast


def analyze_tokenizer(tokenizer_path, jsonl_path, max_lines=10000):
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

                # Count [UNK] token IDs
                unk_id = tokenizer.unk_token_id
                total_unk_tokens += tokens.count(unk_id)

            except json.JSONDecodeError:
                continue

    avg_length = total_tokens / total_sequences if total_sequences > 0 else 0
    unk_rate = (total_unk_tokens / total_tokens) * 100 if total_tokens > 0 else 0

    print(f"Analyzed {total_sequences} sequences.")
    print(f"Average sequence length: {avg_length:.2f} tokens")
    print(f"[UNK] token rate: {unk_rate:.4f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze tokenizer performance on a JSONL file"
    )
    parser.add_argument(
        "--tokenizer", required=True, help="Path to tokenizer directory"
    )
    parser.add_argument("--jsonl", required=True, help="Path to JSONL file")
    parser.add_argument(
        "--max_lines", type=int, default=100000, help="Number of lines to sample"
    )

    args = parser.parse_args()

    analyze_tokenizer(args.tokenizer, args.jsonl, args.max_lines)

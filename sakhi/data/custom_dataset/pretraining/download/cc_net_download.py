import json
import os

from datasets import load_dataset

dataset = load_dataset("mc4", "te", split="train")

_dir = "YOUR CUSTOM PATH"
output_file = os.path.join(_dir, "telugu_dataset.jsonl")

with open(output_file, "w", encoding="utf-8") as f_out:
    for item in dataset:
        json.dump({"text": item["text"]}, f_out, ensure_ascii=False)
        f_out.write("\n")

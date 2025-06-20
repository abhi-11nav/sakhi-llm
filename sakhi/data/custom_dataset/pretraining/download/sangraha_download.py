import json
import os

from datasets import load_dataset
from tqdm import tqdm

configs = ["synthetic"]  # "unverified" # "verified" -> commented ones are completed

_dir = "YOUR CUSTOM PATH"
output_file = os.path.join(_dir, "ai4bharat.jsonl")

with open(output_file, "a", encoding="utf-8") as f_out:
    for config in configs:
        dataset = load_dataset(
            "ai4bharat/sangraha", config, split="tel", streaming=True
        )

        for item in tqdm(dataset, desc=f"Processing {config}"):
            json.dump(
                {"text": item["text"], "config": config}, f_out, ensure_ascii=False
            )
            f_out.write("\n")

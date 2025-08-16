import json
from typing import List

import torch
from torch.utils.data import Dataset


class SakhiPreTrainDataset(Dataset):
    def __init__(self, dataset_json: str, chunk_length: int, tokenizer):
        self.chunk_length = chunk_length
        self.tokenizer = tokenizer

        with open(dataset_json, "r") as f:
            self.samples = json.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        item = self.samples[index]
        text = item["text"]

        tokens = self.tokenizer.encode(text, add_special_tokens=False)

        if len(tokens) > self.chunk_length:
            tokens = tokens[: self.chunk_length]
        else:
            pad_id = self.tokenizer.pad_token_id
            tokens = tokens + [pad_id] * (self.chunk_length - len(tokens))

        input_ids = torch.tensor(tokens, dtype=torch.long)
        labels = torch.tensor(
            tokens[1:] + [self.tokenizer.pad_token_id], dtype=torch.long
        )

        return {
            "input_ids": input_ids,
            "labels": labels,
        }

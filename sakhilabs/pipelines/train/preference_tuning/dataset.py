import json

import torch
from torch.utils.data import Dataset


def prepare_instruct_prompt(prompt: str):
    prefix = "<|instruction|>"
    response_tag = "<|response|>"

    instruct_prompt = f"{prefix} {prompt} {response_tag} "
    return instruct_prompt


class DPODataset(Dataset):
    def __init__(self, dataset_file: str, tokenizer, max_length: int = 1024):
        with open(dataset_file, "r") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]
        prompt = prepare_instruct_prompt(entry["instruction"])
        pos_resp = entry["response"]["positive"]
        neg_resp = entry["response"]["negative"]

        prompt_tokens = self.tokenizer(
            prompt, return_tensors="pt", truncation=True, max_length=self.max_length
        ).input_ids[0]
        pos_tokens = self.tokenizer(
            pos_resp, return_tensors="pt", truncation=True, max_length=self.max_length
        ).input_ids[0]
        neg_tokens = self.tokenizer(
            neg_resp, return_tensors="pt", truncation=True, max_length=self.max_length
        ).input_ids[0]

        return {
            "prompt_tokens": prompt_tokens,
            "pos_tokens": pos_tokens,
            "neg_tokens": neg_tokens,
        }


def dpo_collate_fn(batch):
    prompt_batch = [item["prompt_tokens"] for item in batch]
    pos_batch = [item["pos_tokens"] for item in batch]
    neg_batch = [item["neg_tokens"] for item in batch]

    prompt_padded = torch.nn.utils.rnn.pad_sequence(
        prompt_batch, batch_first=True, padding_value=0
    )
    pos_padded = torch.nn.utils.rnn.pad_sequence(
        pos_batch, batch_first=True, padding_value=0
    )
    neg_padded = torch.nn.utils.rnn.pad_sequence(
        neg_batch, batch_first=True, padding_value=0
    )

    return {
        "prompt_tokens": prompt_padded,
        "pos_tokens": pos_padded,
        "neg_tokens": neg_padded,
    }

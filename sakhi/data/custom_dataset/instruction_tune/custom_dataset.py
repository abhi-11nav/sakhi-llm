import json
import re

import torch
from torch.utils.data import (DataLoader, Dataset, DistributedSampler, Subset,
                              random_split)


class InstructionDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_length=1024):
        with open(path, "r") as f:
            self.data = json.load(f)

        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        entry = self.data[idx]

        prefix = "<|instruction|>"
        response_tag = "<|response|>"
        eos_token = self.tokenizer.eos_token

        # Combine instruction + input
        if entry["telugu_input"] == entry["telugu_input"]:
            prompt = f"{prefix} {entry['telugu_instruction']} {entry['telugu_input']} "
        else:
            prompt = f"{prefix} {entry['telugu_instruction']} "

        response = f"{response_tag} {entry['telugu_output']} {eos_token}"
        full_text = prompt + response

        full_text = full_text.strip()
        full_text = re.sub(r"\s+", " ", full_text)

        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
            padding="max_length",
        )

        input_ids = tokenized["input_ids"].squeeze(0)
        attention_mask = tokenized["attention_mask"].squeeze(0)

        labels = input_ids.clone()

        prompt_len = len(self.tokenizer(prompt)["input_ids"])
        response_tag_len = len(self.tokenizer(f"{response_tag} ")["input_ids"])

        # Label masking (masking prompt and padding tokens)
        labels[: prompt_len + response_tag_len] = -100
        labels[input_ids == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def get_dataloaders(
    data_path: str,
    tokenizer,
    batch_size: int = 12,
    max_length: int = 1024,
    val_ratio: float = 0.05,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: int = 42,
    pin_memory: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    dataset = InstructionDataset(data_path, tokenizer, max_length=max_length)

    total_len = len(dataset)
    val_len = int(total_len * val_ratio)
    test_len = int(total_len * test_ratio)
    train_len = total_len - val_len - test_len

    if shuffle:
        train_dataset, val_dataset, test_dataset = random_split(
            dataset,
            lengths=[train_len, val_len, test_len],
            generator=torch.Generator().manual_seed(seed),
        )
    else:
        train_dataset = Subset(dataset, range(0, train_len))
        val_dataset = Subset(dataset, range(train_len, train_len + val_len))
        test_dataset = Subset(dataset, range(train_len + val_len, total_len))

    # DistributedSampler if in DDP mode
    if world_size > 1:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        test_sampler = DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    else:
        train_sampler = val_sampler = test_sampler = None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None and shuffle),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader

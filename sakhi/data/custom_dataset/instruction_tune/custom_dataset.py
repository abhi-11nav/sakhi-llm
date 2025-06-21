import json

import torch
from torch.utils.data import DataLoader, Dataset, random_split


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

        if entry["telugu_input"] == entry["telugu_input"]:
            prompt = f"{entry['telugu_instruction']} {entry['telugu_input']}"
        else:
            prompt = entry["telugu_instruction"]

        input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        label_ids = self.tokenizer(
            entry["telugu_output"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )["input_ids"].squeeze(0)

        return {
            "input_ids": input_ids,
            "labels": label_ids,
        }


def get_dataloaders(
    data_path: str,
    tokenizer,
    batch_size: int = 12,
    max_length: int = 1024,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    num_workers: int = 4,
    seed: int = 42,
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
        train_dataset = torch.utils.data.Subset(dataset, range(0, train_len))
        val_dataset = torch.utils.data.Subset(
            dataset, range(train_len, train_len + val_len)
        )
        test_dataset = torch.utils.data.Subset(
            dataset, range(train_len + val_len, total_len)
        )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, val_loader, test_loader

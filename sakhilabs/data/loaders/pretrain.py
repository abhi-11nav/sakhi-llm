import json
from typing import Optional

import torch
from torch.utils.data import IterableDataset, get_worker_info


class CustomDataset(IterableDataset):
    def __init__(
        self,
        jsonl_path: str,
        chunk_length: int,
        max_samples: Optional[int] = None,
    ):
        self.jsonl_path = jsonl_path
        self.chunk_length = chunk_length
        self.max_samples = max_samples

        if torch.distributed.is_initialized():
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

    def __iter__(self):
        """
        Yields torch tensors of input_ids and labels from pre-tokenized JSON lines.
        """
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        total_shards = self.world_size * num_workers
        shard_id = self.rank * num_workers + worker_id

        sample_count = 0
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i % total_shards != shard_id:
                    continue

                try:
                    sample = json.loads(line)
                    input_ids = sample["input_ids"]
                    labels = sample["labels"]

                    if (
                        len(input_ids) != self.chunk_length
                        or len(labels) != self.chunk_length
                    ):
                        continue

                    yield {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long),
                    }

                    sample_count += 1
                    if self.max_samples and sample_count >= self.max_samples:
                        return
                except (json.JSONDecodeError, KeyError, TypeError):
                    continue

    def __len__(self):
        # Pre computed; Change according to your dataset size                      
        dataset_len = 14_115_069
        return dataset_len // self.world_size

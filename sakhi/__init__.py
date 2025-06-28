import json
import os
from typing import Optional

import torch
from huggingface_hub import snapshot_download
from transformers import PreTrainedTokenizerFast

from sakhi.model.model import SakhiModel


def load_tokenizer(path: str):
    return PreTrainedTokenizerFast.from_pretrained(path, subfolder="tokenizer")


def prepare_instruct_prompt(prompt: str):
    prefix = "<|instruction|>"
    response_tag = "<|response|>"

    instruct_prompt = f"{prefix} {prompt} {response_tag} "
    return instruct_prompt


def load_model(path: str, device: Optional[str] = None):
    if not device:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isdir(path):
        path = snapshot_download(repo_id=path)

    config_path = f"{path}/config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    model = SakhiModel(
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        ff_dim=config["ff_dim"],
        num_layers=config["num_layers"],
        vocab_size=config["vocab_size"],
    )

    weights_path = f"{path}/pytorch_model.bin"
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

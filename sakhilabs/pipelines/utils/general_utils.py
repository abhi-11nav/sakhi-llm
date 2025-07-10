import logging
import os
from pathlib import Path

import torch
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

from sakhilabs.configs.utils.load_config import SakhiConfig
from sakhilabs.model.model import SakhiModel


def do_sanity_checks(config):
    if not os.path.exists(config.paths.tokenizer_path):
        raise FileNotFoundError(
            f"Tokenizer file not found at {config.paths.tokenizer_path}"
        )

    if not os.path.exists(config.paths.dataset_path):
        raise FileNotFoundError(
            f"Dataset file not found at {config.paths.dataset_path}"
        )

    Path(config.paths.log_dir).mkdir(parents=True, exist_ok=True)
    Path(config.paths.model_dir).mkdir(parents=True, exist_ok=True)

    assert os.access(config.paths.log_dir, os.W_OK), "Cannot write to log directory"
    assert os.access(config.paths.model_dir, os.W_OK), (
        "Cannot write to model save directory"
    )


def setup(rank: int, world_size: int, config):
    if world_size > 1:
        assert os.environ["MASTER_ADDR"], "Master Node variable not found"
        os.environ["MASTER_PORT"] = config.train_parameters.master_port

        init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
    else:
        torch.cuda.set_device(0)


def get_sakhi_model(rank: int, world_size: int, config: SakhiConfig, tokenizer):
    model = SakhiModel(
        embed_dim=config.model_parameters.embed_dim,
        num_heads=config.model_parameters.num_heads,
        ff_dim=config.model_parameters.ff_dim,
        vocab_size=config.model_parameters.vocab_size,
        num_layers=config.model_parameters.num_layers,
    ).to(rank)

    assert len(tokenizer) == 64002
    if config.train_parameters.call_torch_compile_on_model:
        model = torch.compile(model)

    if config.train_parameters.resume:
        if os.path.isfile(config.train_parameters.resume):
            state_dict = torch.load(
                config.train_parameters.resume, map_location=f"cuda:{rank}"
            )
            model.load_state_dict(state_dict)

    model.resize_token_embeddings(new_vocab_size=len(tokenizer))
    sakhi_model = DDP(model, device_ids=[rank]) if world_size > 1 else model

    return sakhi_model


def setup_logging(rank, log_dir="logs"):
    """Setup logging for each process"""
    Path(log_dir).mkdir(exist_ok=True)

    # Setup logger
    logger = logging.getLogger(f"rank_{rank}")
    logger.setLevel(logging.INFO)

    logger.handlers.clear()

    log_file = Path(log_dir) / f"training_rank_{rank}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler (only for rank 0 to avoid spam)
    if rank == 0:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logger.addHandler(console_handler)

    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - Rank %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger

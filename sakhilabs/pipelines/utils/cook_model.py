import os
from functools import partial
from typing import Optional

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (CPUOffload,
                                                                MixedPrecision)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP

from sakhilabs.model.components.decoder import TransformerDecoderBlock
from sakhilabs.model.model import SakhiModel
from sakhilabs.pipelines.utils.constants import TrainMode


def get_sakhi_model(
    embed_dim: int,
    num_heads: int,
    ff_dim: int,
    vocab_size: int,
    num_layers: int,
    rank: int = 0,
    world_size: int = torch.cuda.device_count(),
    train_mode: TrainMode = TrainMode.GENERAL,
    call_torch_compile_on_model: bool = False,
    resume: Optional[str] = None,
    resize_model_output_to_size: Optional[int] = None,
    fp16: bool = True,
):
    if train_mode != TrainMode.GENERAL:
        if len(world_size) <= 1:
            raise ValueError(
                "world_size must be greater than 1 for multitraining setting"
            )

    model = SakhiModel(
        embed_dim=embed_dim,
        num_heads=num_heads,
        ff_dim=ff_dim,
        vocab_size=vocab_size,
        num_layers=num_layers,
    ).to(rank)

    if call_torch_compile_on_model:
        model = torch.compile(model)

    if resume:
        if os.path.isfile(resume):
            state_dict = torch.load(resume, map_location=f"cuda:{rank}")
            model.load_state_dict(state_dict)
        else:
            raise FileNotFoundError(f"Model {resume} does not exist")

    if resize_model_output_to_size:
        model.resize_token_embeddings(new_vocab_size=resize_model_output_to_size)

    if train_mode == TrainMode.GENERAL:
        sakhi_model = model
    elif train_mode == TrainMode.DDP:
        sakhi_model = DDP(model, device_ids=[rank]) if world_size > 1 else model
    elif train_mode == TrainMode.FSDP:
        mixed_precision_policy = None
        if fp16:
            mixed_precision_policy = MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16,
            )

        auto_wrap_policy = partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={TransformerDecoderBlock},
        )
        sakhi_model = FSDP(
            model,
            device_id=rank,
            auto_wrap_policy=auto_wrap_policy,
            mixed_precision=mixed_precision_policy,
            cpu_offload=CPUOffload(offload_params=False),
            use_orig_params=True,
        )

    return sakhi_model

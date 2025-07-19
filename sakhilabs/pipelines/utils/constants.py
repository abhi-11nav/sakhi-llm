from enum import Enum


class TrainMode(Enum):
    DDP = "ddp"
    FSDP = "fsdp"
    GENERAL = "general"

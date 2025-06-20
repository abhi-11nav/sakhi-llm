import hashlib
import random

import numpy as np
import torch


def hash_tensor(tensor):
    s = str(tensor.tolist())
    return hashlib.md5(s.encode()).hexdigest()[:6]


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

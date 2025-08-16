# https://github.com/openai/gpt-oss/blob/main/gpt_oss/torch/model.py


import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6, device=None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, device=device, dtype=torch.float32))

    def forward(self, x):
        assert x.shape[-1] == self.dim
        orig_dtype = x.dtype
        t = x.float()
        t = t * torch.rsqrt(torch.mean(t**2, dim=-1, keepdim=True) + self.eps)
        return (t * self.weight).to(orig_dtype)

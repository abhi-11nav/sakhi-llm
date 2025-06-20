import torch


def apply_rotary_pos_emb(q, k, seq_len, device):
    """rotate matrices q dn k based on position"""
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)

    head_dim = q.shape[-1]

    if head_dim % 2 != 0:
        raise ValueError("head dimension cannot be an odd value")

    # calculate theta
    theta_numerator = torch.arange(0, head_dim, 2, device=device).float()
    theta = 1.0 / (10000 ** (theta_numerator / head_dim))

    # calculate positional embeddings
    pos = torch.arange(seq_len, device=device).float()

    # computing the outer product
    freqs = torch.outer(pos, theta)

    cos = freqs.cos()[None, None, :, :]
    sin = freqs.sin()[None, None, :, :]

    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    q_rot = torch.stack([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1).flatten(-2)
    k_rot = torch.stack([k1 * cos - k2 * sin, k1 * sin + k2 * cos], dim=-1).flatten(-2)

    return q_rot.transpose(1, 2), k_rot.transpose(1, 2)

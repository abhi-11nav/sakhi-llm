import torch.nn as nn

from sakhi.model.components.mha_rope import MultiheadSelfAttentionRoPE
from sakhi.model.components.rms_norm import RMSNorm
from sakhi.model.components.swiglu import SwiGLU


class TransformerDecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.masked_attention = MultiheadSelfAttentionRoPE(
            embed_dim, num_heads, dropout=dropout
        )
        self.norm1 = RMSNorm(dim=embed_dim)
        self.dropout1 = nn.Dropout(dropout)

        self.ff = SwiGLU(dim=embed_dim, hidden_dim=ff_dim)
        self.norm2 = RMSNorm(dim=embed_dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, tgt, tgt_mask=None):
        """foward call for decoder"""
        attn_output = self.masked_attention(tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout1(attn_output))

        ff_output = self.ff(tgt)
        tgt = self.norm2(tgt + self.dropout2(ff_output))

        return tgt

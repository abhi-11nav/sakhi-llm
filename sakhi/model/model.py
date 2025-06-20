import torch
import torch.nn as nn

from sakhi.model.components.decoder import TransformerDecoderBlock
from sakhi.model.components.nn_utils import generate_square_subsequent_mask


class SakhiModel(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        ff_dim: int,
        vocab_size: int,
        num_layers: int,
    ):
        super(SakhiModel, self).__init__()

        self.embed_dim = embed_dim
        self.decoder_embedding = nn.Embedding(vocab_size, embed_dim)
        self.decoder_blocks = nn.ModuleList(
            [
                TransformerDecoderBlock(embed_dim, num_heads, ff_dim)
                for _ in range(num_layers)
            ]
        )
        self.output_projection = nn.Linear(embed_dim, vocab_size)

    def forward(self, tgt_input):
        batch_size, seq_len = tgt_input.shape
        tgt_embedded = self.decoder_embedding(tgt_input)
        tgt_mask = generate_square_subsequent_mask(seq_len).to(tgt_input.device)

        decoder_output = tgt_embedded
        for decoder_block in self.decoder_blocks:
            decoder_output = decoder_block(decoder_output, tgt_mask=tgt_mask)

        output_logits = self.output_projection(decoder_output)
        return output_logits

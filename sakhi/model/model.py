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

    def resize_token_embeddings(self, new_vocab_size: int):
        old_vocab_size, embed_dim = self.decoder_embedding.weight.shape
        device = self.decoder_embedding.weight.device

        if new_vocab_size <= old_vocab_size:
            print("New vocab size is not larger than existing. No resizing done.")
            return

        # Resize embedding layer
        new_embed = nn.Embedding(new_vocab_size, embed_dim).to(device)
        new_embed.weight.data[:old_vocab_size] = self.decoder_embedding.weight.data

        std = self.decoder_embedding.weight.data.std()
        new_embed.weight.data[old_vocab_size:] = (
            torch.randn(new_vocab_size - old_vocab_size, embed_dim) * std
        )
        self.decoder_embedding = new_embed

        # Resize output projection
        old_out_dim, in_dim = self.output_projection.weight.shape
        device = self.output_projection.weight.device
        if old_out_dim != old_vocab_size:
            raise ValueError(
                "Old output dim is not equal to old vocab size. Something's wrong"
            )

        new_out = nn.Linear(in_dim, new_vocab_size).to(device)
        new_out.weight.data[:old_out_dim] = self.output_projection.weight.data
        new_out.bias.data[:old_out_dim] = self.output_projection.bias.data

        std = self.output_projection.weight.data.std()
        new_out.weight.data[old_out_dim:] = (
            torch.randn(new_vocab_size - old_out_dim, in_dim) * std
        )
        new_out.bias.data[old_out_dim:] = 0.0
        self.output_projection = new_out

    def forward(self, tgt_input):
        batch_size, seq_len = tgt_input.shape
        tgt_embedded = self.decoder_embedding(tgt_input)
        tgt_mask = generate_square_subsequent_mask(seq_len).to(tgt_input.device)

        decoder_output = tgt_embedded
        for decoder_block in self.decoder_blocks:
            decoder_output = decoder_block(decoder_output, tgt_mask=tgt_mask)

        output_logits = self.output_projection(decoder_output)
        return output_logits

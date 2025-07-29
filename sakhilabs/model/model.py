import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from sakhilabs.model.components.decoder import TransformerDecoderBlock
from sakhilabs.model.components.nn_utils import generate_square_subsequent_mask


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

        if new_vocab_size <= old_vocab_size:
            print("New vocab size is not larger than existing. No resizing done.")
            return

        # Get device of current embedding
        device = self.decoder_embedding.weight.device

        # Resize embedding layer
        new_embed = nn.Embedding(new_vocab_size, embed_dim).to(device)
        new_embed.weight.data[:old_vocab_size] = self.decoder_embedding.weight.data

        std = self.decoder_embedding.weight.data.std()
        new_embed.weight.data[old_vocab_size:] = (
            torch.randn(new_vocab_size - old_vocab_size, embed_dim, device=device) * std
        )
        self.decoder_embedding = new_embed

        # Resize output projection
        old_out_dim, in_dim = self.output_projection.weight.shape
        if old_out_dim != old_vocab_size:
            raise ValueError(
                "Old output dim is not equal to old vocab size. Something's wrong"
            )

        # Get device of current output projection
        out_device = self.output_projection.weight.device

        new_out = nn.Linear(in_dim, new_vocab_size).to(out_device)
        new_out.weight.data[:old_out_dim] = self.output_projection.weight.data
        new_out.bias.data[:old_out_dim] = self.output_projection.bias.data

        std = self.output_projection.weight.data.std()
        new_out.weight.data[old_out_dim:] = (
            torch.randn(new_vocab_size - old_out_dim, in_dim, device=out_device) * std
        )
        new_out.bias.data[old_out_dim:] = 0.0
        self.output_projection = new_out

    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 640,
        temperature: float = 0.4,
        top_k: int = 100,
        top_p: float = 0.90,
        repetition_penalty: float = 1.2,
        no_repeat_ngram_size: int = 4,
        num_responses: int = 4,
        tokenizer=None,
    ) -> torch.Tensor:
        all_responses = []

        for _ in range(num_responses):
            generated = input_ids.clone()

            with torch.no_grad():
                for _ in range(max_new_tokens):
                    output = self(generated)
                    logits = output[:, -1, :]

                    # Repetition penalty
                    for token_id in set(generated[0].tolist()):
                        logits[0, token_id] /= repetition_penalty

                    # Top-k filtering
                    if top_k > 0:
                        top_k_values, _ = torch.topk(logits, top_k)
                        logits[logits < top_k_values[:, -1].unsqueeze(1)] = -float(
                            "Inf"
                        )

                    # Top-p (nucleus) filtering
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(
                            logits, descending=True
                        )
                        cumulative_probs = torch.cumsum(
                            F.softmax(sorted_logits, dim=-1), dim=-1
                        )
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[
                            :, :-1
                        ].clone()
                        sorted_indices_to_remove[:, 0] = 0
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        logits[0, indices_to_remove] = -float("Inf")

                    # Sampling with temperature
                    probs = F.softmax(logits / temperature, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                    generated = torch.cat([generated, next_token], dim=1)

                    # Stop if repeating n-grams
                    if (
                        no_repeat_ngram_size > 0
                        and generated.shape[1] > no_repeat_ngram_size
                    ):
                        last_ngram = generated[0, -no_repeat_ngram_size:].tolist()
                        all_ngrams = [
                            generated[0, i : i + no_repeat_ngram_size].tolist()
                            for i in range(generated.shape[1] - no_repeat_ngram_size)
                        ]
                        if last_ngram in all_ngrams[:-1]:
                            break

            output_tokens = generated[0][input_ids.shape[1] :]
            all_responses.append(output_tokens)

        # Find max length for padding
        max_length = max(response.shape[0] for response in all_responses)

        # Get pad token id
        pad_token_id = tokenizer.pad_token_id if tokenizer is not None else 0

        # Pad all responses to the same length
        padded_responses = []
        for response in all_responses:
            if response.shape[0] < max_length:
                padding = torch.full(
                    (max_length - response.shape[0],),
                    pad_token_id,
                    dtype=response.dtype,
                    device=response.device,
                )
                padded_response = torch.cat([response, padding])
            else:
                padded_response = response
            padded_responses.append(padded_response)

        return torch.stack(padded_responses)

    def forward(self, tgt_input):
        batch_size, seq_len = tgt_input.shape
        tgt_embedded = self.decoder_embedding(tgt_input)
        tgt_mask = generate_square_subsequent_mask(seq_len).to(tgt_input.device)

        decoder_output = tgt_embedded
        for decoder_block in self.decoder_blocks:
            decoder_output = decoder_block(decoder_output, tgt_mask=tgt_mask)

        output_logits = self.output_projection(decoder_output)
        return output_logits

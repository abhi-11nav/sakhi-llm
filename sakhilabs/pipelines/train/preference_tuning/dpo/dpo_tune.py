import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from sakhilabs.configs.utils.load_config import SakhiConfig
from sakhilabs.pipelines.train.preference_tuning.dpo.dataset import (
    DPODataset, dpo_collate_fn)
from sakhilabs.pipelines.utils.constants import TrainMode
from sakhilabs.pipelines.utils.cook_model import get_sakhi_model


class DirectPreferenceOptimization:
    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        tokenizer,
        dataset: str,
        save_dir: str,
        beta: float = 0.9,
        max_grad_norm: float = 2.0,
        learning_rate: float = 1e-5,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.policy_model = policy_model.to(device)
        self.reference_model = reference_model.to(device)
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.beta = beta
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.save_dir = save_dir  # <- NEW

        os.makedirs(self.save_dir, exist_ok=True)  # <- Ensure directory exists

        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=learning_rate
        )

        self.reference_model.eval()
        for param in self.reference_model.parameters():
            param.requires_grad = False

    def generate_log_probs(
        self, model: nn.Module, tokens: torch.Tensor, prompt_cutoff_length: int
    ):
        with torch.no_grad() if not model.training else torch.enable_grad():
            logits = model(tokens)  # [batch, seq_len, vocab_size]
            log_probs = F.log_softmax(logits, dim=-1)
            token_log_probs = torch.gather(log_probs, 2, tokens.unsqueeze(-1)).squeeze(
                -1
            )

        sliced_token_log_probs = token_log_probs[:, prompt_cutoff_length:]
        sliced_tokens = tokens[:, prompt_cutoff_length:]

        mask = (
            sliced_tokens != self.tokenizer.pad_token_id
        ).float()  # [batch, seq_len]

        return sliced_token_log_probs, mask

    def __call__(self, batch_size: int = 2, epochs: int = 3):
        self.policy_model.train()
        dataset = DPODataset(self.dataset, self.tokenizer)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dpo_collate_fn,
        )

        for epoch in range(epochs):
            total_loss = 0.0

            for step, batch in enumerate(dataloader):
                prompt = batch["prompt_tokens"].to(self.device)
                pos_resp = batch["pos_tokens"].to(self.device)
                neg_resp = batch["neg_tokens"].to(self.device)

                pos_tokens = torch.cat([prompt, pos_resp], dim=1)
                neg_tokens = torch.cat([prompt, neg_resp], dim=1)

                prompt_cutoff = prompt.shape[1]

                pos_log_policy, pos_mask = self.generate_log_probs(
                    self.policy_model, pos_tokens, prompt_cutoff
                )
                pos_log_ref, ref_mask_pos = self.generate_log_probs(
                    self.reference_model, pos_tokens, prompt_cutoff
                )

                neg_log_policy, neg_mask = self.generate_log_probs(
                    self.policy_model, neg_tokens, prompt_cutoff
                )
                neg_log_ref, ref_mask_neg = self.generate_log_probs(
                    self.reference_model, neg_tokens, prompt_cutoff
                )

                assert torch.all(pos_mask == ref_mask_pos)
                assert torch.all(neg_mask == ref_mask_neg)

                # Compute masked reward
                pos_reward = torch.sum(
                    (pos_log_policy - pos_log_ref) * pos_mask, dim=1
                ) / pos_mask.sum(dim=1).clamp(min=1)
                neg_reward = torch.sum(
                    (neg_log_policy - neg_log_ref) * neg_mask, dim=1
                ) / neg_mask.sum(dim=1).clamp(min=1)

                reward_diff = self.beta * (pos_reward - neg_reward)
                loss = -torch.log(torch.sigmoid(reward_diff + 1e-8)).mean()

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()

                print(
                    f"Epoch [{epoch + 1}/{epochs}] Step [{step + 1}/{len(dataloader)}] Loss: {loss.item():.4f}"
                )

            avg_loss = total_loss / len(dataloader)

            model_path = os.path.join(
                self.save_dir, f"policy_model_epoch_{epoch + 1}.pt"
            )
            torch.save(self.policy_model.state_dict(), model_path)
            print(f"Saved model to {model_path}")
            print(f"Epoch [{epoch + 1}] Completed. Average Loss: {avg_loss:.4f}")


def get_model(config: SakhiConfig):
    sakhi_model = get_sakhi_model(
        embed_dim=config.model_parameters.embed_dim,
        num_heads=config.model_parameters.num_heads,
        ff_dim=config.model_parameters.ff_dim,
        vocab_size=config.model_parameters.vocab_size,
        num_layers=config.model_parameters.num_layers,
        train_mode=TrainMode(config.train_parameters.mode),
        resume=config.train_parameters.resume,
        resize_model_output_to_size=config.model_parameters.vocab_size,
        fp16=True,
        for_inference=True,
    )
    return sakhi_model


if __name__ == "__main__":
    config_path = "sakhilabs/configs/sakhi-telugu-681M-instruct-0625.yaml"
    config = SakhiConfig._load_config(config_path=config_path)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.paths.tokenizer_path)

    dpo_dataset = "sakhilabs/pipelines/train/preference_tuning/preference_data.json"

    policy_model = get_model(config=config)
    reference_model = get_model(config=config)

    dpo = DirectPreferenceOptimization(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        dataset=dpo_dataset,
        save_dir="preference_tuning_save_dir",
    )

    dpo(epochs=3, batch_size=10)

    print("YES")

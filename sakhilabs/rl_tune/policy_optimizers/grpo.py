from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from sakhilabs.configs.utils.load_config import SakhiConfig
from sakhilabs.pipelines.utils.constants import TrainMode
from sakhilabs.pipelines.utils.cook_model import get_sakhi_model


class GRPO:
    """
    Group Relative Policy Optimization (GRPO) implementation.

    GRPO is a policy optimization algorithm that uses group-based relative comparisons
    to improve policy learning in reinforcement learning from human feedback (RLHF).
    """

    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: Optional[nn.Module] = None,
        tokenizer=None,
        beta: float = 0.1,
        group_size: int = 4,
        epsilon: float = 1e-8,
        max_grad_norm: float = 1.0,
    ):
        """
        Initialize GRPO optimizer.

        Args:
            policy_model: The policy model to optimize
            reference_model: Reference model for KL divergence computation
            tokenizer: Tokenizer for text processing
            beta: KL divergence coefficient
            group_size: Size of groups for relative comparison
            epsilon: Small constant for numerical stability
            max_grad_norm: Maximum gradient norm for clipping
        """
        self.policy_model = policy_model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.beta = beta
        self.group_size = group_size
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm

        # Freeze reference model if provided
        if self.reference_model is not None:
            for param in self.reference_model.parameters():
                param.requires_grad = False

    def compute_log_probs(
        self, model: nn.Module, input_ids: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute log probabilities for given input and labels.

        Args:
            model: Model to compute log probs with
            input_ids: Input token ids
            labels: Target labels

        Returns:
            Log probabilities for each token
        """
        outputs = model(input_ids)
        logits = outputs.logits if hasattr(outputs, "logits") else outputs

        # Shift logits and labels for causal language modeling
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)

        # Gather log probabilities for actual tokens
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        # Mask out padding tokens
        mask = (shift_labels != -100).float()
        token_log_probs = token_log_probs * mask

        # Sum log probs for each sequence
        sequence_log_probs = token_log_probs.sum(dim=-1)

        return sequence_log_probs

    def compute_kl_divergence(
        self, policy_log_probs: torch.Tensor, reference_log_probs: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute KL divergence between policy and reference distributions.

        Args:
            policy_log_probs: Log probabilities from policy model
            reference_log_probs: Log probabilities from reference model

        Returns:
            KL divergence values
        """
        return policy_log_probs - reference_log_probs

    def group_relative_loss(
        self,
        rewards: torch.Tensor,
        log_probs: torch.Tensor,
        ref_log_probs: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute the GRPO loss using group-based relative comparisons.

        Args:
            rewards: Reward scores for each sample
            log_probs: Log probabilities from policy model
            ref_log_probs: Log probabilities from reference model

        Returns:
            Loss value and metrics dictionary
        """
        batch_size = rewards.shape[0]

        # Ensure batch size is divisible by group size
        if batch_size % self.group_size != 0:
            # Trim to make divisible
            trim_size = batch_size - (batch_size % self.group_size)
            rewards = rewards[:trim_size]
            log_probs = log_probs[:trim_size]
            if ref_log_probs is not None:
                ref_log_probs = ref_log_probs[:trim_size]
            batch_size = trim_size

        # Reshape into groups
        num_groups = batch_size // self.group_size
        grouped_rewards = rewards.view(num_groups, self.group_size)
        grouped_log_probs = log_probs.view(num_groups, self.group_size)

        if ref_log_probs is not None:
            grouped_ref_log_probs = ref_log_probs.view(num_groups, self.group_size)
            kl_divergence = self.compute_kl_divergence(
                grouped_log_probs, grouped_ref_log_probs
            )
        else:
            kl_divergence = torch.zeros_like(grouped_log_probs)

        # Compute relative advantages within each group
        group_means = grouped_rewards.mean(dim=1, keepdim=True)
        relative_advantages = grouped_rewards - group_means

        # Compute policy ratios with KL penalty
        policy_scores = grouped_log_probs - self.beta * kl_divergence

        # GRPO loss: maximize policy scores weighted by relative advantages
        loss = -torch.mean(policy_scores * relative_advantages)

        # Compute metrics
        metrics = {
            "loss": loss.item(),
            "mean_reward": rewards.mean().item(),
            "std_reward": rewards.std().item(),
            "mean_advantage": relative_advantages.mean().item(),
            "mean_kl_div": kl_divergence.mean().item()
            if ref_log_probs is not None
            else 0.0,
            "policy_score_mean": policy_scores.mean().item(),
        }

        return loss, metrics

    def update_policy(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        rewards: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> Dict[str, float]:
        """
        Perform a single policy update step.

        Args:
            input_ids: Input token ids
            labels: Target labels
            rewards: Reward scores
            optimizer: Optimizer for the policy model

        Returns:
            Dictionary of training metrics
        """
        # Compute log probabilities from policy model
        policy_log_probs = self.compute_log_probs(self.policy_model, input_ids, labels)

        # Compute reference log probabilities if reference model is available
        ref_log_probs = None
        if self.reference_model is not None:
            with torch.no_grad():
                ref_log_probs = self.compute_log_probs(
                    self.reference_model, input_ids, labels
                )

        # Compute GRPO loss
        loss, metrics = self.group_relative_loss(
            rewards, policy_log_probs, ref_log_probs
        )

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.policy_model.parameters(), self.max_grad_norm
        )

        optimizer.step()

        # Add gradient norm to metrics
        metrics["grad_norm"] = grad_norm.item()

        return metrics

    def generate_and_score(
        self,
        prompts: List[str],
        reward_model: nn.Module,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        do_sample: bool = True,
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Generate responses and compute rewards.

        Args:
            prompts: List of prompt strings
            reward_model: Model to compute rewards
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            do_sample: Whether to use sampling

        Returns:
            Generated responses and their rewards
        """
        responses = []
        all_rewards = []

        for prompt in prompts:
            # Tokenize prompt
            inputs = self.tokenizer(
                prompt, return_tensors="pt", padding=True, truncation=True
            )

            # Generate response
            with torch.no_grad():
                outputs = self.policy_model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                )

            # Decode response
            response = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
            responses.append(response)

            # Compute reward
            full_text = prompt + response
            reward_inputs = self.tokenizer(
                full_text, return_tensors="pt", padding=True, truncation=True
            )

            with torch.no_grad():
                reward_outputs = reward_model(**reward_inputs)
                reward = reward_outputs.logits.squeeze().item()
                all_rewards.append(reward)

        return responses, torch.tensor(all_rewards)


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
    )
    return sakhi_model


def reward_model(query: str, responses: List[str]):
    pass


if __name__ == "__main__":
    config_path = "/home/abhi11/projects/def-tusharma/abhi11/sakhi/repos/sakhi-llm/sakhilabs/configs/sakhi-telugu-1B-pretrained-0725.yaml"
    config = SakhiConfig._load_config(config_path=config_path)

    policy_model = get_model(config=config_path)
    reference_model = get_model(config=config_path)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.paths.tokenizer_path)

    grpo = GRPO(
        policy_model=policy_model, reference_model=reference_model, tokenizer=tokenizer
    )

    sample_prompts = ["నమస్కారం, మీరు ఎలా ఉన్నారు?", "తెలుగు భాష గురించి మీకు ఏమి తెలుసు?"]
    grpo.generate_and_score(prompts=sample_prompts, reward_model=reward_model)

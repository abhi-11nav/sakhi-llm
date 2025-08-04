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
        reference_model: nn.Module,
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

    def reward(self, prompt: str, responses: List[str]) -> torch.Tensor:
        judge_model = "gpt4o"
        print(judge_model)
        import random

        return torch.tensor([random.randint(1, 10) for _ in range(len(responses))])

    def generate_samples_with_poilcy(
        self,
        prompt: str,
        max_new_tokens: int,
        num_responses: int,
        temperature: float = 1.0,
    ) -> List[str]:
        inputs = self.tokenizer(prompt, return_tensors="pt")["input_ids"]

        # Generate multiple responses
        with torch.no_grad():
            raw_response_tokens, probs = self.policy_model.generate(
                input_ids=inputs,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                num_responses=num_responses,
                tokenizer=self.tokenizer,
            )

        # Decode response
        decode_response = [
            self.tokenizer.decode(raw_response_tokens[i], skip_special_tokens=True)
            for i in range(len(raw_response_tokens))
        ]
        return decode_response, raw_response_tokens, probs

    def generate_probs_with_reference_model(
        self, tokens: torch.Tensor, prompt_cutoff_length: int, temperature: float = 1.0
    ):
        # Get logits from the reference model
        reference_tokens_output = self.reference_model(
            tokens
        )  # [batch, seq_len, vocab_size]

        # Convert logits to probabilities
        probs = F.softmax(
            reference_tokens_output / temperature, dim=-1
        )  # [batch, seq_len, vocab_size]

        # Gather the probability for the actual token at each position
        token_probs = torch.gather(probs, 2, tokens.unsqueeze(-1)).squeeze(
            -1
        )  # [batch, seq_len]

        # Return only the probabilities after the prompt
        return token_probs[:, prompt_cutoff_length:]

    def compute_advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() if len(rewards) > 1 else 1.0

        advantages = (rewards_tensor - mean_reward) / std_reward
        return advantages

    def __call__(self, prompts: List[str], max_new_tokens: int, num_responses: int):
        state_action_pairs = {}

        for i in range(len(prompts)):
            (
                online_policy_generated_samples,
                online_policy_tokens,
                online_policy_generated_probs,
            ) = self.generate_samples_with_poilcy(
                prompt=prompts[i],
                num_responses=num_responses,
                max_new_tokens=max_new_tokens,
            )

            online_policy_prob_sum = torch.sum(
                torch.log(online_policy_generated_probs), dim=1
            )
            rewards = self.reward(
                prompt=prompts[i], responses=online_policy_generated_samples
            )

            self.tokenizer(prompts[i], return_tensors="pt")["input_ids"]
            advantage_factor = self.compute_advantage(rewards=rewards)

            prompt_tokens = self.tokenizer(prompts[i], return_tensors="pt")["input_ids"]
            prompt_tokens_repeated = prompt_tokens.repeat(
                online_policy_tokens.size(0), 1
            )
            reference_model_input_tokens = torch.cat(
                [prompt_tokens_repeated, online_policy_tokens], dim=1
            )

            reference_model_probs = self.generate_probs_with_reference_model(
                tokens=reference_model_input_tokens,
                prompt_cutoff_length=prompt_tokens.shape[1],
            )

            reference_policy_prob_sum = torch.sum(
                torch.log(reference_model_probs), dim=1
            )

            state_action_pairs.update(
                {
                    i: {
                        "prompt": prompts[i],
                        "responses": online_policy_generated_samples,
                        "rewards": rewards,
                    }
                }
            )

        return state_action_pairs


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


def reward_model(query: str, responses: List[str]):
    pass


def prepare_instruct_prompt(prompt: str):
    prefix = "<|instruction|>"
    response_tag = "<|response|>"

    instruct_prompt = f"{prefix} {prompt} {response_tag} "
    return instruct_prompt


if __name__ == "__main__":
    config_path = "sakhilabs/configs/sakhi-telugu-681M-instruct-0625.yaml"
    config = SakhiConfig._load_config(config_path=config_path)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.paths.tokenizer_path)

    sample_prompts = ["నమస్కారం, మీరు ఎలా ఉన్నారు?", "తెలుగు భాష గురించి మీకు ఏమి తెలుసు?"]
    sample_prompts = [prepare_instruct_prompt(prompt) for prompt in sample_prompts]

    policy_model = get_model(config=config)
    reference_model = get_model(config=config)

    grpo = GRPO(
        policy_model=policy_model, reference_model=reference_model, tokenizer=tokenizer
    )

    grpo(prompts=sample_prompts, max_new_tokens=128, num_responses=4)

    print("YES")

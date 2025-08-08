from typing import List

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

    def generate_log_probs_with_reference_model(
        self, tokens: torch.Tensor, prompt_cutoff_length: int, temperature: float = 1.0
    ):
        # Get logits from the reference model
        reference_logits = self.reference_model(tokens)  # [batch, seq_len, vocab_size]

        # Convert logits to log-probabilities
        log_probs = F.log_softmax(reference_logits / temperature, dim=-1)

        # Get log-probs of the actual tokens
        token_log_probs = torch.gather(log_probs, 2, tokens.unsqueeze(-1)).squeeze(-1)

        # Slice to get only generated part (i.e., after prompt)
        return token_log_probs[:, prompt_cutoff_length:]

    def compute_advantage(self, rewards: torch.Tensor) -> torch.Tensor:
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        mean_reward = rewards_tensor.mean()
        std_reward = rewards_tensor.std() if len(rewards) > 1 else 1.0

        advantages = (rewards_tensor - mean_reward) / std_reward
        return advantages

    def __call__(self, prompts: List[str], max_new_tokens: int, num_responses: int):
        total_loss = []

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

            rewards = self.reward(
                prompt=prompts[i], responses=online_policy_generated_samples
            )

            # self.tokenizer(prompts[i], return_tensors="pt")["input_ids"]
            advantage_factor = self.compute_advantage(rewards=rewards)

            prompt_tokens = self.tokenizer(prompts[i], return_tensors="pt")["input_ids"]
            prompt_tokens_repeated = prompt_tokens.repeat(
                online_policy_tokens.size(0), 1
            )
            reference_model_input_tokens = torch.cat(
                [prompt_tokens_repeated, online_policy_tokens], dim=1
            )

            reference_model_probs = self.generate_log_probs_with_reference_model(
                tokens=reference_model_input_tokens,
                prompt_cutoff_length=prompt_tokens.shape[1],
            )

            padding_start_indices = []
            for token_seq in online_policy_tokens:
                # Find the first occurrence of padding token (pad_token_id = 1)
                pad_positions = (token_seq == 1).nonzero(as_tuple=True)[0]
                if len(pad_positions) > 0:
                    padding_start_indices.append(pad_positions[0].item())
                else:
                    padding_start_indices.append(len(token_seq))

            online_policy_generated_probs = [
                online_policy_generated_probs[i, : padding_start_indices[i]]
                for i in range(4)
            ]

            reference_model_probs = [
                reference_model_probs[i, : padding_start_indices[i]] for i in range(4)
            ]

            per_sample_losses = []

            for i in range(len(online_policy_generated_probs)):
                ratio = online_policy_generated_probs[i] / (
                    reference_model_probs[i] + self.epsilon
                )
                clipped_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)

                # per-token loss using clipped surrogate
                loss_per_token = torch.min(
                    ratio * advantage_factor[i], clipped_ratio * advantage_factor[i]
                )

                # mean over tokens in this sample
                per_sample_loss = loss_per_token.mean()
                per_sample_losses.append(per_sample_loss)

                print(
                    "Advantage mean/std:",
                    advantage_factor.mean(),
                    advantage_factor.std(),
                )
                print("Ratio min/max:", ratio.min(), ratio.max())
                print(
                    "Clipped Ratio min/max:", clipped_ratio.min(), clipped_ratio.max()
                )
                print("Token count:", online_policy_generated_probs[i].shape[0])

            loss = torch.mean(torch.stack(per_sample_losses))

            total_loss.append(loss)

        return -(torch.mean(torch.stack(total_loss)))


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

    sample_prompts = [
        "నమస్కారం, మీరు ఎలా ఉన్నారు?",
        "తెలుగు భాష గురించి మీకు ఏమి తెలుసు?",
        "తెలుగు పదబంధం 'చెట్టు నీడ' అర్థం ఏమిటి?",
    ]
    sample_prompts = [prepare_instruct_prompt(prompt) for prompt in sample_prompts]

    policy_model = get_model(config=config)
    reference_model = get_model(config=config)

    grpo = GRPO(
        policy_model=policy_model, reference_model=reference_model, tokenizer=tokenizer
    )

    loss = grpo(prompts=sample_prompts, max_new_tokens=128, num_responses=4)

    print("YES")

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from sakhilabs.configs.utils.load_config import SakhiConfig
from sakhilabs.pipelines.utils.constants import TrainMode
from sakhilabs.pipelines.utils.cook_model import get_sakhi_model


class DirectPreferenceOptimization:
    def __init__(
        self,
        policy_model: nn.Module,
        reference_model: nn.Module,
        tokenizer,
        dataset: List[Dict],
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

        self.optimizer = torch.optim.AdamW(
            self.policy_model.parameters(), lr=learning_rate
        )

        # Freeze reference model
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
        return token_log_probs[:, prompt_cutoff_length:]

    def __call__(self, epochs: int = 3):
        self.policy_model.train()

        for epoch in range(epochs):
            total_loss = 0.0

            for step, data in enumerate(self.dataset):
                # Tokenize
                prompt = self.tokenizer(
                    data["instruction"], return_tensors="pt"
                ).input_ids.to(self.device)
                pos_resp = self.tokenizer(
                    data["response"]["positive"], return_tensors="pt"
                ).input_ids.to(self.device)
                neg_resp = self.tokenizer(
                    data["response"]["negative"], return_tensors="pt"
                ).input_ids.to(self.device)

                pos_tokens = torch.cat([prompt, pos_resp], dim=1)
                neg_tokens = torch.cat([prompt, neg_resp], dim=1)

                prompt_cutoff = prompt.shape[1]

                # Get log probs
                pos_log_policy = self.generate_log_probs(
                    self.policy_model, pos_tokens, prompt_cutoff
                )
                pos_log_ref = self.generate_log_probs(
                    self.reference_model, pos_tokens, prompt_cutoff
                )

                neg_log_policy = self.generate_log_probs(
                    self.policy_model, neg_tokens, prompt_cutoff
                )
                neg_log_ref = self.generate_log_probs(
                    self.reference_model, neg_tokens, prompt_cutoff
                )

                # Compute reward differences
                pos_reward = torch.sum(pos_log_policy - pos_log_ref)
                neg_reward = torch.sum(neg_log_policy - neg_log_ref)

                reward_diff = self.beta * (pos_reward - neg_reward)

                # DPO Loss
                loss = -torch.log(torch.sigmoid(reward_diff + 1e-8))

                # Backprop
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.policy_model.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_loss += loss.item()

                print(
                    f"Epoch [{epoch + 1}/{epochs}] Step [{step + 1}/{len(self.dataset)}] Loss: {loss.item():.4f}"
                )

            avg_loss = total_loss / len(self.dataset)
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


def prepare_instruct_prompt(prompt: str):
    prefix = "<|instruction|>"
    response_tag = "<|response|>"

    instruct_prompt = f"{prefix} {prompt} {response_tag} "
    return instruct_prompt


if __name__ == "__main__":
    config_path = "sakhilabs/configs/sakhi-telugu-681M-instruct-0625.yaml"
    config = SakhiConfig._load_config(config_path=config_path)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(config.paths.tokenizer_path)

    dpo_dataset = [
        {
            "instruction": prepare_instruct_prompt("నమస్కారం, మీరు ఎలా ఉన్నారు?"),
            "response": {
                "positive": "కొంచెం పెద్దదిగా, చూసే స్వామీ! - అచ్చంగా నేచర్ తోనే, డైలీ రొటీతో చేసే పని. ఇప్పుడు సీన్ ఫ్యాషన్, లాంగ్ షూటింగ్ కలిసి వచ్చేశాం. అదొకటి, ఒక హిస్టరీ, ఎప్పటికీ ఎవర్ గ్రీన్. - ఇవి ప్రపంచమంతటా జరి",
                "negative": "కోట్లు, గోడలు, మంది జనం వింటర్ సైలెంట్\u200cగా, నేచర్\u200cతో కనెక్షన్ పొందడం. ఇది జీవిత చరమాంకం అని చెప్పడానికి ఒక టెక్స్ట్. దీంట్లో 'హే, సముద్రం' లాంటి బొమ్మలు గీయడం ఉంటుంది. ఈ డేంజర్ ఎవిడెన్స్ వల్లన",
            },
        },
        {
            "instruction": prepare_instruct_prompt("తెలుగు భాష గురించి మీకు ఏమి తెలుసు?"),
            "response": {
                "positive": "జుగాటుతో ప్రశాంతమైన, గౌరవించబడే రూమ్! మీ ఉదయం ఇంటి నుండి, పని సూపర్బ్\u200cగా, అన్వేషించే వంటకం. ఎవరైనా నడుస్తూ, మెట్ల దగ్గర కొందరు, చేతులతో, సరదాగా! కాబట్టి టైం సేవ్ అవుతుంది. ఇక్కడ కొనడానికి వెళ్లడా",
                "negative": "సంవత్సరం పొడవునా, అనేక రకాల వంటలు చేసి మీ తోట లోకి, దృశ్య కళ్ళుగప్పి రుచి చూడండి. ఒక బలమైన నిర్మాణం. తెలుసా, - ఈ పర్సన్ ఫుడ్ యొక్క భౌతికశాస్త్రము గుండె. ఉత్తర దేశాన్ని పరిశీలించి",
            },
        },
        {
            "instruction": prepare_instruct_prompt("తెలుగు పదబంధం 'చెట్టు నీడ' అర్థం ఏమిటి?"),
            "response": {
                "positive": "కొంచెం పెద్దదిగా, చూసే స్వామీ! - అచ్చంగా నేచర్ తోనే, డైలీ రొటీతో చేసే పని. ఇప్పుడు సీన్ ఫ్యాషన్, లాంగ్ షూటింగ్ కలిసి వచ్చేశాం. అదొకటి, ఒక హిస్టరీ, ఎప్పటికీ ఎవర్ గ్రీన్. - ఇవి ప్రపంచమంతటా జరి",
                "negative": "సంవత్సరం పొడవునా, అనేక రకాల వంటలు చేసి మీ తోట లోకి, దృశ్య కళ్ళుగప్పి రుచి చూడండి. ఒక బలమైన నిర్మాణం. తెలుసా, - ఈ పర్సన్ ఫుడ్ యొక్క భౌతికశాస్త్రము గుండె. ఉత్తర దేశాన్ని పరిశీలించి",
            },
        },
    ]

    policy_model = get_model(config=config)
    reference_model = get_model(config=config)

    dpo = DirectPreferenceOptimization(
        policy_model=policy_model,
        reference_model=reference_model,
        tokenizer=tokenizer,
        dataset=dpo_dataset,
    )

    dpo(epochs=3)

    print("YES")

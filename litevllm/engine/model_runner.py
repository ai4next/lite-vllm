from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from litevllm.config import Config
from litevllm.engine.sequence import Sequence


class ModelRunner:
    def __init__(self, config: Config, tokenizer: AutoTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.device = torch.device(config.device or "cpu")
        kwargs = {"trust_remote_code": config.trust_remote_code}
        if config.dtype != "auto":
            kwargs["torch_dtype"] = getattr(torch, config.dtype)
        elif self.device.type == "cuda":
            kwargs["torch_dtype"] = torch.float16
        self.model = AutoModelForCausalLM.from_pretrained(config.model, **kwargs).to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        del is_prefill  # This lite runner recomputes the full context every step.
        input_ids, attention_mask = self._batch_inputs(seqs)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        lengths = attention_mask.sum(dim=1) - 1
        logits = outputs.logits[torch.arange(len(seqs), device=self.device), lengths]
        return self._sample(logits, seqs).tolist()

    def _batch_inputs(self, seqs: list[Sequence]) -> tuple[torch.Tensor, torch.Tensor]:
        max_len = max(len(seq) for seq in seqs)
        if max_len > self.config.max_model_len:
            raise ValueError(f"sequence length {max_len} exceeds max_model_len {self.config.max_model_len}")
        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id or 0
        input_ids = torch.full((len(seqs), max_len), pad_id, dtype=torch.long, device=self.device)
        attention_mask = torch.zeros((len(seqs), max_len), dtype=torch.long, device=self.device)
        for row, seq in enumerate(seqs):
            ids = torch.tensor(seq.token_ids, dtype=torch.long, device=self.device)
            input_ids[row, : ids.numel()] = ids
            attention_mask[row, : ids.numel()] = 1
        return input_ids, attention_mask

    def _sample(self, logits: torch.Tensor, seqs: list[Sequence]) -> torch.Tensor:
        temperatures = torch.tensor([seq.temperature for seq in seqs], dtype=torch.float32, device=logits.device)
        greedy = temperatures == 0
        safe_temperatures = temperatures.clamp_min(1e-6).unsqueeze(1)
        scores = logits.float() / safe_temperatures
        scores = self._apply_top_k_top_p(scores, seqs)
        probs = torch.softmax(scores, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
        greedy_ids = logits.argmax(dim=-1)
        return torch.where(greedy, greedy_ids, sampled)

    @staticmethod
    def _apply_top_k_top_p(scores: torch.Tensor, seqs: list[Sequence]) -> torch.Tensor:
        filtered = scores.clone()
        for row, seq in enumerate(seqs):
            row_scores = filtered[row]
            if seq.top_k > 0 and seq.top_k < row_scores.numel():
                kth = torch.topk(row_scores, seq.top_k).values[-1]
                row_scores[row_scores < kth] = -float("inf")
            if seq.top_p < 1.0:
                sorted_scores, sorted_idx = torch.sort(row_scores, descending=True)
                sorted_probs = torch.softmax(sorted_scores, dim=-1)
                remove = torch.cumsum(sorted_probs, dim=-1) > seq.top_p
                remove[1:] = remove[:-1].clone()
                remove[0] = False
                row_scores[sorted_idx[remove]] = -float("inf")
        return filtered

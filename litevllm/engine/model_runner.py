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
        try:
            self.model = AutoModelForCausalLM.from_pretrained(config.model, **kwargs).to(self.device)
        except OSError as exc:
            raise RuntimeError(
                f"Failed to load model weights from '{config.model}'. "
                "Verify model.safetensors / pytorch_model.bin are present."
            ) from exc
        self.model.eval()

    @torch.inference_mode()
    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        del is_prefill  # This lite runner recomputes the full context every step.
        input_ids, attention_mask = self._batch_inputs(seqs)
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        lengths = attention_mask.sum(dim=1) - 1
        logits = outputs.logits[torch.arange(len(seqs), device=self.device), lengths]
        return self._sample(logits, seqs).tolist()

    def _sample(self, logits: torch.Tensor, seqs: list[Sequence]) -> torch.Tensor:
        temperatures = torch.tensor(
            [seq.temperature for seq in seqs],
            dtype=torch.float32, device=logits.device,
        ).unsqueeze(1)
        greedy = temperatures.squeeze(1) == 0
        safe_temperatures = temperatures.clamp_min(1e-6)
        scores = logits.float() / safe_temperatures
        scores = self._apply_top_k_top_p(scores, seqs)
        greedy_ids = logits.argmax(dim=-1)
        if greedy.all():
            return greedy_ids
        probs = torch.softmax(scores, dim=-1)
        sampled = torch.multinomial(probs, num_samples=1).squeeze(1)
        return torch.where(greedy, greedy_ids, sampled)

    @staticmethod
    def _apply_top_k_top_p(scores: torch.Tensor, seqs: list[Sequence]) -> torch.Tensor:
        B, V = scores.shape
        device = scores.device
        sorted_scores, sorted_idx = scores.sort(descending=True, dim=-1)
        top_ks = torch.tensor(
            [max(0, min(seq.top_k, V)) for seq in seqs],
            dtype=torch.long, device=device,
        ).unsqueeze(1)
        top_ps = torch.tensor(
            [seq.top_p for seq in seqs],
            dtype=torch.float32, device=device,
        ).unsqueeze(1)
        positions = torch.arange(V, device=device).unsqueeze(0)
        topk_valid = top_ks > 0
        topk_mask = (positions >= top_ks) & topk_valid
        sorted_scores.masked_fill_(topk_mask, float("-inf"))
        topp_valid = top_ps < 1.0
        if topp_valid.any():
            probs = torch.softmax(sorted_scores, dim=-1)
            cumsum = torch.cumsum(probs, dim=-1)
            topp_mask = (cumsum > top_ps) & (positions > 0) & topp_valid
            sorted_scores.masked_fill_(topp_mask, float("-inf"))
        result = torch.zeros_like(scores)
        result.scatter_(1, sorted_idx, sorted_scores)
        return result

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


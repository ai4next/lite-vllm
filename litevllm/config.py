from dataclasses import dataclass

import torch
from transformers import AutoConfig


@dataclass(slots=True)
class Config:
    model: str
    max_num_seqs: int = 16
    max_model_len: int = 4096
    device: str | None = None
    dtype: str = "auto"
    trust_remote_code: bool = True
    eos: int = -1
    hf_config: AutoConfig | None = None

    def __post_init__(self) -> None:
        if self.max_num_seqs < 1:
            raise ValueError("max_num_seqs must be >= 1")
        if self.max_model_len < 1:
            raise ValueError("max_model_len must be >= 1")
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.hf_config = AutoConfig.from_pretrained(
            self.model,
            trust_remote_code=self.trust_remote_code,
        )
        max_positions = getattr(self.hf_config, "max_position_embeddings", None)
        if max_positions is not None:
            self.max_model_len = min(self.max_model_len, int(max_positions))

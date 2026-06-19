from dataclasses import dataclass, field


@dataclass(slots=True)
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    top_p: float = 1.0
    top_k: int = 0
    repetition_penalty: float = 1.0
    seed: int | None = None
    ignore_eos: bool = False
    stop_strings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be >= 0")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        for s in self.stop_strings:
            if not s:
                raise ValueError("stop_strings cannot contain empty strings")

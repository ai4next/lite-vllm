from copy import copy
from enum import Enum, auto
from itertools import count

from litevllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams,
        stop_string_ids: list[list[int]] | None = None,
    ):
        if not token_ids:
            raise ValueError("prompt must contain at least one token")
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.top_p = sampling_params.top_p
        self.top_k = sampling_params.top_k
        self.ignore_eos = sampling_params.ignore_eos
        self.stop_string_ids: list[list[int]] = stop_string_ids or []

    def __len__(self) -> int:
        return len(self.token_ids)

    @property
    def last_token(self) -> int:
        return self.token_ids[-1]

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self) -> int:
        return len(self.token_ids) - self.num_prompt_tokens

    @property
    def prompt_token_ids(self) -> list[int]:
        return self.token_ids[: self.num_prompt_tokens]

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.num_prompt_tokens :]

    def append_token(self, token_id: int) -> None:
        self.token_ids.append(int(token_id))

    def check_stop(self) -> bool:
        """Return True if completion ends with any configured stop string."""
        if not self.stop_string_ids:
            return False
        completion = self.completion_token_ids
        for stop_ids in self.stop_string_ids:
            if len(stop_ids) == 0:
                continue
            if len(completion) < len(stop_ids):
                continue
            if completion[-len(stop_ids) :] == stop_ids:
                return True
        return False

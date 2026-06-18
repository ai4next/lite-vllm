from collections import deque
from dataclasses import dataclass

from litevllm.config import Config
from litevllm.engine.sequence import Sequence, SequenceStatus


@dataclass
class ScheduleResult:
    prefill: list[Sequence]
    decode: list[Sequence]


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.eos = config.eos
        self.waiting: deque[Sequence] = deque()
        self.running: list[Sequence] = []

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def schedule(self) -> ScheduleResult:
        """Return prefill and decode groups. Both can be non-empty (continuous batching)."""
        self.running = [s for s in self.running if s.status == SequenceStatus.RUNNING]
        prefill: list[Sequence] = []
        decode: list[Sequence] = []

        # Phase 1: fill prefill slots from waiting queue
        while self.waiting and len(prefill) < self.max_num_seqs:
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            prefill.append(seq)

        # Phase 2: fill remaining decode slots from already-running seqs
        remaining = self.max_num_seqs - len(prefill)
        if remaining > 0 and self.running:
            decode = list(self.running)[:remaining]

        return ScheduleResult(prefill=prefill, decode=decode)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[Sequence]:
        finished: list[Sequence] = []
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            hit_eos = token_id == self.eos and not seq.ignore_eos
            hit_limit = seq.num_completion_tokens >= seq.max_tokens
            hit_stop = seq.check_stop()
            if hit_eos or hit_limit or hit_stop:
                seq.status = SequenceStatus.FINISHED
                finished.append(seq)
        return finished

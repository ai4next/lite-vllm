from collections import deque

from litevllm.config import Config
from litevllm.engine.sequence import Sequence, SequenceStatus


class Scheduler:
    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.eos = config.eos
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

    def add(self, seq: Sequence) -> None:
        self.waiting.append(seq)

    def is_finished(self) -> bool:
        return not self.waiting and not self.running

    def schedule(self) -> tuple[list[Sequence], bool]:
        prefill: list[Sequence] = []
        while self.waiting and len(self.running) + len(prefill) < self.max_num_seqs:
            seq = self.waiting.popleft()
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            prefill.append(seq)
        if prefill:
            return prefill, True
        return list(self.running)[: self.max_num_seqs], False

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> None:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            hit_eos = token_id == self.eos and not seq.ignore_eos
            hit_limit = seq.num_completion_tokens >= seq.max_tokens
            if hit_eos or hit_limit:
                seq.status = SequenceStatus.FINISHED
                self.running.remove(seq)

import warnings
from dataclasses import fields
from time import perf_counter

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from litevllm.config import Config
from litevllm.engine.model_runner import ModelRunner
from litevllm.engine.scheduler import ScheduleResult, Scheduler
from litevllm.engine.sequence import Sequence, SequenceStatus
from litevllm.sampling_params import SamplingParams


class LLMEngine:
    def __init__(self, model: str, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        known = {k: v for k, v in kwargs.items() if k in config_fields}
        unknown = [k for k in kwargs if k not in config_fields]
        if unknown:
            warnings.warn(
                f"LLMEngine: ignoring unknown kwargs {unknown}; "
                f"valid Config fields are {sorted(config_fields)}",
                stacklevel=2,
            )
        self.config = Config(model, **known)
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model,
                use_fast=True,
                trust_remote_code=self.config.trust_remote_code,
            )
        except OSError as exc:
            raise RuntimeError(
                f"Failed to load tokenizer from '{self.config.model}'. "
                "Ensure the directory contains tokenizer.json / tokenizer.model."
            ) from exc
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config.eos = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else -1
        self.scheduler = Scheduler(self.config)
        self.model_runner = ModelRunner(self.config, self.tokenizer)

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> None:
        token_ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
        stop_ids = (
            [self.tokenizer.encode(s, add_special_tokens=False) for s in sampling_params.stop_strings]
            if sampling_params.stop_strings
            else []
        )
        self.scheduler.add(Sequence(token_ids, sampling_params, stop_string_ids=stop_ids))

    def step(self) -> list[tuple[int, list[int], list[float]]]:
        result: ScheduleResult = self.scheduler.schedule()
        finished: list[Sequence] = []

        if result.prefill:
            token_ids, logprobs = self.model_runner.run(result.prefill, is_prefill=True)
            finished.extend(self.scheduler.postprocess(result.prefill, token_ids, logprobs))

        decode_seqs = [
            s for s in self.scheduler.running
            if s.status == SequenceStatus.RUNNING
        ]
        if decode_seqs:
            token_ids, logprobs = self.model_runner.run(decode_seqs, is_prefill=False)
            finished.extend(self.scheduler.postprocess(decode_seqs, token_ids, logprobs))

        return [
            (seq.seq_id, seq.completion_token_ids, seq.logprobs)
            for seq in finished
        ]

    def is_finished(self) -> bool:
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams] | None = None,
        use_tqdm: bool = True,
    ) -> list[dict[str, str | list[int]]]:
        if sampling_params is None:
            sampling_params = SamplingParams()
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        if len(prompts) != len(sampling_params):
            raise ValueError("prompts and sampling_params must have the same length")

        for prompt, params in zip(prompts, sampling_params):
            self.add_request(prompt, params)

        outputs: dict[int, dict[str, list[int] | list[float]]] = {}
        total_tokens = 0
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        started_all = perf_counter()
        while not self.is_finished():
            step_started = perf_counter()
            finished = self.step()
            total_tokens += len(finished) or 1
            for seq_id, token_ids, logprobs in finished:
                outputs[seq_id] = {"token_ids": token_ids, "logprobs": logprobs}
                pbar.update(1)
            pbar.set_postfix_str(
                f"step_ms={1000 * (perf_counter() - step_started):.1f} "
                f"tok/s={total_tokens / max(perf_counter() - started_all, 1e-6):.1f}"
            )
        pbar.close()

        return [
            {
                "text": self.tokenizer.decode(outputs[seq_id]["token_ids"], skip_special_tokens=True),
                "token_ids": outputs[seq_id]["token_ids"],
                "logprobs": outputs[seq_id]["logprobs"],
            }
            for seq_id in sorted(outputs)
        ]

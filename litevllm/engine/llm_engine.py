from dataclasses import fields
from time import perf_counter

from tqdm.auto import tqdm
from transformers import AutoTokenizer

from litevllm.config import Config
from litevllm.engine.model_runner import ModelRunner
from litevllm.engine.scheduler import Scheduler
from litevllm.engine.sequence import Sequence
from litevllm.sampling_params import SamplingParams


class LLMEngine:
    def __init__(self, model: str, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {key: value for key, value in kwargs.items() if key in config_fields}
        self.config = Config(model, **config_kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model,
            use_fast=True,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config.eos = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else -1
        self.scheduler = Scheduler(self.config)
        self.model_runner = ModelRunner(self.config, self.tokenizer)

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> None:
        token_ids = self.tokenizer.encode(prompt) if isinstance(prompt, str) else prompt
        self.scheduler.add(Sequence(token_ids, sampling_params))

    def step(self) -> list[tuple[int, list[int]]]:
        seqs, is_prefill = self.scheduler.schedule()
        token_ids = self.model_runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids)
        return [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]

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

        outputs: dict[int, list[int]] = {}
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        while not self.is_finished():
            started = perf_counter()
            finished = self.step()
            for seq_id, token_ids in finished:
                outputs[seq_id] = token_ids
                pbar.update(1)
            pbar.set_postfix({"step_s": f"{perf_counter() - started:.3f}"})
        pbar.close()

        return [
            {"text": self.tokenizer.decode(outputs[seq_id], skip_special_tokens=True), "token_ids": outputs[seq_id]}
            for seq_id in sorted(outputs)
        ]

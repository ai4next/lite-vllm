# lite-vLLM

A minimal, educational offline inference engine with a compact `LLM.generate` API.

This project intentionally keeps the runtime small and readable:

- no paged attention
- no KV-cache block manager
- no tensor parallelism
- no CUDA graph capture
- no custom model implementation

Instead, `lite-vllm` focuses on the smallest useful architecture: request objects, a tiny scheduler, a model runner, and an `LLM.generate` API built on Hugging Face `AutoModelForCausalLM`.

## Install

```bash
cd lite-vllm
pip install -e .
```

## Usage

```python
from litevllm import LLM, SamplingParams

llm = LLM("/path/to/local/causal-lm", device="cuda", max_num_seqs=8)
outputs = llm.generate(
    ["Hello, lite-vLLM."],
    SamplingParams(temperature=0.7, top_p=0.9, max_tokens=64),
)
print(outputs[0]["text"])
```

## Layout

```text
litevllm/
  llm.py                  # public LLM class
  sampling_params.py      # sampling options
  config.py               # runtime/model config
  engine/
    llm_engine.py         # request lifecycle and generate loop
    scheduler.py          # tiny waiting/running scheduler
    sequence.py           # per-request state
    model_runner.py       # Hugging Face forward + sampling
```

## Design Notes

This implementation is deliberately simple: every decode step recomputes the full context for each active sequence. That makes it slower than optimized inference runtimes, but easier to read and useful as a baseline before adding cache-aware decoding and batching optimizations.

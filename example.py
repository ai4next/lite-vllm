from litevllm import LLM, SamplingParams


# Replace this with a local Hugging Face causal-LM directory, for example:
# model_path = "/home/me/huggingface/Qwen3.5-0.8B"
model_path = "./model"

llm = LLM(model_path, device="cuda", max_num_seqs=4)
outputs = llm.generate(
    ["Hello, lite-vLLM."],
    SamplingParams(temperature=0.7, top_p=0.9, max_tokens=64),
)
print(outputs[0]["text"])

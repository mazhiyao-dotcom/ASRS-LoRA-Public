import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "models/Qwen3-1.7B"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading Qwen3-1.7B...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    device_map="cuda"
)

print("Model loaded successfully!")
print("GPU:", torch.cuda.get_device_name(0))

messages = [
    {
        "role": "user",
        "content": (
            "Answer only YES or NO.\n"
            "Does this aviation event involve Distraction?\n"
            "Report: During landing, the pilot was distracted "
            "by a warning message and failed to notice "
            "the altitude change."
        )
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=False,
    return_tensors="pt",
    return_dict=True
).to("cuda")

with torch.no_grad():
    outputs = model.generate(
    **inputs,
    max_new_tokens=10,
    do_sample=False,
    pad_token_id=tokenizer.eos_token_id
)

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[-1]:],
    skip_special_tokens=True
)

print("\nModel response:", response)
print(
    "GPU memory allocated:",
    round(torch.cuda.memory_allocated() / 1024**3, 2),
    "GB"
)
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Phi-3-mini - much smarter model (3.8B parameters)
model_name = "microsoft/phi-1_5"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

device = torch.device("cpu")
model.to(device)

prompt = "A nice place to ski in the United States is"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

output_ids = model.generate(
    **inputs,
    max_new_tokens=50,
    temperature=0.5,
    do_sample=True,
    top_k=50,
    top_p=0.9,
    repetition_penalty=1.1,
    pad_token_id=tokenizer.eos_token_id
)

output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)
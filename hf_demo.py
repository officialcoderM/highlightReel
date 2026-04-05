from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load tokenizer and model
model_name = "EleutherAI/gpt-neo-125m" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Add a padding token if missing (GPT‑2 doesn't have one by default)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Move model to the best available device (CPU, MPS for Apple Silicon, or CUDA)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
model.to(device)
# Prepare input text
prompt = "A nice place to ski is "
inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(device)

# Generate output
output_ids = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=0.5,
    top_k=25,   # only sample from top 50 tokens
    top_p=0.99, # nucleus sampling
    pad_token_id=tokenizer.pad_token_id
)

# Decode and print
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(output_text)


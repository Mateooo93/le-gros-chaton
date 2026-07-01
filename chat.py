from tokenizer import decode, encode, VOCAB_SIZE
import torch
from model import GPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model ONCE, before the loop. (Before it was reloading every message!)
model = GPT().to(device)
model.load_state_dict(torch.load("model.pt"))
model.eval()

print("Model loaded. Type 'quit' to exit.\n")

while True:
    prompt = input("prompt: ")
    if prompt.strip().lower() in ("quit", "exit"):
        break
    if not prompt.strip():
        continue

    # Encode the prompt into a (1, len) tensor of token IDs on the GPU.
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    # Generate a continuation. The sampler now has repetition_penalty (kills
    # the 'Singapore Singapore' loops a small model falls into) + top_p nucleus.
    out = model.generate(idx, max_new_tokens=100, temperature=0.8, top_k=50,
                         top_p=0.9, repetition_penalty=1.2)

    # decode turns the full token sequence (prompt + new) back into text.
    print(decode(out[0].tolist()))
    print("-" * 40)
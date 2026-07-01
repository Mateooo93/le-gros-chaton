from tokenizer import decode, encode, VOCAB_SIZE
import torch
from model import GPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = GPT().to(device)
model.load_state_dict(torch.load("model_finetuned.pt"))
model.eval()

# inference prompt format MUST match the fine-tuning template.
import os
INSTRUCT_SOURCE = os.environ.get("CHATON_INSTRUCT", "local")
HUMAN = "### Human:"
ASSIST = "### Assistant:"
EOT_ID = 50256  # gpt2 tiktoken EOT; fine for the local/dolly default tokenizer

print("Fine-tuned model loaded (format source:", INSTRUCT_SOURCE, "). Type 'quit' to exit.\n")

while True:
    question = input("you: ")
    if question.strip().lower() in ("quit", "exit"):
        break
    if not question.strip():
        continue

    if INSTRUCT_SOURCE == "dolly":
        prompt = f"{HUMAN} {question}\n{ASSIST}"
    else:
        prompt = f"Q: {question}\nA:"
    idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)

    out = model.generate(idx, max_new_tokens=80, temperature=0.4, top_k=50,
                         top_p=0.9, repetition_penalty=1.15)
    new_tokens = out[0].tolist()[len(idx[0]):]
    if EOT_ID in new_tokens:
        new_tokens = new_tokens[:new_tokens.index(EOT_ID)]
    answer = decode(new_tokens).strip()
    print(f"chaton: {answer}")
    print("-" * 40)
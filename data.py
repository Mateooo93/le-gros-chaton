import json
import torch
from tokenizer import encode, decode, VOCAB_SIZE, EOT_TOKEN
from datasets import load_dataset



ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")   # Dataset
texts = [t for t in ds["text"] if t.strip()]                          # list of strings, empties removed
raw = "\n\n".join(texts)                                              # one big string
data_encoded = encode(raw)

data_tensor = torch.tensor(data_encoded, dtype=torch.long)

# Move the whole dataset to the GPU ONCE. Then get_batch slices it on-device,
# so the training loop never has to copy batches across the CPU<->GPU bus.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
data_tensor = data_tensor.to(device)

# --- Train / validation split ---
# Hold out the last 10% as validation: we train on the first 90% and check the
# loss on the held-out 10% to tell real learning from memorization.
split_point = int(0.9 * len(data_tensor))
train_data = data_tensor[:split_point]
val_data = data_tensor[split_point:]


def get_batch(split, batch_size, block_size):
    """Grab a random batch of (input, target) windows from train or val data."""
    data = train_data if split == "train" else val_data
    max_start = len(data) - block_size - 1
    # starts must be on the same device as data, or GPU indexing breaks.
    starts = torch.randint(max_start, (batch_size,), device=device)
    x = [data[i : i + block_size] for i in starts]
    y = [data[i + 1 : i + block_size + 1] for i in starts]
    return torch.stack(x), torch.stack(y)



if __name__ == "__main__":
    print("total tokens:", len(data_tensor))
    print("train tokens:", len(train_data), "| val tokens:", len(val_data))
    x, y = get_batch("train", 2, 128)
    print("x shape:", x.shape, "| y shape:", y.shape)
    print("x[0] first 5:", x[0][:5])
    print("y[0] first 5:", y[0][:5])
    # y[0] should be x[0] shifted left by one
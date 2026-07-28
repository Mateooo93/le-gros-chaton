"""Fallback streaming data pipeline (lazy init).

Used only when ``data2.py`` cannot import (e.g. missing datasets dependency).
Downloads wikitext-2, encodes it to a GPU tensor once, then slices batches
from it.

Imports are side-effect free — preparation (download, tokenize, GPU upload)
happens lazily on the first call to ``get_batch``.
"""
import torch

try:
    from tokenizer import encode, VOCAB_SIZE, EOT_TOKEN
except ImportError as e:
    raise ImportError(
        "tokenizer dependencies not installed. "
        "Run: pip install tiktoken tokenizers"
    ) from e

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Lazy-initialised globals
_train_data: torch.Tensor | None = None
_val_data: torch.Tensor | None = None


def _prepare():
    """Download, tokenise, and split the corpus.

    Idempotent: subsequent calls return the already-built tensors.
    """
    global _train_data, _val_data  # noqa: PLW0603
    if _train_data is not None and _val_data is not None:
        return _train_data, _val_data

    from datasets import load_dataset
    print("[data] loading wikitext-2 (fallback)...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in ds["text"] if t.strip()]
    raw = "\n\n".join(texts)
    data_encoded = encode(raw)
    data_tensor = torch.tensor(data_encoded, dtype=torch.long, device=_device)

    split_point = int(0.9 * len(data_tensor))
    _train_data = data_tensor[:split_point]
    _val_data = data_tensor[split_point:]

    print(f"[data] {len(data_tensor)} total tokens → "
          f"{len(_train_data)} train / {len(_val_data)} val")
    return _train_data, _val_data


def get_batch(split: str, batch_size: int, block_size: int):
    """Random-window batch from the GPU-resident tensor.

    The first call triggers preparation (download, tokenize, upload).
    """
    train_data, val_data = _prepare()
    data = train_data if split == "train" else val_data
    max_start = len(data) - block_size - 1
    starts = torch.randint(max_start, (batch_size,), device=_device)
    x = torch.stack([data[i:i + block_size] for i in starts])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in starts])
    return x, y


if __name__ == "__main__":
    x, y = get_batch("train", 2, 128)
    print("x shape:", x.shape, "| y shape:", y.shape)
    print("x[0] first 5:", x[0][:5].tolist())
    print("y[0] first 5:", y[0][:5].tolist())
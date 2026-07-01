"""Streaming token data pipeline.

Encodes the corpus ONCE to a uint16 memmap on disk (~100M tokens = ~200MB),
then streams random windows from it. This lets us train on hundreds of
millions of tokens without holding the whole tensor in GPU/CPU memory.

Corpus is chosen by CORPUS below:
  "wikitext-2"   -> ~2M tokens, fast, good for smoke tests
  "wikitext-103" -> ~100M tokens, the real training corpus (use on a bigger GPU)
"""
import os
import numpy as np
import torch
from datasets import load_dataset
from tokenizer import encode, EOT_TOKEN, VOCAB_SIZE

CORPUS = os.environ.get("CHATON_CORPUS", "wikitext-2")   # override via env
BLOCK = 512  # must match config.block_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# namespace memmap files by (corpus, vocab_size) so switching tokenizer or
# corpus auto-rebuilds the cache — prevents the silent stale-id garbage bug.
_TAG = f"{CORPUS}_v{VOCAB_SIZE}"
TRAIN_BIN = f"train_tokens_{_TAG}.bin"
VAL_BIN = f"val_tokens_{_TAG}.bin"
VAL_GPU_TOKENS = 262144   # ~256k-token fixed val shard kept on GPU for fast eval


def _build_corpus_memmap(name):
    """Download + filter + tokenize one of the supported corpora to a single
    int token list and a length. Returns (np.array(int64 total_list?), ...) —
    actually returns a Python list of ints (caller writes the memmap)."""
    if name == "wikitext-2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif name == "wikitext-103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    else:
        raise ValueError(f"unknown corpus {name}")
    texts = [t for t in ds["text"] if t.strip()]
    raw = "\n\n".join(texts)
    return encode(raw)


def _prepare():
    """Build train/val uint16 memmaps if they don't exist. Returns the numpy
    memmap arrays for train and val."""
    if not (os.path.exists(TRAIN_BIN) and os.path.exists(VAL_BIN)):
        print(f"[data2] encoding corpus {CORPUS!r} -> memmap (one-time)...")
        tokens = _build_corpus_memmap(CORPUS)
        tokens = np.array(tokens, dtype=np.int64)
        # 90/10 split
        split = int(0.9 * len(tokens))
        train_tokens = tokens[:split].astype(np.uint16)
        val_tokens = tokens[split:].astype(np.uint16)
        train_tokens.tofile(TRAIN_BIN)
        val_tokens.tofile(VAL_BIN)
        print(f"[data2] {len(train_tokens):,} train | {len(val_tokens):,} val tokens written")
    else:
        print("[data2] memmap files already exist, skipping encode")
    train_mmap = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    val_mmap = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    print(f"[data2] train {len(train_mmap):,} | val {len(val_mmap):,} tokens (loaded as memmap)")
    return train_mmap, val_mmap


train_mmap, val_mmap = _prepare()

# Fixed val shard resident on GPU so eval is fast and reproducible across runs.
_val_len = min(VAL_GPU_TOKENS, len(val_mmap) - BLOCK - 1)
val_tensor_gpu = torch.from_numpy(np.array(val_mmap[:_val_len], dtype=np.int64)).to(device)


def get_batch(split, batch_size, block_size):
    """Random-window batch. Reads a slice from the memmap and uploads ONLY
    that slice to the GPU each step (so the big corpus never lives in VRAM)."""
    if split == "val":
        # deterministic, fast, GPU-resident shard
        max_start = val_tensor_gpu.size(0) - block_size - 1
        starts = torch.randint(max_start, (batch_size,), device=device)
        x = torch.stack([val_tensor_gpu[i:i + block_size] for i in starts])
        y = torch.stack([val_tensor_gpu[i + 1:i + block_size + 1] for i in starts])
        return x, y

    mmap = train_mmap
    max_start = len(mmap) - block_size - 1
    starts = np.random.randint(0, max_start, size=batch_size)
    xs = np.stack([mmap[s:s + block_size].astype(np.int64) for s in starts])
    ys = np.stack([mmap[s + 1:s + block_size + 1].astype(np.int64) for s in starts])
    return torch.from_numpy(xs).to(device), torch.from_numpy(ys).to(device)


if __name__ == "__main__":
    print("corpus:", CORPUS)
    x, y = get_batch("train", 4, BLOCK)
    print("train batch:", tuple(x.shape), "| dtype", x.dtype, "| device", x.device)
    xv, yv = get_batch("val", 4, BLOCK)
    print("val batch:", tuple(xv.shape))
    print("y[0][:5]:", y[0][:5].tolist(), "| x[0][:5]:", x[0][:5].tolist())
    print("(y[0] should equal x[0] shifted left by one)")
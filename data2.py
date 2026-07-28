"""Streaming token data pipeline (lazy init).

Encodes the corpus ONCE to a uint16 memmap on disk (~100M tokens = ~200MB),
then streams random windows from it. This lets us train on hundreds of
millions of tokens without holding the whole tensor in GPU/CPU memory.

Corpus is chosen by CORPUS below:
  "wikitext-2"   -> ~2M tokens, fast, good for smoke tests
  "wikitext-103" -> ~100M tokens, the real training corpus (use on a bigger GPU)

Imports are side-effect free — preparation (download, tokenize, memmap, GPU
upload) happens lazily on the first call to ``get_batch``.
"""
import os
import numpy as np
import torch
from datasets import load_dataset
from tokenizer import encode, EOT_TOKEN, VOCAB_SIZE

CORPUS = os.environ.get("CHATON_CORPUS", "wikitext-2")   # override via env
# BLOCK = the window size for the on-GPU val shard. Must match config.block_size.
# Read CHATON_BLOCK_SIZE so the T4 (block 512) and L4/A100 (block 1024/2048) runs
# each get a correctly-sized val shard instead of a hardcoded-512 one that would
# overflow for bigger blocks.
BLOCK = int(os.environ.get("CHATON_BLOCK_SIZE", "512"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# namespace memmap files by (corpus, vocab_size) so switching tokenizer or
# corpus auto-rebuilds the cache — prevents the silent stale-id garbage bug.
_TAG = f"{CORPUS}_v{VOCAB_SIZE}"
TRAIN_BIN = f"train_tokens_{_TAG}.bin"
VAL_BIN = f"val_tokens_{_TAG}.bin"
VAL_GPU_TOKENS = 262144   # ~256k-token fixed val shard kept on GPU for fast eval

# Lazy-initialised globals — set on first get_batch call.
_train_mmap: np.memmap | None = None
_val_mmap: np.memmap | None = None
_val_tensor_gpu: torch.Tensor | None = None


def _build_corpus_memmap(name: str) -> list[int]:
    """Download + filter + tokenize one of the supported corpora.

    Returns a flat Python list of token ids (caller writes the memmap).
    """
    if name == "wikitext-2":
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    elif name == "wikitext-103":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    else:
        raise ValueError(f"unknown corpus {name!r}; expected 'wikitext-2' or 'wikitext-103'")
    texts = [t for t in ds["text"] if t.strip()]
    raw = "\n\n".join(texts)
    return encode(raw)


def _prepare():
    """Build train/val uint16 memmaps if they don't exist.

    Returns (train_mmap, val_mmap) — numpy memmap arrays.  Idempotent:
    subsequent calls return the already-constructed memmaps.
    """
    global _train_mmap, _val_mmap, _val_tensor_gpu  # noqa: PLW0603
    if _train_mmap is not None and _val_mmap is not None:
        return _train_mmap, _val_mmap

    if not (os.path.exists(TRAIN_BIN) and os.path.exists(VAL_BIN)):
        print(f"[data2] encoding corpus {CORPUS!r} → memmap (one-time)...")
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
    _train_mmap = np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r")
    _val_mmap = np.memmap(VAL_BIN, dtype=np.uint16, mode="r")
    print(f"[data2] train {len(_train_mmap):,} | val {len(_val_mmap):,} tokens (loaded as memmap)")

    # Fixed val shard resident on GPU so eval is fast and reproducible.
    _val_len = min(VAL_GPU_TOKENS, len(_val_mmap) - BLOCK - 1)
    _val_tensor_gpu = torch.from_numpy(
        np.array(_val_mmap[:_val_len], dtype=np.int64)
    ).to(device)

    return _train_mmap, _val_mmap


def get_batch(split: str, batch_size: int, block_size: int):
    """Random-window batch. Reads a slice from the memmap and uploads ONLY
    that slice to the GPU each step (so the big corpus never lives in VRAM).

    The first call triggers preparation (download + tokenize + memmap + GPU
    upload).  Subsequent calls stream from the already-loaded memmaps.
    """
    train_mmap, val_mmap = _prepare()

    if split == "val":
        # deterministic, fast, GPU-resident shard
        max_start = _val_tensor_gpu.size(0) - block_size - 1
        starts = torch.randint(max_start, (batch_size,), device=device)
        x = torch.stack([_val_tensor_gpu[i:i + block_size] for i in starts])
        y = torch.stack([_val_tensor_gpu[i + 1:i + block_size + 1] for i in starts])
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
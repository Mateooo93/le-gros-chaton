"""Code-corpus streaming pipeline for "le fat chaton".

Same memmap idea as data2.py but for CODE corpora (Stack v2 / StarCoderData /
the-smol-corpus). Designed to run on a CLOUD VM (more disk than your 2070 box —
Stack v2 is huge, so we stream + sample, never pull the whole thing).

Pick the corpus with CHATON_CODE_CORPUS:
  "smollm-corpus"  -> HuggingFaceTB/smollm-corpus (a distilled/curated mix incl.
                      code + textbooks — the Phi route: information-dense data
                      that punches above its token count). RECOMMENDED for solo
                      budget: you want quality over raw quantity.
  "stack-v2"       -> bigcode/the-stack-v2 (raw code, large, disk-heavy)
  "starcoderdata"  -> bigcode/starcoderdata (older, very large)

Like data2.py: encodes once to a uint16 memmap, then streams random windows.
For a coding model the vocab stays 50257 (gpt2 BPE) — code tokenizes fine with it,
and keeping the vocab shared with the WikiText base lets you continue-pretrain
on top of an existing checkpoint if you want.
(For the BIG fat model you train from scratch, so no checkpoint reuse needed.)
"""
import os
import numpy as np
import torch

CORPUS = os.environ.get("CHATON_CODE_CORPUS", "smollm-corpus")
BLOCK = int(os.environ.get("CHATON_CODE_BLOCK", "4096"))   # match config.block_size

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# memmap files namespaced by corpus + vocab (switching corpus rebuilds)
from tokenizer import VOCAB_SIZE
_TAG = f"{CORPUS}_v{VOCAB_SIZE}"
TRAIN_BIN = f"code_train_{_TAG}.bin"
VAL_BIN = f"code_val_{_TAG}.bin"
VAL_GPU_TOKENS = 262144   # fixed val shard on GPU for fast eval


def _stream_text(name, max_docs=None):
    """Yield text strings from the chosen corpus. Uses HuggingFace datasets.
    max_docs caps how many docs to pull — keeps the memmap build bounded on
    a free-tier VM with limited disk/time."""
    from datasets import load_dataset
    if name == "smollm-corpus":
        # smollm-corpus has several subsets; "smol-edit" / "cosmopedia" / etc.
        # For code focus use the python/normalized-code style subset.
        ds = load_dataset("HuggingFaceTB/smollm-corpus", "smol-edu",
                          split="train", streaming=True)
        for i, row in enumerate(ds):
            if max_docs and i >= max_docs:
                break
            yield row.get("text") or row.get("content") or ""
    elif name == "stack-v2":
        ds = load_dataset("bigcode/the-stack-v2", split="train", streaming=True)
        for i, row in enumerate(ds):
            if max_docs and i >= max_docs:
                break
            yield row.get("content") or ""
    elif name == "starcoderdata":
        ds = load_dataset("bigcode/starcoderdata", split="train", streaming=True)
        for i, row in enumerate(ds):
            if max_docs and i >= max_docs:
                break
            yield row.get("content") or ""
    else:
        raise ValueError(f"unknown code corpus {name}")


def _prepare(max_docs):
    """Encode a bounded number of corpus docs to a uint16 memmap. One-time per VM."""
    from tokenizer import encode, EOT_TOKEN
    if not (os.path.exists(TRAIN_BIN) and os.path.exists(VAL_BIN)):
        max_tokens = int(os.environ.get("CHATON_CODE_MAX_TOKENS", str(50_000_000)))
        print(f"[code] encoding corpus {CORPUS!r} (cap {max_tokens:,} tokens)...")
        all_ids = []
        total = 0
        for text in _stream_text(CORPUS, max_docs=max_docs):
            if not text or not text.strip():
                continue
            ids = encode(text) + [EOT_TOKEN]   # EOT between docs
            all_ids.extend(ids)
            total += len(ids)
            if total >= max_tokens:
                break
        if total == 0:
            raise RuntimeError(f"no tokens encoded from {CORPUS} — check dataset name/access")
        arr = np.array(all_ids[:total], dtype=np.int64)
        split = int(0.98 * len(arr))   # tiny val (code val is less meaningful)
        train = arr[:split].astype(np.uint16)
        val = arr[split:].astype(np.uint16)
        train.tofile(TRAIN_BIN); val.tofile(VAL_BIN)
        print(f"[code] {len(train):,} train | {len(val):,} val tokens written")
    else:
        print("[code] memmap files already exist, skipping encode")
    return (np.memmap(TRAIN_BIN, dtype=np.uint16, mode="r"),
            np.memmap(VAL_BIN, dtype=np.uint16, mode="r"))


train_mmap, val_mmap = _prepare(max_docs=int(os.environ.get("CHATON_CODE_MAX_DOCS", "200000")))
_val_len = min(VAL_GPU_TOKENS, len(val_mmap) - BLOCK - 1)
val_tensor_gpu = torch.from_numpy(np.array(val_mmap[:_val_len], dtype=np.int64)).to(device)


def get_batch(split, batch_size, block_size):
    if split == "val":
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
    print("corpus:", CORPUS, "| block:", BLOCK)
    x, y = get_batch("train", 2, min(BLOCK, 512))
    print("train batch:", tuple(x.shape), "| device", x.device)
    from tokenizer import decode
    print("sample text:", decode(x[0][:40].tolist())[:200])
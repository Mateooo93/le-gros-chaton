"""CAPSTONE (STEP 8): train a custom byte-level BPE with a SMALL vocab.

Why: the GPT-2 tokenizer has 50,257 tokens. The embedding table for that is
50257 x n_embd params, which at n_embd=256 is ~12.8M params = 73% of the whole
model — a giant lookup table, not a reasoning brain. Shrinking the vocab to
~16k cuts the embedding to ~25% of params, freeing capacity to widen the body
(n_embd 256->384, n_layer 6->8 -> ~26M params) where actual knowledge lives.

This script trains that tokenizer ONCE on a sample of the corpus and saves:
  custom_bpe-vocab.json + custom_bpe-merges.txt   (HuggingFace tokenizers format)

Run: python train_custom_tokenizer.py
"""
import os
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

VOCAB_SIZE = 16384
OUT_DIR = "."
CORPUS_SAMPLE = "corpus_sample.txt"   # text sample to train the tokenizer on


def main():
    # Write a corpus sample from the wikitext memmap source (reuse the filter).
    from data2 import _build_corpus_memmap, CORPUS
    print(f"[tok] sampling corpus {CORPUS!r} for tokenizer training...")
    text = _build_corpus_memmap(CORPUS)
    # tokenizers trains on a file; write a sample (cap ~50MB to keep it quick)
    sample = text
    with open(CORPUS_SAMPLE, "w", encoding="utf-8") as f:
        f.write(sample)
    print(f"[tok] wrote {len(sample):,} chars to {CORPUS_SAMPLE}")

    tokenizer = Tokenizer(models.BPE(unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["<|endoftext|>", "<|unk|>"],
        show_progress=True,
    )
    tokenizer.train([CORPUS_SAMPLE], trainer)

    vocab_path = os.path.join(OUT_DIR, "custom_bpe-vocab.json")
    merges_path = os.path.join(OUT_DIR, "custom_bpe-merges.txt")
    tokenizer.save(OUT_DIR, "custom_bpe")
    print(f"[tok] saved {vocab_path} + {merges_path}")
    print(f"[tok] vocab size: {tokenizer.get_vocab_size()}")

    # quick sanity: round-trip
    enc = tokenizer.encode("The nature reserves of Singapore")
    print("[tok] encode sample:", enc.ids)
    print("[tok] decode sample:", tokenizer.decode(enc.ids))
    print("[tok] EOT token id:",
          tokenizer.token_to_id("<|endoftext|>"))


if __name__ == "__main__":
    main()
"""Tokenizer wrapper.

Two modes, chosen by USE_CUSTOM_BPE:
  False (default) -> GPT-2 tiktoken BPE (vocab 50,257)
  True             -> the custom 16k byte-level BPE from train_custom_tokenizer.py
                     (only available after you run that script + the capstone)

Keeping one interface (encode/decode/VOCAB_SIZE/EOT_TOKEN) so the rest of the
code never has to care which tokenizer is active.
"""
import os
from tokenizers import Tokenizer

USE_CUSTOM_BPE = os.environ.get("CHATON_CUSTOM_BPE", "0") == "1"

if USE_CUSTOM_BPE:
    _tok = Tokenizer.from_file("custom_bpe.json")
    VOCAB_SIZE = _tok.get_vocab_size()
    _eot_id = _tok.token_to_id("")
    EOT_TOKEN = _eot_id if _eot_id is not None else 0

    def encode(text):
        return _tok.encode(text).ids

    def decode(ids):
        return _tok.decode(ids)

else:
    import tiktoken
    _enc = tiktoken.get_encoding("gpt2")
    VOCAB_SIZE = _enc.n_vocab          # 50257
    EOT_TOKEN = _enc.eot_token         # 50256 (the  token)

    def encode(text):
        return _enc.encode(text, allowed_special={""})

    def decode(ids):
        return _enc.decode(ids)


if __name__ == "__main__":
    print("USE_CUSTOM_BPE:", USE_CUSTOM_BPE)
    print("Vocab size:", VOCAB_SIZE)
    print("EOT token id:", EOT_TOKEN)
    s = "Hello, world! This is Chaton."
    ids = encode(s)
    print("Encoded:", ids)
    print("Round-trip OK:", decode(ids) == s)
    print("Tokens:", [decode([i]) for i in ids])
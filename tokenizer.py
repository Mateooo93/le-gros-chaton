"""Tokenizer wrapper.

Two modes, chosen by USE_CUSTOM_BPE:
  False (default) -> GPT-2 tiktoken BPE (vocab 50,257)
  True             -> custom 16k byte-level BPE (run the training script first).

Keeping one interface (encode/decode/VOCAB_SIZE/EOT_TOKEN) so the rest of the
code never has to care which tokenizer is active.
"""
import os

USE_CUSTOM_BPE = os.environ.get("CHATON_CUSTOM_BPE", "0") == "1"

if USE_CUSTOM_BPE:
    from tokenizers import Tokenizer
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
        return _enc.encode(text, allowed_special={"<|endoftext|>"})

    def decode(ids):
        # Replace tool token IDs with sentinel values that tiktoken can handle,
        # then restore the string representations.
        _ensure_tool_tokens()
        result_parts: list[str] = []
        for tid in ids:
            if tid in _TOOL_DECODE_MAP:
                result_parts.append(_TOOL_DECODE_MAP[tid])
            else:
                result_parts.append(_enc.decode([tid]))
        return "".join(result_parts)


# --- Tool tokens ----------------------------------------------------------
# Tool token IDs sit above the base GPT-2 vocabulary.  ``decode()`` maps them
# back to their string representations; ``extend_vocab()`` in model.py extends
# the embedding table to accommodate them.

_TOOL_ENCODE_MAP: dict[str, int] = {}   # populated on first use
_TOOL_DECODE_MAP: dict[int, str] = {}   # populated on first use


def _ensure_tool_tokens():
    if not _TOOL_DECODE_MAP:
        _TOOL_DECODE_MAP.update({
            VOCAB_SIZE:     "<|tool_call|>",
            VOCAB_SIZE + 1: "<|tool_result|>",
            VOCAB_SIZE + 2: "<|done|>",
        })
        _TOOL_ENCODE_MAP.update({v: k for k, v in _TOOL_DECODE_MAP.items()})


def is_tool_token(token_id: int) -> bool:
    """True if *token_id* is one of our tool tokens."""
    _ensure_tool_tokens()
    return token_id in _TOOL_DECODE_MAP


def tool_token_name(token_id: int) -> str | None:
    """Return the string name of a tool token, or None."""
    _ensure_tool_tokens()
    return _TOOL_DECODE_MAP.get(token_id)


def tool_token_id(name: str) -> int | None:
    """Return the ID of a tool token by name, or None."""
    _ensure_tool_tokens()
    return _TOOL_ENCODE_MAP.get(name)


if __name__ == "__main__":
    print("USE_CUSTOM_BPE:", USE_CUSTOM_BPE)
    print("Vocab size:", VOCAB_SIZE)
    print("EOT token id:", EOT_TOKEN)
    s = "Hello, world! This is Chaton."
    ids = encode(s)
    print("Encoded:", ids)
    print("Round-trip OK:", decode(ids) == s)
    print("Tokens:", [decode([i]) for i in ids])
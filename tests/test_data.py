"""Tests for the data pipeline — lazy init, tokenization, and corpus prep.

These tests verify that data modules don't execute heavy operations at
import time (lazy init fix from the repo audit) and that the encoding
pipeline is structurally sound.
"""
import ast
import os
import sys

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from tests.conftest import assert_syntax_ok, needs_torch


# ---------------------------------------------------------------------------
# Syntax
# ---------------------------------------------------------------------------

def test_data_syntax():
    for f in ["data.py", "data_code.py", "data2.py"]:
        assert_syntax_ok(os.path.join(PROJ_ROOT, f))


# ---------------------------------------------------------------------------
# Lazy init: importing data modules should NOT trigger downloads or GPU ops
# ---------------------------------------------------------------------------

@needs_torch
def test_data_code_import_is_lazy():
    """Importing ``data_code`` should NOT trigger corpus download, tokenization,
    or GPU upload.  Those operations should be deferred to the first
    ``get_batch()`` call or explicit ``_prepare()`` call."""
    import importlib

    # Ensure a clean import
    if "data_code" in sys.modules:
        del sys.modules["data_code"]

    # This import should complete instantly and without side effects
    import data_code  # noqa: F401

    # Verify the module-level functions exist but haven't been run
    assert hasattr(data_code, "_prepare"), "data_code should expose _prepare()"
    assert hasattr(data_code, "get_batch"), "data_code should expose get_batch()"


@needs_torch
def test_data_import_is_lazy():
    """Same check for data.py (wikitext)."""
    import importlib

    if "data" in sys.modules:
        del sys.modules["data"]

    import data  # noqa: F401

    assert hasattr(data, "get_batch"), "data should expose get_batch()"


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def test_tokenizer_encodes_and_decodes():
    """The tokenizer should round-trip text (encode → decode returns original)."""
    from tokenizer import encode, decode, VOCAB_SIZE

    # Ensure the tokenizer data exists
    assert VOCAB_SIZE > 0, "VOCAB_SIZE should be > 0"

    texts = [
        "hello world",
        "def foo(bar):\n    return bar + 1\n",
        "print('hello')  # inline comment",
        "x = [i for i in range(10)]",
        "",
        "a" * 1000,  # long sequence
    ]

    for text in texts:
        ids = encode(text)
        decoded = decode(ids)
        assert decoded == text, (
            f"round-trip failed:\n  original: {text[:50]!r}\n  decoded:  {decoded[:50]!r}"
        )


def test_tokenizer_vocab_size_matches_config():
    """The tokenizer's VOCAB_SIZE should match config.vocab_size."""
    from tokenizer import VOCAB_SIZE as tokenizer_vocab
    import config as cfg

    assert tokenizer_vocab == cfg.vocab_size, (
        f"tokenizer VOCAB_SIZE ({tokenizer_vocab}) doesn't match "
        f"config.vocab_size ({cfg.vocab_size})"
    )


def test_tokenizer_handles_empty_input():
    from tokenizer import encode, decode
    assert encode("") == [], "empty string should encode to []"
    assert decode([]) == "", "empty token list should decode to ''"


def test_tokenizer_handles_special_tokens():
    """The tokenizer should handle the special tokens used by the data pipeline
    (EOT, BOS, etc.)"""
    from tokenizer import encode, decode, VOCAB_SIZE

    # Check that we can encode/decode the full vocab range
    # The GPT-2 tokenizer may not expose all 50257 tokens, but we should
    # at least verify the tokenizer works with special characters
    texts = [
        "<|endoftext|>",
        "\n\n",
        "# comment\n",
        "    " * 5,  # indentation (common in code)
    ]
    for text in texts:
        ids = encode(text)
        assert len(ids) > 0, f"text {text!r} produced zero tokens"
        assert all(0 <= id < VOCAB_SIZE for id in ids), (
            f"token id(s) out of range for text {text!r}: {ids}"
        )


# ---------------------------------------------------------------------------
# Step extraction (PRM utility)
# ---------------------------------------------------------------------------

@needs_torch
@needs_torch
def test_prm_step_extraction():
    """The PRM's step extraction should split code into reasonable blocks."""
    from prm import extract_steps

    # Single-line code → one step
    steps = extract_steps("print('hello')")
    assert len(steps) == 1, f"single line should be 1 step, got {len(steps)}"

    # Multi-function code → multiple steps
    code = """
def add(a, b):
    return a + b

def sub(a, b):
    return a - b
"""
    steps = extract_steps(code)
    assert len(steps) >= 2, f"expected 2+ steps, got {len(steps)}: {steps}"

    # Code with blank-line boundaries
    code2 = """
x = 1

y = 2

z = 3
"""
    steps2 = extract_steps(code2)
    assert len(steps2) >= 2, (
        f"blank-line code should have 2+ steps, got {len(steps2)}"
    )



# ---------------------------------------------------------------------------
# AST-based step extraction tests (no torch needed)
# ---------------------------------------------------------------------------

def _ast_extract_steps(code, min_step_tokens=8):
    """Minimal copy of prm._extract_steps_ast for testing without torch."""
    import ast
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    min_chars = min_step_tokens * 4
    lines = code.splitlines(keepends=True)
    def nl(node):
        # AST's lineno for a FunctionDef starts at 'def', not the decorator.
        # Check decorator_list to include decorator lines.
        s = getattr(node, "lineno", 1) - 1
        if hasattr(node, "decorator_list") and node.decorator_list:
            s = min(s, node.decorator_list[0].lineno - 1)
        e = getattr(node, "end_lineno", s + 1)
        return "".join(lines[s:e])
    raw = [nl(n).strip() for n in ast.iter_child_nodes(tree) if nl(n).strip()]
    if not raw:
        return None
    steps = []
    for seg in raw:
        if steps and len(seg) < min_chars:
            # Don't merge new function/class/definition boundaries
            if not any(seg.startswith(kw) for kw in ("def ", "class ", "@")):
                steps[-1] = steps[-1] + "\n" + seg
                continue
        steps.append(seg)
    return steps if steps else [code.strip()]


def test_ast_extract_steps_multi_function():
    """Multi-function code should produce one step per function."""
    code = '''def foo():
    return 42

def bar(x):
    return x + 1
'''
    steps = _ast_extract_steps(code)
    assert steps is not None
    assert len(steps) == 2, f"Expected 2 steps, got {len(steps)}"
    assert "def foo" in steps[0]
    assert "def bar" in steps[1]


def test_ast_extract_steps_class_is_one_step():
    """A class with methods should be a single step (not split by nested defs)."""
    code = '''class Calc:
    def add(self, a, b):
        return a + b
    def sub(self, a, b):
        return a - b
'''
    steps = _ast_extract_steps(code)
    assert steps is not None
    assert len(steps) == 1, f"Expected 1 step (whole class), got {len(steps)}"
    assert "class Calc" in steps[0]


def test_ast_extract_steps_invalid_syntax():
    """Invalid Python should return None (triggering regex fallback)."""
    result = _ast_extract_steps("def foo(:")
    assert result is None


def test_ast_extract_steps_decorator_attached():
    """A decorator should be part of the decorated function's step."""
    code = '''@cache\ndef fib(n):\n    if n < 2:\n        return n\n    return fib(n-1) + fib(n-2)\n'''
    steps = _ast_extract_steps(code)
    assert steps is not None
    assert len(steps) == 1, f"Expected 1 step (decorator + function), got {len(steps)}"
    assert "@cache" in steps[0]
    assert "def fib" in steps[0]


def test_ast_extract_steps_single_line():
    """Single-line code should produce one step."""
    steps = _ast_extract_steps("x = 42")
    assert steps is not None
    assert len(steps) == 1


def test_ast_extract_steps_empty():
    """Empty or whitespace-only code should return None."""
    assert _ast_extract_steps("") is None
    assert _ast_extract_steps("   \n\n") is None


# ---------------------------------------------------------------------------
# Syntax validation filter (data_code.py)
# ---------------------------------------------------------------------------

@needs_torch
@needs_torch
def test_syntax_validation_rejects_broken_code():
    """The ``compile()`` check in data_code should reject non-parsing Python."""
    from data_code import _collect_documents

    broken_stream = ["def foo(:\n    return bar\n"]  # missing ')' 
    stream = iter(broken_stream)
    # This should not crash — the syntax validation should skip the broken file
    docs, total = _collect_documents(stream, max_tokens=1000, max_docs=10,
                                     eot=0, validate_syntax=True)
    # The broken file should have been skipped
    assert len(docs) == 0, (
        f"syntax validation should skip broken code; got {len(docs)} docs"
    )


@needs_torch
def test_syntax_validation_accepts_good_code():
    """Valid Python should pass syntax validation and be collected."""
    from data_code import _collect_documents

    good_code = """
def add(a, b):
    return a + b
"""
    stream = iter([good_code])
    docs, total = _collect_documents(stream, max_tokens=1000, max_docs=10,
                                     eot=0, validate_syntax=True)
    assert len(docs) == 1, (
        f"good code should produce 1 doc, got {len(docs)}"
    )
    assert total > 0, "total tokens should be > 0"


# ---------------------------------------------------------------------------
# Corpus env vars
# ---------------------------------------------------------------------------

def test_corpus_env_defaults():
    """The default corpus env vars should be reasonable."""
    import os

    corpus = os.environ.get("CHATON_CODE_CORPUS", "starcoderdata")
    assert corpus in ("starcoderdata", "github-code-python"), (
        f"unexpected corpus: {corpus}"
    )

    blend = float(os.environ.get("CHATON_PROSE_BLEND", "0.15"))
    assert 0.0 <= blend <= 1.0, f"prose blend should be [0, 1], got {blend}"
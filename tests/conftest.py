"""Shared fixtures and helpers for the test suite.

Many tests need torch, which may not be importable on this dev VM
(libtorch_global_deps.so issue).  We mark those tests with a custom
``needs_torch`` marker and skip them gracefully when torch is absent.
"""
import pytest


def _torch_available() -> bool:
    try:
        import torch  # noqa: F401
        return True
    except Exception:
        return False


needs_torch = pytest.mark.skipif(
    not _torch_available(),
    reason="torch not available on this VM (CUDA lib issue)",
)


def assert_syntax_ok(path: str):
    """Assert that a Python file parses without syntax errors."""
    import ast
    with open(path) as f:
        ast.parse(f.read())
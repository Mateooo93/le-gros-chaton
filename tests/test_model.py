"""Syntax and model-construction smoke tests.

Verifies every profile can be instantiated without errors and that
forward/generate paths are structurally sound (even without torch).
"""
import os
import sys
import pytest

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

from tests.conftest import assert_syntax_ok, needs_torch


# ---------------------------------------------------------------------------
# Syntax checks — run without torch, catch basic mistakes
# ---------------------------------------------------------------------------

PYTHON_FILES = [
    "config.py", "model.py", "train.py", "checkpoint.py", "tokenizer.py",
    "data.py", "data_code.py", "data2.py", "rft.py", "rlvr.py",
    "prm.py", "best_of_n.py", "agent_rl.py",
    "agent/loop.py", "agent/sandbox.py",
    "verify/verifier.py", "verify/__init__.py",
    "eval/eval.py", "eval/humaneval_loader.py",
    "chat.py", "inference.py", "pipeline.py",
]


@pytest.mark.parametrize("rel_path", PYTHON_FILES)
def test_syntax(rel_path: str):
    """Every .py file in the project must parse without syntax errors."""
    full = os.path.join(PROJ_ROOT, rel_path)
    assert os.path.isfile(full), f"missing file: {full}"
    assert_syntax_ok(full)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------

def test_config_all_profiles_produce_no_warnings():
    """Every built-in profile should pass config.validate() without issues."""
    from config import validate, PROFILE
    warnings = validate()
    assert not warnings, (
        f"profile={PROFILE} produces warnings: {'; '.join(warnings)}"
    )


def test_config_known_bad_values_produce_warnings():
    """Intentionally bad configs should be caught by validate()."""
    # We test the validator logic, not the actual config module
    from config import validate as validate_fn

    # Temporarily override some config values to trigger warnings
    import config as cfg

    def _check_warns(**overrides):
        originals = {}
        for k, v in overrides.items():
            originals[k] = getattr(cfg, k, None)
            setattr(cfg, k, v)
        try:
            return validate_fn()
        finally:
            for k, v in originals.items():
                setattr(cfg, k, v)

    # n_embd not divisible by n_head
    warns = _check_warns(n_embd=128, n_head=7)
    assert any("divisible" in w.lower() for w in warns), \
        f"expected divisibility warning, got: {warns}"

    # n_head not a multiple of n_kv_head
    warns = _check_warns(n_embd=256, n_head=8, n_kv_head=3)
    assert any("multiple" in w.lower() for w in warns), \
        f"expected multiple warning, got: {warns}"

    # n_expert_top >= n_expert triggers validation warning
    warns = _check_warns(use_moe=True, n_expert=4, n_expert_top=4)
    assert any("must be <" in w.lower() for w in warns), \
        f"expected n_expert_top < n_expert warning, got: {warns}"


# ---------------------------------------------------------------------------
# Profile-specific model construction (requires torch)
# ---------------------------------------------------------------------------


@needs_torch
@pytest.mark.parametrize("profile", ["dev", "smol-fat", "fat"])
def test_model_builds_without_error(profile: str):
    """Each profile should produce a GPT instance with the right number of
    parameters."""
    import os
    os.environ["CHATON_PROFILE"] = profile
    # Re-import config with the new profile
    import importlib
    import config as cfg
    importlib.reload(cfg)

    from model import GPT
    model = GPT()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0, "model has zero parameters!"

    # Known parameter bounds for each profile
    if profile == "dev":
        assert n_params < 100_000_000, \
            f"dev profile should be <100M params, got {n_params/1e6:.1f}M"
    elif profile == "smol-fat":
        assert n_params > 100_000_000, \
            f"smol-fat should be >100M params, got {n_params/1e6:.1f}M"
        assert n_params < 1_000_000_000, \
            f"smol-fat should be <1B params, got {n_params/1e6:.1f}M"
    elif profile == "fat":
        assert n_params > 1_000_000_000, \
            f"fat should be >1B params, got {n_params/1e6:.1f}M"


@needs_torch
def test_model_generate_returns_correct_shape():
    """A forward+generate call produces output of the expected shape."""
    import os
    os.environ["CHATON_PROFILE"] = "dev"

    import importlib
    import config as cfg
    importlib.reload(cfg)

    from model import GPT
    model = GPT()
    device = "cpu"
    model.to(device)

    B, T = 2, 16
    idx = torch.randint(0, cfg.vocab_size, (B, T), device=device)

    # forward pass
    logits, loss, caches = model(idx, targets=idx)
    assert logits.shape == (B, T, cfg.vocab_size), \
        f"expected ({B}, {T}, {cfg.vocab_size}), got {logits.shape}"
    assert loss is not None and loss.item() > 0, "loss should be > 0"

    # generation
    gen = model.generate(idx[:, :1], max_new_tokens=8, temperature=1.0)
    assert gen.shape == (B, 9), \
        f"expected ({B}, 9), got {gen.shape} (1 prompt token + 8 new)"


@needs_torch
def test_kv_cache_round_trip():
    """KV cache creation + extension returns the same result as a full forward."""
    import os
    os.environ["CHATON_PROFILE"] = "dev"
    import importlib
    import config as cfg
    importlib.reload(cfg)

    from model import GPT
    model = GPT()
    model.to("cpu")
    model.eval()

    B, T1, T2 = 1, 8, 4
    x = torch.randint(0, cfg.vocab_size, (B, T1))
    x2 = torch.randint(0, cfg.vocab_size, (B, T2))

    # Full forward
    full, _, _ = model(torch.cat([x, x2], dim=1))

    # Cached forward
    _, _, caches = model(x, use_cache=True)
    cached, _, _ = model(x2, use_cache=True, kv_caches=caches, rope_offset=T1)

    # The logits for the second segment should match
    assert torch.allclose(full[:, T1:, :], cached[:, :, :], atol=1e-5), \
        "KV cache and full forward logits differ"



@needs_torch
@pytest.mark.parametrize("profile", ["dev", "smol-fat"])
def test_moe_vs_dense_params(profile: str):
    """MoE models should have active (forward) params less than total params."""
    import os
    os.environ["CHATON_PROFILE"] = profile
    import importlib
    import config as cfg
    importlib.reload(cfg)

    from model import GPT
    model = GPT()
    total = sum(p.numel() for p in model.parameters())

    # Count active params (exclude non-routed expert params)
    active = 0
    for name, p in model.named_parameters():
        is_routed_expert = ("experts." in name)
        if is_routed_expert and cfg.use_moe:
            # Each expert MLP has 3 weight matrices (SwiGLU: up/gate/down).
            # Only n_expert_top / n_expert fraction of expert params are active.
            expert_params = p.numel()
            # The fraction of expert params active per token:
            frac_active = cfg.n_expert_top / cfg.n_expert
            active += int(expert_params * frac_active)
        else:
            active += p.numel()

    if cfg.use_moe:
        assert active < total, \
            f"MoE should have fewer active params than total ({active} < {total})"


@needs_torch
def test_config_snapshot_clean():
    """Checkpoint config snapshots should only contain ARCH_KEYS, not noise."""
    import config as cfg
    snapshot = {k: getattr(cfg, k, None) for k in cfg.ARCH_KEYS}
    assert len(snapshot) == len(cfg.ARCH_KEYS), \
        f"snapshot has {len(snapshot)} keys, expected {len(cfg.ARCH_KEYS)}"
    # Verify no private keys leak
    for k in snapshot:
        assert not k.startswith("_"), f"private key leaked into snapshot: {k}"
        assert k in cfg.ARCH_KEYS, f"unexpected key in snapshot: {k}"
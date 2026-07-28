# Contributing to Le Gros Chaton

Thanks for your interest! This project is a from-scratch Mixture-of-Experts
coding language model. Contributions of all kinds are welcome — code, bug
reports, documentation, and research ideas.

## Quick Start

```bash
# Clone and install
git clone https://github.com/Mateooo93/le-gros-chaton.git
cd le-gros-chaton
pip install -e ".[dev]"
pre-commit install

# Verify everything works
make check       # environment readiness check
make demo        # smoke test (checks imports, model build, generation)
make test-quick  # run tests that don't need a GPU

# View the effective configuration
make info
```

## Development Workflow

1. **Pick an issue** or propose a change by opening an issue first.
2. **Create a branch**: `git checkout -b feat/my-change`.
3. **Make your change** — one logical change per commit.
4. **Run the tests**: `make test` (GPU required for some tests).
5. **Format and lint**: `make format && make lint` (auto-fixes most issues).
6. **Commit** with a descriptive message (see below).
7. **Push and open a PR**.

## Code Conventions

- **Flat imports at root** — core modules (`model.py`, `config.py`, `train.py`)
  live at the project root, NOT in `src/`. This is load-bearing (see README).
- **Type hints** — use Python 3.11+ style annotations for all public functions.
- **Docstrings** — Google/NumPy style for non-trivial functions.
- **Line length** — 99 characters (enforced by black + ruff).
- **No unused imports** — ruff will flag them in CI.
- **No bare `except:`** — always catch specific exceptions.

## Commit Message Style

```
<type>: <short description>

<optional body explaining what and why, not how>
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`, `perf`.

Examples:
```
feat: add MinHash near-deduplication for code corpus
fix: from_checkpoint handles vocab size mismatch
docs: update TECHNICAL.md with MoE routing section
```

## Testing

- Tests live in `tests/` and use pytest.
- Tests that need a GPU are marked with `@pytest.mark.needs_torch`
  (defined in `tests/conftest.py`).
- Run `make test-quick` for GPU-free tests, `make test` for the full suite.
- Add tests for new features, especially data pipeline and model changes.

## Project Structure

```
├── model.py            # MoE transformer
├── config.py           # Profiles + env-var overrides
├── train.py            # Training loop (resumable)
├── inference.py        # Unified inference engine
├── pipeline.py         # 6-stage RL pipeline orchestrator
├── rft.py / rlvr.py    # RL training stages
├── prm.py              # Process Reward Model
├── agent_rl.py         # Agent-loop-as-rollout RL
├── data_code.py        # Code data pipeline
├── agent/              # Terminal agent harness
├── verify/             # Solution verifier
├── eval/               # Evaluation harnesses
├── tests/              # pytest suite
└── context/            # Documentation
```

## Getting Help

Open an issue or ask in the project's discussion board.

## License

MIT — see [LICENSE](LICENSE).

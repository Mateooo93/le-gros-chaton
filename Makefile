# Le Gros Chaton — common development tasks
#
# Usage:
#   make install      # Install dependencies (dev + pre-commit)
#   make test         # Run the test suite
#   make test-quick   # Run tests that don't need torch (step extraction, etc.)
#   make format       # Format all Python files (black + isort)
#   make lint         # Lint all Python files (ruff)
#   make demo         # Run the smoke test (go.py)
#   make clean        # Remove temp files and caches
#   make serve        # Start the interactive chat

.PHONY: install test test-quick format lint demo clean serve

PYTHON := python3
FILES := *.py agent/*.py eval/*.py verify/*.py tests/*.py

install:
	$(PYTHON) -m pip install -e ".[dev]"
	pre-commit install

test:
	$(PYTHON) -m pytest tests/ -v

test-quick:
	$(PYTHON) -m pytest tests/ -k "not needs_torch" -v

format:
	$(PYTHON) -m black --line-length 99 $(FILES)
	$(PYTHON) -m isort --profile black --line-length 99 $(FILES)

lint:
	$(PYTHON) -m ruff check $(FILES)

demo:
	$(PYTHON) go.py

check:
	$(PYTHON) check_env.py

info:
	$(PYTHON) train.py --info

sanity:
	$(PYTHON) pipeline.py --sanity

serve:
	$(PYTHON) inference.py --ckpt model.pt

# Qwen fine-tuning targets
qwen-inspect:
	$(PYTHON) finetune_qwen.py --mode inspect --model Qwen/Qwen2.5-Coder-7B

qwen-selfplay:
	$(PYTHON) self_play_data.py --qwen --model Qwen/Qwen2.5-Coder-7B --problems humaneval --n 10

qwen-train:
	$(PYTHON) finetune_qwen.py --mode rlvr --model Qwen/Qwen2.5-Coder-7B --n-steps 50 --limit 5

qwen-eval:
	$(PYTHON) eval_qwen.py --model Qwen/Qwen2.5-Coder-7B --mode humaneval --limit 10

qwen-agent:
	$(PYTHON) agent_qwen.py --model Qwen/Qwen2.5-Coder-7B --max-steps 3 "list .py files"

# SWE-bench evaluation
swebench:
	$(PYTHON) eval_swebench.py --model Qwen/Qwen3.5-9B --ckpt qwen_coding_agent --limit 10

swebench-save:
	$(PYTHON) eval_swebench.py --results swebench_results.json

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '.pytest_cache' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
	find . -name '.ruff_cache' -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf *.egg-info dist build 2>/dev/null || true

research-check:
	rm -rf /tmp/research-check && mkdir -p /tmp/research-check
	$(PYTHON) self_play_data.py --dry-run
	@echo "Research scripts verified."

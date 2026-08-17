#!/usr/bin/env python3
"""Poll Kaggle traj-SFT run to completion; exit 0 on success, 1 on NaN/error.

Also sniffs the log for 'grad_norm=nan' / 'loss=nan' to kill a diverging run
early (v22 burned hours on a NaN fp16-overflow run).

Self-sufficient: reads Kaggle creds from gpus.md (gitignored) so it works
under any daemon env, and prints stderr on failure instead of empty lines.
"""
import os
import re
import subprocess
import sys
import time

PROJ_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KERNEL = "mateomanceron/le-gros-chaton-qwen-training"
INTERVAL = 180  # seconds
MAX_WAIT = 9 * 3600  # Kaggle 9h session cap


def load_creds():
    """Inject creds from gpus.md into env (kaggle CLI prefers env over json)."""
    gpus = os.path.join(PROJ_ROOT, "gpus.md")
    if not os.path.exists(gpus):
        return
    for line in open(gpus):
        m = re.match(r"(KAGGLE_USERNAME|KAGGLE_KEY|HF_TOKEN)\s*=\s*(\S+)", line)
        if m:
            os.environ.setdefault(m.group(1), m.group(2))


def status():
    r = subprocess.run(["kaggle", "kernels", "status", KERNEL],
                       capture_output=True, text=True, timeout=60)
    out = r.stdout.strip()
    if not out:
        out = f"(empty; rc={r.returncode} err={r.stderr.strip()[:120]})"
    return out


def log_tail():
    r = subprocess.run(["kaggle", "kernels", "logs", KERNEL],
                       capture_output=True, text=True, timeout=90)
    return r.stdout + r.stderr


def main():
    load_creds()
    t0 = time.time()
    seen_nan = False
    while time.time() - t0 < MAX_WAIT:
        st = status()
        elapsed = (time.time() - t0) / 3600
        print(f"[monitor] {elapsed:.2f}h | {st}", flush=True)
        if "complete" in st.lower():
            print("[monitor] DONE (complete)", flush=True)
            return 0
        if "error" in st.lower():
            print("[monitor] FAILED (error)", flush=True)
            return 1
        # Sniff for NaN divergence so we don't burn the whole 9h like v22.
        try:
            tail = log_tail()
            for pat in (r"grad_norm=nan", r"grad_norm=NaN", r"'loss': nan",
                        r"loss: nan", r"loss=nan", r"illegal memory access",
                        r"out of memory", r"CUDA error"):
                if re.search(pat, tail):
                    print(f"[monitor] CRASH SIGNATURE ({pat})", flush=True)
                    seen_nan = True
                    break
        except Exception as e:
            print(f"[monitor] log sniff skipped: {e}", flush=True)
        if seen_nan:
            print("[monitor] aborting early — crashed run", flush=True)
            return 1
        time.sleep(INTERVAL)
    print("[monitor] timed out", flush=True)
    return 2


if __name__ == "__main__":
    sys.exit(main())

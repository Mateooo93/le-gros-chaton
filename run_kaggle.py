"""Push + run + monitor Kaggle training via the Kaggle API.

Requires KAGGLE_USERNAME and KAGGLE_KEY env vars (or kaggle.json).
Saves them in gpus.md (gitignored) — never in source.

Usage:
    python run_kaggle.py                      # push + run + monitor
    python run_kaggle.py --limit 5000         # smaller run
    python run_kaggle.py --status             # check current run status
    python run_kaggle.py --output             # download model artifacts
"""
import argparse
import json
import os
import subprocess
import sys
import time

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
KAGGLE_DIR = os.path.join(PROJ_ROOT, "kaggle")
KERNEL_ID = "mateomanceron/le-gros-chaton-qwen-training"


def load_creds():
    """Load Kaggle credentials from gpus.md or env."""
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if not (username and key):
        # Try gpus.md
        gpus = os.path.join(PROJ_ROOT, "gpus.md")
        if os.path.exists(gpus):
            with open(gpus) as f:
                for line in f:
                    if "KAGGLE_USERNAME" in line and "=" in line:
                        username = line.split("=", 1)[1].strip()
                    elif "KAGGLE_KEY" in line and "=" in line:
                        key = line.split("=", 1)[1].strip()
    if not (username and key):
        print("[kaggle] ERROR: Need KAGGLE_USERNAME + KAGGLE_KEY")
        print("[kaggle] Add them to gpus.md (gitignored) or set as env vars")
        sys.exit(1)
    return username, key


def setup_env():
    """Set Kaggle env vars from creds."""
    u, k = load_creds()
    os.environ["KAGGLE_USERNAME"] = u
    os.environ["KAGGLE_KEY"] = k


def push_kernel(limit: int, phase: str = "sft", rlvr_steps: int = 100,
                rlvr_group: int = 4):
    """Write kernel-metadata.json with current limit, push the kernel."""
    setup_env()
    meta_path = os.path.join(KAGGLE_DIR, "kernel-metadata.json")
    with open(meta_path) as f:
        meta = json.load(f)
    # Inject HF token + limit + phase into a TEMP copy (never committed)
    with open(os.path.join(KAGGLE_DIR, "kaggle_train.py")) as f:
        script = f.read()
    script = script.replace('os.environ.get("TRAIN_LIMIT", "10000")',
                            f'os.environ.get("TRAIN_LIMIT", "{limit}")')
    script = script.replace('os.environ.get("TRAIN_PHASE", "sft")',
                            f'os.environ.get("TRAIN_PHASE", "{phase}")')
    script = script.replace('os.environ.get("RLVR_STEPS", "100")',
                            f'os.environ.get("RLVR_STEPS", "{rlvr_steps}")')
    script = script.replace('os.environ.get("RLVR_GROUP", "4")',
                            f'os.environ.get("RLVR_GROUP", "{rlvr_group}")')

    # Pull token from gpus.md at push time only
    import re
    hf_token = ""
    gpus = os.path.join(PROJ_ROOT, "gpus.md")
    if os.path.exists(gpus):
        with open(gpus) as f:
            for line in f:
                m = re.match(r"HF_TOKEN\s*=\s*(\S+)", line)
                if m:
                    hf_token = m.group(1)
    if hf_token:
        # Inject into the upload copy so the token is set unconditionally on
        # Kaggle (HF_TOKEN is not pre-set there). gitignored temp file only.
        script = script.replace(
            'hf_token = os.environ.get("HF_TOKEN", "")',
            f'os.environ["HF_TOKEN"] = "{hf_token}"\n    hf_token = os.environ.get("HF_TOKEN", "")')
        print("[kaggle] HF token injected into upload (temp only)")

    temp_script = os.path.join(KAGGLE_DIR, "kaggle_train_upload.py")
    with open(temp_script, "w") as f:
        f.write(script)

    # Point metadata at the temp file
    with open(os.path.join(KAGGLE_DIR, "kernel-metadata.json")) as f:
        meta = json.load(f)
    meta["code_file"] = "kaggle_train_upload.py"
    with open(os.path.join(KAGGLE_DIR, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[kaggle] Pushing kernel {KERNEL_ID} (limit={limit})...")
    r = subprocess.run(["kaggle", "kernels", "push", "-p", KAGGLE_DIR],
                       capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print("[kaggle] Push error:", r.stderr)
        sys.exit(1)

    # Restore metadata code_file to the committed source so the gitignored
    # upload file isn't referenced in the repo.
    meta["code_file"] = "kaggle_train.py"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def monitor(interval: int = 60, max_wait: int = 9 * 3600):
    """Poll kernel status until complete or timeout (9hr Kaggle limit)."""
    setup_env()
    t0 = time.time()
    while time.time() - t0 < max_wait:
        r = subprocess.run(["kaggle", "kernels", "status", KERNEL_ID],
                           capture_output=True, text=True)
        out = r.stdout.strip()
        elapsed = (time.time() - t0) / 3600
        print(f"[kaggle] {elapsed:.1f}h | {out}")
        if "complete" in out.lower():
            print("[kaggle] ✓ Training complete!")
            return True
        if "error" in out.lower():
            print("[kaggle] ✗ Training failed")
            return False
        time.sleep(interval)
    print("[kaggle] Timed out after 9h (Kaggle session limit)")
    return False


def download_output():
    """Download kernel output (model artifacts)."""
    setup_env()
    print("[kaggle] Downloading output...")
    r = subprocess.run(["kaggle", "kernels", "output", KERNEL_ID,
                        "-p", os.path.join(PROJ_ROOT, "kaggle_output")],
                       capture_output=True, text=True)
    print(r.stdout)
    if r.returncode != 0:
        print("[kaggle] Download error:", r.stderr)
    else:
        print(f"[kaggle] Output saved to {PROJ_ROOT}/kaggle_output/")


def main():
    parser = argparse.ArgumentParser(description="Run Kaggle training via API")
    parser.add_argument("--limit", type=int, default=10000,
                        help="Dataset rows for SFT")
    parser.add_argument("--phase", default="sft", choices=["sft", "rlvr"],
                        help="Which training phase to run")
    parser.add_argument("--rlvr-steps", type=int, default=100,
                        help="RLVR training steps (phase=rlvr)")
    parser.add_argument("--rlvr-group", type=int, default=4,
                        help="GRPO group size (phase=rlvr)")
    parser.add_argument("--status", action="store_true",
                        help="Check current kernel status")
    parser.add_argument("--output", action="store_true",
                        help="Download kernel output")
    parser.add_argument("--monitor-only", action="store_true",
                        help="Only monitor (assume already pushed)")
    args = parser.parse_args()

    if args.status:
        setup_env()
        r = subprocess.run(["kaggle", "kernels", "status", KERNEL_ID],
                           capture_output=True, text=True)
        print(r.stdout)
        return
    if args.output:
        download_output()
        return

    if not args.monitor_only:
        push_kernel(args.limit, phase=args.phase,
                    rlvr_steps=args.rlvr_steps, rlvr_group=args.rlvr_group)
    monitor()


if __name__ == "__main__":
    main()

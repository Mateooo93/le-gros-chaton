"""Lightweight experiment logger — structured JSONL with run metadata.

Records per-step training metrics to a newline-delimited JSON file:

    {"step": 0, "loss": 4.23, "lr": 3e-5, "grad_norm": 1.2, "elapsed_s": 12.3}
    {"step": 250, "loss": 3.12, "lr": 3e-4, "grad_norm": 0.8, "elapsed_s": 310.5}

Each run also writes a header record with config snapshot, git hash, and CLI args
for provenance.  The format is `jq`-parseable and pandas-readable.

USAGE
-----
    from log import ExperimentLog

    log = ExperimentLog("runs/my_run")
    log.write({"step": 0, "loss": 4.23, "lr": 3e-4})
    # ... later ...
    log.write({"step": 250, "loss": 2.89, "lr": 2e-5})
    log.close()  # also writes summary footer

VIEWING
-------
    # With jq:
    jq -s '.[]' runs/my_run/log.jsonl        # prettify all records
    jq -s 'map(select(.step)) | sort_by(.step) | .[] | "\\(.step) \\(.loss)"' \
        runs/my_run/log.jsonl -r              # TSV of step vs loss

    # With pandas:
    import pandas as pd
    df = pd.read_json("runs/my_run/log.jsonl", lines=True)
    df[df.step.notna()].plot(x="step", y="loss")

RESUMING
--------
Pass ``resume_from="runs/previous_run"`` to append to an existing log
(instead of starting fresh).  The header/footer are not duplicated.

INTEGRATION WITH train.py
--------------------------
At the top of the training loop:
    log = ExperimentLog("runs/train_20240727_120000")
Inside the loop:
    if step % cfg.eval_interval == 0:
        log.write({...})

The log is flushed after every write (crash-safe).
"""
import json
import os
import subprocess
import time
from typing import Any, TextIO


class ExperimentLog:
    """Append-only JSONL experiment logger.

    Writes to ``{run_dir}/log.jsonl``.  The first record is a header with
    run metadata (git hash, config snapshot, start time).  Each subsequent
    ``write()`` call appends one JSON line.  The file is flushed after every
    write so a crash loses at most one record.
    """

    def __init__(self, run_dir: str, resume_from: str | None = None):
        self.run_dir = run_dir
        self.start_time = time.time()
        self._n_writes = 0

        if resume_from:
            # Append to existing log
            if os.path.isdir(resume_from):
                self.log_path = os.path.join(resume_from, "log.jsonl")
            else:
                self.log_path = resume_from
            if not os.path.exists(self.log_path):
                print(f"[log] resume_from={resume_from} but no log.jsonl found; "
                      f"starting fresh")
                resume_from = None

        if not resume_from:
            os.makedirs(run_dir, exist_ok=True)
            self.log_path = os.path.join(run_dir, "log.jsonl")

        self._file: TextIO | None = open(self.log_path, "a")  # noqa: SIM115

        if not resume_from:
            self._write_header()

    def _write_header(self):
        """Write the run header record (metadata, not a metric step)."""
        # Capture git hash (best-effort)
        git_hash = ""
        try:
            r = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, timeout=5,
                cwd=os.path.dirname(os.path.abspath(__file__)),
            )
            if r.returncode == 0:
                git_hash = r.stdout.strip()
        except Exception:
            pass

        # Capture environment (only CHATON_* vars)
        env_vars = {
            k: v for k, v in os.environ.items()
            if k.startswith("CHATON_")
        }

        # Config snapshot
        config_snapshot = {}
        try:
            import config as cfg
            for k in cfg.ARCH_KEYS:
                config_snapshot[k] = getattr(cfg, k, None)
        except Exception:
            pass

        header = {
            "_type": "header",
            "git_hash": git_hash,
            "start_time_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "start_time_unix": self.start_time,
            "config": config_snapshot,
            "env": env_vars,
        }
        self._write_record(header)

    def write(self, record: dict[str, Any]):
        """Append one metric record.  The dict should include at least ``step``
        plus any scalars you want tracked (loss, lr, grad_norm, val_loss, ...).
        """
        record["elapsed_s"] = round(time.time() - self.start_time, 1)
        self._write_record(record)
        self._n_writes += 1

    def _write_record(self, record: dict):
        if self._file is None:
            return
        self._file.write(json.dumps(record, default=str) + "\n")
        self._file.flush()
        os.fsync(self._file.fileno())

    @property
    def path(self) -> str:
        return self.log_path

    def get_last_step(self) -> int:
        """Read the most recent metric step from the log (for resumption).

        Returns 0 if the log is empty or has no metric records.
        """
        if not os.path.exists(self.log_path):
            return 0
        last = 0
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    step = r.get("step")
                    if step is not None and isinstance(step, (int, float)):
                        last = int(step)
                except json.JSONDecodeError:
                    continue
        return last

    def close(self, summary: dict | None = None):
        """Close the log file.  Optionally write a summary footer."""
        if self._file is None:
            return
        if summary:
            summary["_type"] = "footer"
            summary["elapsed_s"] = round(time.time() - self.start_time, 1)
            self._write_record(summary)
        self._file.close()
        self._file = None
        print(f"[log] closed {self.log_path} ({self._n_writes} metric records)")

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


# ---------------------------------------------------------------------------
# CLI viewer: ``python log.py runs/my_run/log.jsonl``
# ---------------------------------------------------------------------------

def _view(path: str, keys: list[str] | None = None):
    """Pretty-print an experiment log.

    If *keys* is provided, only those fields are shown (plus step and elapsed).
    """
    import json
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    if not records:
        print(f"[log] no records in {path}")
        return

    # Print header if present
    headers = [r for r in records if r.get("_type") == "header"]
    if headers:
        h = headers[0]
        print(f"Run:   {path}")
        print(f"Git:   {h.get('git_hash', '?')}")
        print(f"Start: {h.get('start_time_iso', '?')}")
        if h.get("config"):
            print(f"Config: {json.dumps(h['config'])}")
        print()

    # Print metric records
    metrics = [r for r in records if r.get("_type") != "header"]
    if not metrics:
        return

    # Show the last 5 metric records, or custom keys if provided
    if keys:
        header = ["step", "elapsed_s"] + keys
        print("  ".join(f"{k:>12}" for k in header))
        print("  " + "-" * (12 * len(header)))
        for r in metrics[-5:]:
            vals = [f"{r.get(k, '?'):>12}" for k in header]
            print("  ".join(vals))
    else:
        # Show all keys present across all metric records
        all_keys = set()
        for r in metrics:
            all_keys.update(r.keys())
        all_keys.discard("_type")
        print("Fields: " + ", ".join(sorted(all_keys)))
        print()
        print(f"Total metric records: {len(metrics)}")
        print(f"Last step: {metrics[-1].get('step', '?')}")
        print(f"Elapsed:  {metrics[-1].get('elapsed_s', '?'):.0f}s")

    # Compute aggregates
    losses = [r.get("loss") for r in metrics if r.get("loss") is not None]
    if losses:
        print(f"Loss range: {min(losses):.4f} – {max(losses):.4f}")


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "runs/train/log.jsonl"
    keys = sys.argv[2:] if len(sys.argv) > 2 else None
    _view(path, keys)
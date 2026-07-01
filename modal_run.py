"""Modal launcher for le fat chaton pretraining.

Ships your local project into a Modal container, installs deps, runs train.py
with the chosen profile, and auto-pushes checkpoints to HuggingFace Hub every
CHATON_CKPT_INTERVAL steps (VM-hopping: Kaggle/Colab/another Modal run can pull
and resume via CHATON_RESUME=1).

USAGE:
  python modal_run.py                         # smol-fat proof run on A100-40GB
  CHATON_PROFILE=fat CHATON_MAX_ITERS=5000 python modal_run.py   # full 9B (costly)

Profiles (set CHATON_PROFILE):
  smol-fat (default) ~240M/63M-active MoE — cheap proof (~$1)
  fat               9.23B/2.69B-active MoE — the real thing ($$$)
  dev               tiny — only for your 2070, don't run here
"""
import os
import modal

LOCAL_DIR = os.path.dirname(os.path.abspath(__file__))

# secrets: read from gpus.md so nothing is hardcoded in source/git
def _secret(key):
    p = os.path.join(LOCAL_DIR, "gpus.md")
    if not os.path.exists(p):
        return ""
    for line in open(p):
        if line.startswith(key):
            return line.split("=", 1)[1].strip()
    return ""

HF_TOKEN = _secret("HF_token")
HF_REPO = "mateo0093/le-fat-chaton-ckpt"
REPO_URL = _secret("REPO_url")

# The HF token can come from EITHER:
#   1. a Modal named secret (set CHATON_HF_SECRET="my-secret-name") -> injected
#      by Modal into the container env as HF_TOKEN. PREFERRED when running from
#      the pushed GitHub repo (gpus.md won't exist there).
#   2. gpus.md locally (the _secret() fallback above) -> baked into a from_dict
#      secret. Fine for local dev where gpus.md is present.
# In the container, the function always reads os.environ["HF_TOKEN"], so both
# 1 and 2 land in the same place.
HF_SECRET_NAME = os.environ.get("CHATON_HF_SECRET", "")   # empty = use from_dict

# Build the secrets list once. If a named Modal secret is configured
# (CHATON_HF_SECRET), use it (the container gets HF_TOKEN from Modal). Otherwise
# bake the token read from gpus.md into an inline from_dict secret. Either way
# the function reads os.environ["HF_TOKEN"] at runtime.
if HF_SECRET_NAME:
    _SECRETS = [modal.Secret.from_name(HF_SECRET_NAME)]
else:
    if not HF_TOKEN:
        print("[modal] WARNING: no HF token found (gpus.md missing and no "
              "CHATON_HF_SECRET set) — checkpoint push/pull will fail. "
              "Create a Modal secret named e.g. 'chaton-hf' with key HF_TOKEN "
              "and run with CHATON_HF_SECRET=chaton-hf.")
    _SECRETS = [modal.Secret.from_dict({"HF_TOKEN": HF_TOKEN,
                                        "CHATON_HF_REPO": HF_REPO})]

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch", "tiktoken", "datasets==2.21.0", "huggingface_hub<0.30",
                 "tokenizers", "numpy")
    # ship the local project into /root/proj (gpus.md is in .gitignore so it
    # won't ship — secrets stay local). We add it to the image explicitly.
    .add_local_dir(LOCAL_DIR, "/root/proj",
                   ignore=["gpus.md", "*.pt", "*.bin", "__pycache__", ".git",
                           "models", "/tmp"])
)

app = modal.App("le-fat-chaton-train")
ckpt_vol = modal.Volume.from_name("le-fat-chaton-ckpt", create_if_missing=True)

# Default GPU = L4 (24GB, ~$0.80/hr). The cost-efficiency sweet spot for the
# smol-fat 240M MoE: 8GB more headroom than a T4 (fits block 1024/micro 8 with
# compile ON), ~2.6x cheaper than an A100-40GB which a 240M model can't use.
# Override per run: CHATON_GPU=L4 | A10G | T4 | A100-40GB | A100-80GB.
#  -> smol-fat proof: L4 (default).  -> fat 10.25B pretrain: A100-40GB or 80GB.
GPU = os.environ.get("CHATON_GPU", "L4")
PROFILE = os.environ.get("CHATON_PROFILE", "smol-fat")
MAX_ITERS = os.environ.get("CHATON_MAX_ITERS", "1000")

# data source + code-corpus config (only used when CHATON_DATA=code). These
# pass through to the container so the fat coding pretrain can switch to the
# code corpus without editing modal_run.py.
DATA_CHOICE = os.environ.get("CHATON_DATA", "wikitext")        # wikitext | code
CODE_CORPUS = os.environ.get("CHATON_CODE_CORPUS", "smollm-corpus")
CODE_MAX_TOKENS = os.environ.get("CHATON_CODE_MAX_TOKENS", "50000000")
CODE_MAX_DOCS = os.environ.get("CHATON_CODE_MAX_DOCS", "200000")


@app.function(
    image=image,
    gpu=GPU,
    timeout=60 * 60,                 # 1h cap per run; resume via Hub for more
    volumes={"/ckpt": ckpt_vol},
    secrets=_SECRETS,
)
def train():
    import os, subprocess, sys
    os.chdir("/root/proj")
    env = dict(os.environ)
    # HF_TOKEN comes from the Modal secret (named or from_dict) -> already in
    # os.environ here. Fall back to the launch-time value if absent.
    env["HF_TOKEN"] = env.get("HF_TOKEN") or HF_TOKEN
    env["CHATON_HF_REPO"] = HF_REPO
    env.update({
        "CHATON_PROFILE": PROFILE,
        "CHATON_MAX_ITERS": MAX_ITERS,
        # Memory settings scale with the GPU. Defaults below suit the L4 24GB
        # default (block 1024 / micro 8 / accum 8 = eff batch 64, compile ON —
        # the healthy config, NOT the squeezed T4 one). For a T4, override at
        # launch with CHATON_BLOCK_SIZE=512 CHATON_MICRO_BATCH=4 CHATON_GRAD_ACCUM=16
        # CHATON_COMPILE=0. For A100-40GB do block 2048 / micro 16 / accum 4.
        "CHATON_BLOCK_SIZE": os.environ.get("CHATON_BLOCK_SIZE", "1024"),
        "CHATON_MICRO_BATCH": os.environ.get("CHATON_MICRO_BATCH", "8"),
        "CHATON_GRAD_ACCUM": os.environ.get("CHATON_GRAD_ACCUM", "8"),  # eff batch 64
        "CHATON_COMPILE": os.environ.get("CHATON_COMPILE", "1"),        # ON for L4+
        "CHATON_CKPT_INTERVAL": "250",
        "CHATON_CKPT_PATH": "/ckpt/checkpoint.pt",  # persistent Modal volume (survives reaper)
        "CHATON_RESUME": "1",            # pull latest ckpt from Hub if present
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # data source: wikitext for the durability proof run, code for the fat
        # coding pretrain (CHATON_DATA=code at launch -> stream smollm-corpus).
        "CHATON_DATA": DATA_CHOICE,
        "CHATON_CODE_CORPUS": CODE_CORPUS,
        "CHATON_CODE_MAX_TOKENS": CODE_MAX_TOKENS,
        "CHATON_CODE_MAX_DOCS": CODE_MAX_DOCS,
    })
    # arch upgrades (only the fat profile turns these on via config.py; for
    # smol-fat they're no-ops since config.py sets gelu/MHA/0-shared there).
    # Only inject if non-empty so empty strings can't override config.py defaults.
    for _k in ("CHATON_MLP_TYPE", "CHATON_N_KV_HEAD", "CHATON_N_SHARED_EXPERT"):
        _v = os.environ.get(_k, "")
        if _v:
            env[_k] = _v
    print(f"[modal] profile={PROFILE} gpu={GPU} iters={MAX_ITERS} data={DATA_CHOICE}")
    p = subprocess.Popen([sys.executable, "-u", "train.py"], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="")
    p.wait()
    print("[modal] train.py exit:", p.returncode)
    ckpt_vol.commit()


if __name__ == "__main__":
    with app.run():
        train.remote()
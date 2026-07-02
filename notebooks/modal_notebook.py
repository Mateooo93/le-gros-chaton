"""Modal web-notebook cell: trains smol-fat on an L4, auto-saves, auto-releases
the GPU when done.

PASTE THIS WHOLE FILE AS ONE CELL in the Modal web notebook (modal.com notebook).

Before running:
  1. Attach the secret named `chaton-hf` (with key HF_TOKEN) to the notebook
     (top toolbar -> Secrets -> add chaton-hf).
  2. The notebook's CPU session is fine — the GPU is spun up separately by the
     .remote() call below and AUTO-RELEASED when it returns = you stop paying
     for the L4 the instant training finishes ("delete the kernel when done").

What it does:
  - builds a container image (torch + pinned datasets)
  - the @app.function runs ON an L4 GPU container:
      * git clones Mateooo93/le-gros-chaton
      * trains smol-fat (block 1024 / micro 8 / accum 8 = eff 64, compile ON)
      * pushes resumable checkpoints to HF Hub mateo0093/le-fat-chaton-ckpt
        every 250 steps  +  saves final model.pt to the persistent Modal volume
  - train.remote() streams the live training logs back to this cell
  - when it returns, the L4 container exits -> billing stops automatically
"""
import os, modal, subprocess, sys

app = modal.App("le-fat-chaton-train")
ckpt_vol = modal.Volume.from_name("le-fat-chaton-ckpt", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "build-essential")
    .pip_install("torch", "tiktoken", "datasets==2.21.0", "huggingface_hub<0.30",
                 "tokenizers", "numpy")
)


@app.function(
    image=image,
    gpu="L4",                        # 24GB, ~$0.80/hr. The cost-efficient pick.
    timeout=60 * 60,                 # 1h safety cap (L4 1000 steps ~15-25 min)
    volumes={"/ckpt": ckpt_vol},     # persistent: survives the reaper
    secrets=[modal.Secret.from_name("chaton-hf")],   # injects HF_TOKEN into env
)
def train():
    import os, subprocess, sys
    # clone the code into the GPU container (no local dir to ship in a web notebook)
    os.chdir("/root")
    subprocess.run(["git", "clone", "https://github.com/Mateooo93/le-gros-chaton.git"],
                   check=True)
    os.chdir("/root/le-gros-chaton")

    env = dict(os.environ)
    env.update({
        "CHATON_PROFILE":       "smol-fat",     # 240M / 63M-active MoE
        "CHATON_BLOCK_SIZE":    "1024",         # L4 24GB fits this comfortably
        "CHATON_MICRO_BATCH":   "8",
        "CHATON_GRAD_ACCUM":    "8",            # eff batch = 64
        "CHATON_COMPILE":       "1",            # ON (L4 has enough SMs)
        "CHATON_MAX_ITERS":     "1000",
        "CHATON_CKPT_INTERVAL": "250",          # push to Hub every 250 steps
        "CHATON_RESUME":        "1",            # pull latest from Hub, resume
        "CHATON_DATA":          "wikitext",     # durability proof on wikitext
        "CHATON_CKPT_PATH":     "/ckpt/checkpoint.pt",   # persistent Modal volume
        "CHATON_HF_REPO":       "mateo0093/le-fat-chaton-ckpt",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        # HF_TOKEN is already in env from the chaton-hf secret
    })
    print("[notebook] HF_TOKEN present:", "HF_TOKEN" in env)

    # stream train.py live
    p = subprocess.Popen([sys.executable, "-u", "train.py"], env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in p.stdout:
        print(line, end="")
    rc = p.wait()
    print("[notebook] train.py exit:", rc)

    # persist the Modal volume (final model.pt + any local checkpoint)
    ckpt_vol.commit()
    print("[notebook] volume committed -> /ckpt survives container exit")
    return rc


# --- the actual launch: this runs on the NOTEBOOK (cheap CPU) session.
#     .remote() spins up the L4 container, runs train(), and AUTO-RELEASES the
#     L4 the moment it returns => "delete the kernel when done", billing stops. ---
with app.run():
    rc = train.remote()
    print(f"[notebook] DONE, exit={rc}. L4 container released -> GPU billing stopped.")
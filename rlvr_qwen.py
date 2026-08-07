#!/usr/bin/env python3
"""RLVR for the Qwen3.5-9B agent — GRPO with creativity rewards.

Port of rlvr.py's GRPO loop + diversity/novelty/strategy-switch rewards from
the nanoGPT-style toy (model.py) to the real deployment stack:

    base = Qwen3.5-9B in 4-bit (nf4)  --or-- any HF causal LM
    + trajectory-SFT LoRA adapter (optional, frozen)
    + fresh trainable LoRA for the RL pass (the SFT adapter is frozen)

Each rollout is a FULL agent episode: SWEAgent (the production harness,
which the trajectory SFT taught the model to use) runs against a bug-repo
template; the reward is the hidden-test verifier pass/fail, shaped with a
novelty bonus for correct-but-fresh solutions (creative objective) plus a
strategy-switch bonus for rollouts that try alternate approaches after a
failure. Advantages are group-normalised; the objective is the clipped
surrogate from GRPO (PPO-style clip).

Usage:
    MODEL_NAME=qwen_merged ADAPTER=none DEVICE_MAP=cuda:0 \\
    python rlvr_qwen.py --n 8 --n-steps 120 --limit 19 --out qwen_rlvr
    (run_sft_pipeline.sh merges base+Fable5+traj into qwen_merged/, then
     runs this with the merged model as the RL base and a fresh RL LoRA)

(nanoGPT rlvr.py remains for the toy MoE — this file replaces it for Qwen.)
"""
import argparse
import os
import shutil
import sys
import tempfile

PROJ_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ_ROOT)

import torch
import torch.nn.functional as F

from rlvr import diversity_reward, strategy_switch_reward  # reuse creativity logic


def log(msg: str) -> None:
    print(f"[rlvr-qwen] {msg}", flush=True)


# ---------------------------------------------------------------------------
# Problems: gen_trajectories' bug templates (verifiable via hidden tests)
# ---------------------------------------------------------------------------

def load_problems(limit: int | None = None) -> list[dict]:
    from gen_trajectories import _buggy_versions
    templates = _buggy_versions()
    if limit:
        templates = templates[:limit]
    return templates


# ---------------------------------------------------------------------------
# Rollout: one full agent episode against a fresh bug repo
# ---------------------------------------------------------------------------

def run_episode(model, tokenizer, tpl: dict, device: str,
                temperature: float = 0.8) -> tuple[str, list[str], bool]:
    """Run the production agent loop on one template; return (finish_msg,
    action list, verifier_passed).  The repo is created in a temp dir and
    removed after verification."""
    from agent_swe import SWEAgent
    from gen_trajectories import make_repo, verify_repo

    work = tempfile.mkdtemp(prefix="rlvr_")
    try:
        repo_dir = os.path.join(work, "task")
        make_repo(tpl, repo_dir)
        real_files = ", ".join(sorted(tpl["files"].keys()))
        issue = (f"{tpl['issue']}\n\nHint: the bug is in {tpl['bug']}. "
                 f"Repo files: {real_files}.")

        agent = SWEAgent(model, tokenizer, repo_dir, device=device,
                         temperature=temperature)
        result = agent.run(issue)
        ok = bool(result.get("success"))
        finish_msg = (result.get("patch") if ok else
                      str(result.get("patch") or result.get("turn") or ""))
        actions = _extract_actions(finish_msg) or _extract_actions(
            "\n".join(m.get("content", "") for m in agent.trace if m))
        return finish_msg or "", actions, ok
    finally:
        shutil.rmtree(work, ignore_errors=True)


def _extract_actions(text: str) -> list[str]:
    """Rough action list for the strategy-switch reward (0..1 bonus only)."""
    lines = [l.strip() for l in (text or "").splitlines() if l.strip()]
    actions = [l for l in lines if any(k in l for k in
                ("```tool", "Tool [", "list_dir", "read_file", "write_file",
                 "run_test", "search_code", "finish"))]
    return actions or lines


# ---------------------------------------------------------------------------
# GRPO core (port of rlvr.grpo_step — policy-gradient on the fresh LoRA)
# ---------------------------------------------------------------------------

def _mean_tok_lp(model, tokenizer, text: str, keep_grad: bool = False) -> torch.Tensor:
    """Mean log-prob of *text* under the policy (single sequence)."""
    ids = tokenizer(text, return_tensors="pt").to(model.device)["input_ids"]
    if ids.size(1) < 2:
        return torch.tensor(0.0, device=model.device)
    if not keep_grad:
        with torch.no_grad():
            out = model(input_ids=ids[:, :-1])
            logits = out.logits
    else:
        out = model(input_ids=ids[:, :-1])
        logits = out.logits
    lp = -F.nll_loss(logits.reshape(-1, logits.size(-1)),
                     ids[:, 1:].reshape(-1), reduction="mean")
    return lp


def grpo_step(model, optimizer, tokenizer, problem: dict, n_completions: int,
              temperature: float, epsilon: float, device: str,
              diversity: bool, novelty_bonus: float) -> dict | None:
    """One GRPO step on *problem*: sample G agent rollouts, reward via the
    hidden-test verifier (+ creativity shaping), group-normalise advantages,
    clipped surrogate update. Returns step stats, or None if nothing passed."""
    completions: list[str] = []
    rollouts: list[list[str]] = []
    rewards: list[float] = []

    for b in range(n_completions):
        finish_msg, acts, ok = run_episode(model, tokenizer, problem, device,
                                            temperature=temperature)
        completions.append(finish_msg)
        rollouts.append(acts)
        rewards.append(1.0 if ok else 0.0)

    if max(rewards) == 0.0:
        return None  # no positive signal — skip (mirrors rlvr.py)

    # --- Creativity shaping (novelty bonus + strategy switch) ---
    if diversity:
        shaped = diversity_reward(completions, rewards, novelty_bonus=novelty_bonus)
        for i in range(n_completions):
            shaped[i] += 0.1 * strategy_switch_reward(rollouts[i] or [finish_msg])
        rewards = shaped

    # --- Old log-probs (text proxies) + group-normalised advantages ---
    with torch.no_grad():
        old_lps = [_mean_tok_lp(model, tokenizer, c).item() for c in completions]
    rewards_t = torch.tensor(rewards, device=device, dtype=torch.float32)
    mean_r, std_r = rewards_t.mean(), rewards_t.std() + 1e-8
    advantages = ((rewards_t - mean_r) / std_r).tolist()

    # --- Clipped surrogate objective ---
    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.tensor(0.0, device=device)
    for b in range(n_completions):
        if not completions[b] or rewards[b] <= 0 and abs(advantages[b]) < 1e-9:
            continue
        new_lp = _mean_tok_lp(model, tokenizer, completions[b], keep_grad=True)
        old_lp = torch.tensor(old_lps[b], device=device)
        ratio = torch.exp(new_lp - old_lp)
        clipped = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon)
        total_loss = total_loss + (-advantages[b] * torch.min(ratio, clipped))
    total_loss = total_loss / max(1, sum(1 for r in rewards if r > 0))
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], 1.0)
    optimizer.step()

    return {"rewards": rewards, "loss": float(total_loss.item()),
            "passed": sum(1 for r in rewards if r > 0)}


# ---------------------------------------------------------------------------
# Loader (mirrors colab/trajectory_sft.py; env-driven for tight GPUs)
# ---------------------------------------------------------------------------

def load_model(model_name: str, adapter: str, lora_r: int):
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import LoraConfig, PeftModel, get_peft_model

    device_map = os.environ.get("DEVICE_MAP", "auto").strip()
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    log(f"Loading base '{model_name}' (4-bit, device_map={device_map}) ...")
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=quant, device_map=device_map,
        trust_remote_code=True, torch_dtype=torch.float16)

    if adapter and adapter.strip().lower() not in ("none", "null", ""):
        log(f"Attaching trajectory-SFT adapter '{adapter}' (frozen) ...")
        model = PeftModel.from_pretrained(model, adapter)
        for name, p in model.named_parameters():
            if name.split(".")[-1].startswith("lora_"):
                p.requires_grad = False

    log(f"Attaching fresh LoRA (r={lora_r}, alpha={2 * lora_r}) ...")
    model = get_peft_model(model, LoraConfig(
        r=lora_r, lora_alpha=2 * lora_r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()
    return model, tok


def main():
    ap = argparse.ArgumentParser(description="RLVR (GRPO) for Qwen3.5-9B agent")
    ap.add_argument("--n", type=int, default=8, help="rollouts per step (G)")
    ap.add_argument("--n-steps", type=int, default=120)
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--limit", type=int, default=None, help="problems cap")
    ap.add_argument("--epsilon", type=float, default=0.2)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--novelty-bonus", type=float, default=0.2)
    ap.add_argument("--no-diversity", dest="diversity", action="store_false",
                    default=True)
    ap.add_argument("--out", default="qwen_rlvr")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        log("no CUDA — RLVR needs a GPU (12GB+ for 9B 4-bit); "
            "set DEVICE_MAP/MAX_MEMORY for tight cards")
        sys.exit(2)

    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3.5-9B")
    adapter = os.environ.get("ADAPTER", "none")
    lora_r = int(os.environ.get("LORA_R", "16"))
    model, tok = load_model(model_name, adapter, lora_r)

    problems = load_problems(args.limit)
    log(f"problems: {len(problems)}")

    from transformers import get_linear_schedule_with_warmup
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, betas=(0.9, 0.95))
    sched = get_linear_schedule_with_warmup(optimizer, 10, args.n_steps)

    os.makedirs(args.out, exist_ok=True)
    episode_scores = []
    for step in range(args.n_steps):
        problem = problems[step % len(problems)]
        try:
            res = grpo_step(model, optimizer, tok, problem, args.n,
                            args.temperature, args.epsilon, args.device,
                            args.diversity, args.novelty_bonus)
        except Exception as e:
            log(f"step {step+1}/{args.n_steps} FAILED: "
                f"{type(e).__name__}: {e}")
            continue
        if res is None:
            log(f"step {step+1}/{args.n_steps} | no positive reward — skip")
            continue
        sched.step()
        episode_scores.extend(res["rewards"])
        log(f"step {step+1}/{args.n_steps} | loss={res['loss']:.4f} "
            f"| rewards={[f'{r:.2f}' for r in res['rewards']]}")

        if (step + 1) % 10 == 0:
            d = os.path.join(args.out, f"step-{step+1}")
            model.save_pretrained(d)
            tok.save_pretrained(d)
            log(f"saved {d}")

    final = os.path.join(args.out, "final")
    model.save_pretrained(final)
    tok.save_pretrained(final)
    log(f"done. final adapter: {final}")
    log(f"reward summary: mean={sum(episode_scores)/len(episode_scores):.3f} "
        f"(n={len(episode_scores)})")


if __name__ == "__main__":
    main()
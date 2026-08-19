#!/usr/bin/env python3
"""Build a patched vLLM ROCm docker image that can serve text-only Qwen3.5
hybrid models (model_type=qwen3_5_text, 8 Gated-Attention + 24 Gated-DeltaNet).

The upstream image (rocm/vllm-dev:nightly_main_20260211) ships vllm 0.16rc2
which only knows the multimodal variants of Qwen3.5. To serve our text-only
merged model we apply three patches inside the image and `docker commit`:

1. registry.py:
   Add a `"Qwen3_5ForCausalLM"` -> `"Qwen3_5ForCausalLM"` entry so vllm's
   architecture resolution routes to the text-only handler (not the
   multimodal one which demands vision_config + vision weights).

2. qwen3_5.py:
   a. Import `ClassVar, Literal` from typing.
   b. Make `Qwen3_5ForCausalLMBase` inherit `IsHybrid` and add the
      `is_hybrid: ClassVar[Literal[True]] = True` attribute. Without this,
      vllm fails with "page size of the layer is not divisible" because it
      doesn't know about the hybrid mamba state shape.
   c. Add `get_mamba_state_dtype_from_config`,
      `get_mamba_state_shape_from_config`, and `get_mamba_state_copy_func`
      classmethods on the base (copied from the multimodal handler).
   d. Patch `Qwen3_5ProcessingInfo.get_hf_config` to fall back to
      `Qwen3_5TextConfig` (no vision_config) when the composite config
      isn't available.
   e. Patch `Qwen3_5ForCausalLMBase.load_weights` to use a `WeightsMapper`
      that strips `model.language_model.` and `language_model.` prefixes
      from checkpoint keys. Our merged safetensors (saved by merge_sft.py)
      was written by `AutoModelForCausalLM.from_pretrained(...)` which
      loaded the multimodal class and saved nested keys.

After commit, the image can serve:
  docker run --rm -d --name vllm --device /dev/kfd --device /dev/dri \\
    --network host --group-add video --ipc=host \\
    -v /root/cache/huggingface:/root/.cache/huggingface \\
    vllm-rocm-patched:latest \\
    vllm serve <model_path> --port 8000 --dtype bfloat16 \\
    --max-model-len 32768

Usage:
  python patch_vllm_docker.py            # patch + commit, image = vllm-rocm-patched:latest
  python patch_vllm_docker.py --no-build # only stage patches to /tmp (for inspection)

Idempotent: re-running applies each patch only if its target string is found.
"""
from __future__ import annotations
import argparse
import re
import subprocess
import sys
from pathlib import Path


BASE_IMAGE = "rocm/vllm-dev:nightly_main_20260211"
TARGET_IMAGE = "vllm-rocm-patched:latest"
VLLM_DIR = "/usr/local/lib/python3.12/dist-packages/vllm/model_executor/models"


# --- registry.py -----------------------------------------------------------

REGISTRY_OLD = """    "Qwen3_5ForConditionalGeneration": (
        "qwen3_5",
        "Qwen3_5ForConditionalGeneration",
    ),"""
REGISTRY_NEW = """    "Qwen3_5ForCausalLM": (
        "qwen3_5",
        "Qwen3_5ForCausalLM",
    ),
    "Qwen3_5ForConditionalGeneration": (
        "qwen3_5",
        "Qwen3_5ForConditionalGeneration",
    ),"""


# --- qwen3_5.py (multiple edits) ------------------------------------------

# Edit 1: imports — add ClassVar/Literal.
QWEN35_IMPORTS_OLD = "import typing\nfrom collections.abc import Callable, Iterable"
QWEN35_IMPORTS_NEW = "import typing\nfrom collections.abc import Callable, Iterable\nfrom typing import ClassVar, Literal"

# Edit 2: make Qwen3_5ProcessingInfo.get_hf_config accept the text-only config.
QWEN35_PROCINFO_OLD = """class Qwen3_5ProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        return self.ctx.get_hf_config(Qwen3_5Config)"""
QWEN35_PROCINFO_NEW = """class Qwen3_5ProcessingInfo(Qwen3VLProcessingInfo):
    def get_hf_config(self):
        # Accept text-only Qwen3.5 (model_type=qwen3_5_text -> Qwen3_5TextConfig,
        # no vision_config). The composite Qwen3_5Config is multimodal.
        try:
            return self.ctx.get_hf_config(Qwen3_5Config)
        except TypeError:
            return self.ctx.get_hf_config(Qwen3_5TextConfig)"""

# Edit 3: make Qwen3_5ForCausalLMBase inherit IsHybrid + is_hybrid attr.
QWEN35_BASE_CLASS_OLD = """class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
):
    packed_modules_mapping = {"""
QWEN35_BASE_CLASS_NEW = """class Qwen3_5ForCausalLMBase(
    nn.Module,
    HasInnerState,
    SupportsLoRA,
    SupportsPP,
    IsHybrid,
):
    # Required for vllm's KV-cache layout to compute hybrid (mamba+attention)
    # block sizes correctly. Without this, vllm 0.16rc2 raises
    # `NotImplementedError: The page size of the layer is not divisible`.
    is_hybrid: ClassVar[Literal[True]] = True
    packed_modules_mapping = {"""

# Edit 4: load_weights with a key mapper (strip model.language_model. / language_model.).
QWEN35_LOAD_OLD = """    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights)


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass"""
QWEN35_LOAD_NEW = """    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        from vllm.model_executor.models.utils import WeightsMapper
        # Our merged safetensors is written by `AutoModelForCausalLM.from_pretrained`
        # which loads the multimodal class and produces nested keys like
        # `model.language_model.layers.X` and `language_model.X`. Strip both.
        mapper = WeightsMapper(orig_to_new_prefix={
            "model.language_model.": "model.",
            "language_model.": "model.",
        })
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=["mtp."],
        )
        return loader.load_weights(weights, mapper=mapper)


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass"""

# Edit 5: add the mamba state classmethods to the base, before `class Qwen3_5ForCausalLM`.
QWEN35_MAMBA_METHODS_OLD = "class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):\n    pass"
QWEN35_MAMBA_METHODS_NEW = """    @classmethod
    def get_mamba_state_dtype_from_config(
        cls, vllm_config: "VllmConfig",
    ) -> tuple[torch.dtype, torch.dtype]:
        mamba_ssm_dtype = vllm_config.model_config.hf_text_config.mamba_ssm_dtype
        return MambaStateDtypeCalculator.gated_delta_net_state_dtype(
            vllm_config.model_config.dtype, mamba_ssm_dtype,
        )

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig",
    ) -> tuple[tuple[int, int], tuple[int, int]]:
        parallel_config = vllm_config.parallel_config
        hf_config = vllm_config.model_config.hf_text_config
        tp_size = parallel_config.tensor_parallel_size
        num_spec = (
            vllm_config.speculative_config.num_speculative_tokens
            if vllm_config.speculative_config else 0
        )
        return MambaStateShapeCalculator.gated_delta_net_state_shape(
            tp_size,
            hf_config.linear_num_key_heads,
            hf_config.linear_num_value_heads,
            hf_config.linear_key_head_dim,
            hf_config.linear_value_head_dim,
            hf_config.linear_conv_kernel_dim,
            num_spec,
        )

    @classmethod
    def get_mamba_state_copy_func(cls) -> tuple:
        return MambaStateCopyFuncCalculator.gated_delta_net_state_copy_func()


class Qwen3_5ForCausalLM(Qwen3_5ForCausalLMBase):
    pass"""


def _patch_in_container(container: str) -> None:
    """Run all patches inside the running container `container`."""
    script = f"""
import pathlib, sys

errors = []
def patch(path, old, new, name):
    p = pathlib.Path(path)
    src = p.read_text()
    if old in src:
        if new in src:
            print(f"{{name}}: already patched")
            return
        p.write_text(src.replace(old, new))
        print(f"{{name}}: patched")
    else:
        print(f"{{name}}: source string not found (already patched or upstream changed?)")

registry = "{VLLM_DIR}/registry.py"
patch(registry, {repr(REGISTRY_OLD)}, {repr(REGISTRY_NEW)}, "registry.py")

qwen35 = "{VLLM_DIR}/qwen3_5.py"
patch(qwen35, {repr(QWEN35_IMPORTS_OLD)}, {repr(QWEN35_IMPORTS_NEW)}, "qwen3_5.py: typing imports")
patch(qwen35, {repr(QWEN35_PROCINFO_OLD)}, {repr(QWEN35_PROCINFO_NEW)}, "qwen3_5.py: Qwen3_5ProcessingInfo")
patch(qwen35, {repr(QWEN35_BASE_CLASS_OLD)}, {repr(QWEN35_BASE_CLASS_NEW)}, "qwen3_5.py: base class IsHybrid")
patch(qwen35, {repr(QWEN35_LOAD_OLD)}, {repr(QWEN35_LOAD_NEW)}, "qwen3_5.py: load_weights mapper")
patch(qwen35, {repr(QWEN35_MAMBA_METHODS_OLD)}, {repr(QWEN35_MAMBA_METHODS_NEW)}, "qwen3_5.py: mamba state methods")
"""
    res = subprocess.run(
        ["docker", "exec", container, "python3", "-c", script],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(res.stdout, file=sys.stdout)
        print(res.stderr, file=sys.stderr)
        raise SystemExit(res.returncode)
    print(res.stdout, end="")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--no-build", action="store_true",
                   help="Stage patches to /tmp only (no docker commit)")
    p.add_argument("--base-image", default=BASE_IMAGE)
    p.add_argument("--target-image", default=TARGET_IMAGE)
    p.add_argument("--container", default="vllm-patchbox",
                   help="container name to use (created+removed)")
    args = p.parse_args()

    # Make sure base image is pulled.
    subprocess.run(["docker", "pull", args.base_image], check=True)

    # Spin up an ephemeral container with a long sleep.
    subprocess.run(["docker", "rm", "-f", args.container], check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    cid = subprocess.check_output([
        "docker", "run", "-d", "--name", args.container,
        "--entrypoint", "",
        args.base_image, "sleep", "600",
    ]).decode().strip()
    print(f"started container {args.container} ({cid[:12]})")

    try:
        _patch_in_container(args.container)

        if args.no_build:
            print("--no-build: skipping docker commit")
            return

        # Commit the patched filesystem to a new image.
        new_id = subprocess.check_output([
            "docker", "commit", args.container, args.target_image,
        ]).decode().strip()
        print(f"committed {args.target_image} ({new_id[:12]})")
    finally:
        subprocess.run(["docker", "rm", "-f", args.container],
                       check=False, stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
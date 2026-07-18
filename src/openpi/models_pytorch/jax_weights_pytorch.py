"""Load a local OpenPI JAX base checkpoint directly into a PyTorch model.

This is an in-memory version of ``examples/convert_jax_model_to_pytorch.py``:
it restores the Orbax pytree, maps JAX parameter layouts to the Hugging Face
PyTorch layouts, and copies them into an already constructed PI0 model.  It
does not create a second multi-GiB checkpoint on disk.

Only full/base JAX checkpoints are accepted.  JAX adapter-only checkpoints
need a separate cross-framework adapter conversion because their parameter
tree intentionally omits the base weights.
"""

from __future__ import annotations

import gc
import logging
import pathlib
from typing import Any

import numpy as np
import torch
from torch import nn

import openpi.models.gemma as _gemma

logger = logging.getLogger(__name__)

JAX_ORBAX_BASE_FORMAT = "jax_orbax"
PYTORCH_SAFETENSORS_BASE_FORMAT = "pytorch_safetensors"


def resolve_jax_params_path(path: pathlib.Path | str) -> pathlib.Path:
    """Resolve either a checkpoint root or its ``params`` directory."""
    raw_path = str(path)
    if raw_path.startswith(("gs://", "http://", "https://")):
        raise ValueError("PyTorch direct-from-JAX loading only accepts a local Orbax checkpoint")
    resolved = pathlib.Path(path).expanduser().resolve()
    if (resolved / "params").is_dir():
        resolved = resolved / "params"
    if not resolved.is_dir():
        raise FileNotFoundError(f"JAX Orbax params directory does not exist: {resolved}")
    return resolved


def canonical_jax_params_path(path: pathlib.Path | str) -> str:
    return str(resolve_jax_params_path(path))


def _projection_state_dict(params: dict[str, Any], model_config) -> dict[str, torch.Tensor]:
    keys = ["action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"]
    if not model_config.pi05:
        keys = ["state_proj", "action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"]

    result = {}
    for key in keys:
        kernel = params[key]["kernel"]
        bias = params[key]["bias"]
        result[f"{key}.weight"] = torch.from_numpy(np.asarray(kernel)).T
        result[f"{key}.bias"] = torch.from_numpy(np.asarray(bias))
    return result


def build_torch_state_dict_from_jax(
    params_path: pathlib.Path | str,
    model_config,
) -> dict[str, torch.Tensor]:
    """Restore and map a full JAX PI0/PI05 base checkpoint."""
    # Keep the mapping used by the offline converter as the single source of
    # truth.  Import lazily so ordinary PyTorch model construction does not
    # initialize Orbax/JAX checkpoint machinery.
    from examples import convert_jax_model_to_pytorch as converter

    params_path = resolve_jax_params_path(params_path)
    initial = converter.slice_initial_orbax_checkpoint(str(params_path), restore_precision="float32")

    class PaliGemmaConfig:
        def __init__(self):
            self.vision_config = type(
                "VisionConfig",
                (),
                {
                    "hidden_size": 1152,
                    "num_hidden_layers": 27,
                    "num_attention_heads": 16,
                    "intermediate_size": 4304,
                    "patch_size": 14,
                    "projection_dim": 2048,
                },
            )()
            self.text_config = type(
                "TextConfig",
                (),
                {
                    "hidden_size": 2048,
                    "num_hidden_layers": 18,
                    "num_attention_heads": 8,
                    "head_dim": 256,
                    "intermediate_size": 16384,
                },
            )()

    paligemma_params, expert_params = converter.slice_paligemma_state_dict(
        initial["paligemma_params"], PaliGemmaConfig()
    )
    expert_config = _gemma.get_config(model_config.action_expert_variant.removesuffix("_lora"))
    expert_params = converter.slice_gemma_state_dict(
        expert_params,
        expert_config,
        num_expert=1,
        checkpoint_dir=str(params_path),
        pi05=model_config.pi05,
    )
    projection_params = _projection_state_dict(initial["projection_params"], model_config)
    return {**paligemma_params, **expert_params, **projection_params}


def load_jax_weights(
    model: nn.Module,
    params_path: pathlib.Path | str,
    *,
    model_config=None,
) -> None:
    """Load a local full JAX checkpoint and validate every base parameter key."""
    model_config = model_config or getattr(model, "config", None)
    if model_config is None:
        raise ValueError("model_config is required for direct JAX checkpoint loading")

    resolved_path = resolve_jax_params_path(params_path)
    logger.info("Restoring JAX base checkpoint directly from %s", resolved_path)
    state = build_torch_state_dict_from_jax(resolved_path, model_config)
    incompatible = model.load_state_dict(state, strict=False)

    allowed_missing = {
        name
        for name in model.state_dict()
        if "lora_" in name or name == "paligemma_with_expert.paligemma.lm_head.weight"
    }
    invalid_missing = sorted(set(incompatible.missing_keys) - allowed_missing)
    invalid_unexpected = sorted(incompatible.unexpected_keys)
    if invalid_missing or invalid_unexpected:
        raise ValueError(
            "JAX-to-PyTorch base mapping is incomplete; "
            f"missing={invalid_missing[:10]}, unexpected={invalid_unexpected[:10]}"
        )

    del state
    gc.collect()
    logger.info("Loaded and validated JAX base checkpoint from %s", resolved_path)

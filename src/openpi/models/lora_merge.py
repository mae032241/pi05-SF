"""Fold OpenPI JAX LoRA adapters into their base weights for inference."""

from __future__ import annotations

import dataclasses
import logging
from typing import Literal

import flax.traverse_util
import numpy as np

from openpi.models import gemma
from openpi.models import pi0_config
from openpi.shared import array_typing as at

logger = logging.getLogger(__name__)

_MergeRule = Literal["standard", "vision_qkv", "vision_out", "gemma_out"]


def inference_config_without_lora(config: pi0_config.Pi0Config) -> pi0_config.Pi0Config:
    """Return an architecture-compatible PI0/PI05 config without LoRA branches."""
    return dataclasses.replace(
        config,
        paligemma_variant=config.paligemma_variant.removesuffix("_lora"),
        action_expert_variant=config.action_expert_variant.removesuffix("_lora"),
        vision_train_mode="full",
    )


def merge_lora_params(params: at.Params, config: pi0_config.Pi0Config) -> tuple[at.Params, pi0_config.Pi0Config]:
    """Fold every JAX LoRA pair into its base weight and remove adapter leaves.

    The adapter checkpoint may also contain ordinary trainable parameters, such
    as the action projections. Those leaves are preserved unchanged. Large
    scanned weights are merged one layer/group at a time to avoid constructing
    a full-model-sized float32 delta tensor.
    """
    flat_params = dict(flax.traverse_util.flatten_dict(params))
    lora_a_paths = sorted(
        (path for path in flat_params if _is_lora_a(path[-1])),
        key=lambda path: "/".join(map(str, path)),
    )
    if not lora_a_paths:
        raise ValueError("Cannot merge LoRA for inference: checkpoint contains no LoRA parameters")

    merged_weights = 0
    merged_bytes = 0
    for a_path in lora_a_paths:
        b_path, target_path, rule = _matching_paths_and_rule(a_path)
        if b_path not in flat_params:
            raise ValueError(f"Missing LoRA B parameter for {_format_path(a_path)}")
        if target_path not in flat_params:
            raise ValueError(
                f"Missing base weight {_format_path(target_path)} for LoRA parameter {_format_path(a_path)}"
            )

        lora_a = flat_params[a_path]
        lora_b = flat_params[b_path]
        scale = _scaling_value(a_path, lora_a, config)
        merged = _merge_pair(flat_params[target_path], lora_a, lora_b, scale=scale, rule=rule)
        flat_params[target_path] = merged
        merged_weights += 1
        merged_bytes += merged.nbytes
        del flat_params[a_path]
        del flat_params[b_path]

    leftover = sorted(_format_path(path) for path in flat_params if "lora" in str(path[-1]).lower())
    if leftover:
        raise ValueError(f"Unrecognized LoRA parameters remain after merge: {leftover[:5]}")

    logger.info(
        "Folded %d LoRA weight pairs into %.2f GiB of base weights for inference",
        merged_weights,
        merged_bytes / 1024**3,
    )
    merged_params = flax.traverse_util.unflatten_dict(flat_params)
    return merged_params, inference_config_without_lora(config)


def _is_lora_a(name: object) -> bool:
    name = str(name)
    return name == "lora_a" or name.endswith("_lora_a")


def _matching_paths_and_rule(path: tuple[object, ...]) -> tuple[tuple[object, ...], tuple[object, ...], _MergeRule]:
    parent = path[:-1]
    leaf = str(path[-1])
    parts = tuple(map(str, parent))
    is_vision = "img" in parts
    is_language = "llm" in parts

    if leaf.endswith("_lora_a") and leaf != "lora_a":
        target_name = leaf.removesuffix("_lora_a")
        return (*parent, f"{target_name}_lora_b"), (*parent, target_name), "standard"

    if leaf != "lora_a":
        raise ValueError(f"Unsupported LoRA parameter name: {_format_path(path)}")

    b_path = (*parent, "lora_b")
    module_name = str(parent[-1])
    if is_vision:
        target_path = (*parent, "kernel")
        if module_name in {"query", "key", "value"}:
            return b_path, target_path, "vision_qkv"
        if module_name == "out":
            return b_path, target_path, "vision_out"
        return b_path, target_path, "standard"
    if is_language:
        rule: _MergeRule = "gemma_out" if module_name.startswith("attn_vec_einsum") else "standard"
        return b_path, (*parent, "w"), rule
    raise ValueError(f"Cannot identify LoRA module for {_format_path(path)}")


def _scaling_value(path: tuple[object, ...], lora_a: object, config: pi0_config.Pi0Config) -> float:
    parts = tuple(map(str, path))
    rank = np.asarray(lora_a).shape[-1]
    if "img" in parts:
        if rank != config.vision_lora_rank:
            raise ValueError(f"Vision LoRA rank mismatch at {_format_path(path)}: {rank} != {config.vision_lora_rank}")
        return config.vision_lora_alpha / config.vision_lora_rank

    if "llm" not in parts:
        raise ValueError(f"Cannot determine LoRA scaling for {_format_path(path)}")
    module_name = parts[-2]
    variant = config.action_expert_variant if module_name.endswith("_1") else config.paligemma_variant
    kind = "ffn" if str(path[-1]).startswith(("gating_einsum", "linear")) else "attn"
    lora_config = gemma.get_config(variant).lora_configs.get(kind)
    if lora_config is None:
        raise ValueError(f"Model variant {variant!r} has no {kind} LoRA config for {_format_path(path)}")
    if rank != lora_config.rank:
        raise ValueError(f"Gemma LoRA rank mismatch at {_format_path(path)}: {rank} != {lora_config.rank}")
    return lora_config.scaling_value


def _merge_pair(target: object, lora_a: object, lora_b: object, *, scale: float, rule: _MergeRule) -> np.ndarray:
    """Merge one possibly-scanned LoRA pair without a full scanned delta."""
    target_array = np.asarray(target)
    a_array = np.asarray(lora_a)
    b_array = np.asarray(lora_b)
    core_dims = {
        "standard": (2, 2, 2),
        "vision_qkv": (3, 2, 3),
        "vision_out": (3, 3, 2),
        "gemma_out": (3, 3, 3),
    }[rule]
    target_core_dims, a_core_dims, b_core_dims = core_dims
    prefix_shape = target_array.shape[:-target_core_dims]
    if a_array.shape[:-a_core_dims] != prefix_shape or b_array.shape[:-b_core_dims] != prefix_shape:
        raise ValueError(
            f"LoRA scan/group prefix mismatch for {rule}: target={target_array.shape}, "
            f"a={a_array.shape}, b={b_array.shape}"
        )

    # Orbax restores NumPy arrays as writable buffers. Reuse those buffers to
    # keep peak host memory bounded; copy only for an immutable input.
    result = target_array if target_array.flags.writeable else np.array(target_array, copy=True)
    indices = np.ndindex(prefix_shape) if prefix_shape else [()]
    for index in indices:
        target_slice = np.asarray(result[index])
        a_slice = np.asarray(a_array[index], dtype=np.float32)
        b_slice = np.asarray(b_array[index], dtype=np.float32)
        if rule == "standard":
            delta = a_slice @ b_slice
        elif rule == "vision_qkv":
            delta = np.einsum("ir,rhd->ihd", a_slice, b_slice, optimize=True)
        elif rule == "vision_out":
            delta = np.einsum("hdr,ro->hdo", a_slice, b_slice, optimize=True)
        else:
            # OpenPI Gemma's output-projection LoRA removes the contracted head
            # label from the intermediate equation. B is therefore summed over
            # its head axis instead of being paired head-by-head with A.
            delta = np.einsum("nhr,mro->nho", a_slice, b_slice, optimize=True)
        if delta.shape != target_slice.shape:
            raise ValueError(f"Merged LoRA shape mismatch for {rule}: target={target_slice.shape}, delta={delta.shape}")
        merged_slice = target_slice.astype(np.float32) + np.float32(scale) * delta
        result[index] = merged_slice.astype(result.dtype)
    return result


def _format_path(path: tuple[object, ...]) -> str:
    return "/".join(map(str, path))

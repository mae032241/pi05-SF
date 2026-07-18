"""PyTorch full and adapter-only checkpoint helpers."""

from __future__ import annotations

import json
import pathlib
import shutil
from typing import Any

import safetensors.torch
import torch
from torch import nn

from openpi.models_pytorch import jax_weights_pytorch
from openpi.models_pytorch import lora_pytorch

PYTORCH_ADAPTER_FORMAT = "openpi_pytorch_trainable_adapter_v1"
ADAPTER_FILENAME = "adapter.safetensors"
FULL_MODEL_FILENAME = "model.safetensors"


def adapter_metadata_path(checkpoint_dir: pathlib.Path | str) -> pathlib.Path:
    return pathlib.Path(checkpoint_dir) / "assets" / "adapter_metadata.json"


def read_adapter_metadata(checkpoint_dir: pathlib.Path | str) -> dict[str, Any] | None:
    path = adapter_metadata_path(checkpoint_dir)
    if not path.is_file():
        return None
    metadata = json.loads(path.read_text())
    return metadata if metadata.get("format") == PYTORCH_ADAPTER_FORMAT else None


def write_adapter_metadata(checkpoint_dir: pathlib.Path | str, metadata: dict[str, Any]) -> None:
    if metadata.get("format") != PYTORCH_ADAPTER_FORMAT:
        raise ValueError(f"Unexpected PyTorch adapter format: {metadata.get('format')}")
    path = adapter_metadata_path(checkpoint_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metadata, indent=2))


def resolve_full_model_path(path: pathlib.Path | str) -> pathlib.Path:
    path = pathlib.Path(path).expanduser()
    return path / FULL_MODEL_FILENAME if path.is_dir() else path


def canonical_model_path(path: pathlib.Path | str) -> str:
    """Return a stable base path for adapter metadata."""
    return str(pathlib.Path(path).expanduser().resolve())


def canonical_base_model_path(path: pathlib.Path | str, base_model_format: str) -> str:
    if base_model_format == jax_weights_pytorch.JAX_ORBAX_BASE_FORMAT:
        return jax_weights_pytorch.canonical_jax_params_path(path)
    if base_model_format == jax_weights_pytorch.PYTORCH_SAFETENSORS_BASE_FORMAT:
        return canonical_model_path(path)
    raise ValueError(f"Unsupported PyTorch adapter base model format: {base_model_format}")


def prune_checkpoints(
    checkpoint_dir: pathlib.Path | str,
    *,
    latest_step: int,
    keep_period: int | None,
) -> list[int]:
    """Match Orbax's max_to_keep=1 plus optional periodic retention."""
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    removed = []
    for path in checkpoint_dir.iterdir():
        if not path.is_dir() or not path.name.isdigit():
            continue
        step = int(path.name)
        if step == latest_step or (keep_period is not None and step % keep_period == 0):
            continue
        shutil.rmtree(path)
        removed.append(step)
    return sorted(removed)


def save_adapter_weights(model: nn.Module, path: pathlib.Path | str) -> None:
    state = {
        name: value.detach().cpu().contiguous()
        for name, value in lora_pytorch.adapter_state_dict(model).items()
    }
    if not state:
        raise ValueError("Refusing to save an empty PyTorch adapter")
    safetensors.torch.save_file(state, str(path))


def load_adapter_weights(
    model: nn.Module,
    path: pathlib.Path | str,
    *,
    device: str | torch.device = "cpu",
) -> None:
    state = safetensors.torch.load_file(str(path), device=str(device))
    expected = lora_pytorch.trainable_parameter_names(model)
    missing = sorted(expected - set(state))
    unknown = sorted(set(state) - set(model.state_dict()))
    if missing or unknown:
        raise ValueError(f"Invalid PyTorch adapter; missing={missing[:5]}, unknown={unknown[:5]}")
    model.load_state_dict(state, strict=False)


def load_full_weights(
    model: nn.Module,
    path: pathlib.Path | str,
    *,
    device: str | torch.device = "cpu",
    strict: bool = False,
) -> None:
    path = resolve_full_model_path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PyTorch base model does not exist: {path}")
    safetensors.torch.load_model(model, str(path), strict=strict, device=str(device))


def load_adapter_model_weights(
    model: nn.Module,
    checkpoint_dir: pathlib.Path | str,
    *,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint_dir = pathlib.Path(checkpoint_dir)
    metadata = read_adapter_metadata(checkpoint_dir)
    if metadata is None:
        raise ValueError(f"Not a PyTorch adapter checkpoint: {checkpoint_dir}")
    base_path = metadata.get("base_model_path")
    if not base_path:
        raise ValueError(f"PyTorch adapter does not specify base_model_path: {checkpoint_dir}")
    base_format = metadata.get(
        "base_model_format",
        jax_weights_pytorch.PYTORCH_SAFETENSORS_BASE_FORMAT,
    )
    if base_format == jax_weights_pytorch.JAX_ORBAX_BASE_FORMAT:
        jax_weights_pytorch.load_jax_weights(model, base_path)
    elif base_format == jax_weights_pytorch.PYTORCH_SAFETENSORS_BASE_FORMAT:
        load_full_weights(model, base_path, device=device, strict=False)
    else:
        raise ValueError(f"Unsupported PyTorch adapter base model format: {base_format}")
    load_adapter_weights(model, checkpoint_dir / ADAPTER_FILENAME, device=device)
    return metadata

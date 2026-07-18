import json
import logging
import os
import pathlib
from typing import Any

import flax.nnx as nnx
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import lora_merge
from openpi.models import pi0_config
import openpi.models.model as _model
from openpi.models_pytorch import lora_pytorch
import openpi.policies.policy as _policy
import openpi.shared.download as download
from openpi.training import checkpoints as _checkpoints
from openpi.training import checkpoints_pytorch
from openpi.training import config as _config
from openpi.training import weight_loaders
import openpi.transforms as transforms


def _load_adapter_model(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path,
    *,
    merge_lora_for_inference: bool = False,
):
    metadata_path = checkpoint_dir / "assets" / "adapter_metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("format") != "openpi_trainable_adapter_v1":
        raise ValueError(f"Unsupported adapter checkpoint format in {metadata_path}")
    base_params_path = metadata.get("base_params_path")
    if not base_params_path:
        raise ValueError(f"Adapter checkpoint does not specify base_params_path: {metadata_path}")

    model_shape = nnx.eval_shape(train_config.model.create, jax.random.key(0))
    expected_params = nnx.state(model_shape).to_pure_dict()
    base_params = weight_loaders.CheckpointWeightLoader(base_params_path).load(expected_params)
    adapter_params = _model.restore_params(checkpoint_dir / "params", restore_type=np.ndarray)

    expected_flat = flax.traverse_util.flatten_dict(expected_params, sep="/")
    merged_flat = flax.traverse_util.flatten_dict(base_params, sep="/")
    adapter_flat = flax.traverse_util.flatten_dict(adapter_params, sep="/")
    unknown = sorted(set(adapter_flat) - set(expected_flat))
    if unknown:
        raise ValueError(f"Adapter checkpoint contains unknown parameters: {unknown[:5]}")
    merged_flat.update(adapter_flat)
    missing = sorted(set(expected_flat) - set(merged_flat))
    unresolved = sorted(k for k, v in merged_flat.items() if isinstance(v, jax.ShapeDtypeStruct))
    if missing or unresolved:
        raise ValueError(
            f"Could not merge adapter with base checkpoint; missing={missing[:5]}, unresolved={unresolved[:5]}"
        )
    merged_params = flax.traverse_util.unflatten_dict(merged_flat, sep="/")
    if merge_lora_for_inference:
        if not isinstance(train_config.model, pi0_config.Pi0Config):
            raise TypeError("Inference-time LoRA merge currently supports Pi0Config/PI05 JAX models only")
        merged_params, inference_model_config = lora_merge.merge_lora_params(merged_params, train_config.model)
        logging.info("Loading JAX policy with LoRA weights folded into the base model")
        return inference_model_config.load(merged_params)
    return train_config.model.load(merged_params)


def create_trained_policy(
    train_config: _config.TrainConfig,
    checkpoint_dir: pathlib.Path | str,
    *,
    repack_transforms: transforms.Group | None = None,
    sample_kwargs: dict[str, Any] | None = None,
    default_prompt: str | None = None,
    norm_stats: dict[str, transforms.NormStats] | None = None,
    pytorch_device: str | None = None,
    merge_lora_for_inference: bool = False,
) -> _policy.Policy:
    """Create a policy from a trained checkpoint.

    Args:
        train_config: The training config to use to create the model.
        checkpoint_dir: The directory to load the model from.
        repack_transforms: Optional transforms that will be applied before any other transforms.
        sample_kwargs: The kwargs to pass to the `sample_actions` method. If not provided, the default
            kwargs will be used.
        default_prompt: The default prompt to use for the policy. Will inject the prompt into the input
            data if it doesn't already exist.
        norm_stats: The norm stats to use for the policy. If not provided, the norm stats will be loaded
            from the checkpoint directory.
        pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda", "cuda:0").
                      If None and is_pytorch=True, will use "cuda" if available, otherwise "cpu".
        merge_lora_for_inference: Fold a JAX adapter into its base weights once
            at load time and instantiate a model without LoRA branches.

    Note:
        The function automatically detects whether the model is PyTorch-based by checking for the
        presence of "model.safensors" in the checkpoint directory.
    """
    repack_transforms = repack_transforms or transforms.Group()
    checkpoint_dir = download.maybe_download(str(checkpoint_dir))

    # Check whether this is a full PyTorch model or an adapter-only one.
    weight_path = os.path.join(checkpoint_dir, "model.safetensors")
    adapter_metadata_path = checkpoint_dir / "assets" / "adapter_metadata.json"
    pytorch_adapter_metadata = checkpoints_pytorch.read_adapter_metadata(checkpoint_dir)
    is_pytorch = os.path.exists(weight_path) or pytorch_adapter_metadata is not None

    logging.info("Loading model...")
    if pytorch_adapter_metadata is not None:
        model = train_config.model.load_pytorch_adapter(train_config, checkpoint_dir)
        if merge_lora_for_inference:
            lora_pytorch.merge_lora_modules_(model)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    elif is_pytorch:
        model = train_config.model.load_pytorch(train_config, weight_path)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
    elif adapter_metadata_path.is_file():
        model = _load_adapter_model(
            train_config,
            checkpoint_dir,
            merge_lora_for_inference=merge_lora_for_inference,
        )
    elif train_config.pytorch_load_from_jax:
        model = train_config.model.load_pytorch_from_jax(train_config, checkpoint_dir)
        model.paligemma_with_expert.to_bfloat16_for_selected_params("bfloat16")
        is_pytorch = True
    else:
        model = train_config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    data_config = train_config.data.create(train_config.assets_dirs, train_config.model)
    if norm_stats is None:
        # We are loading the norm stats from the checkpoint instead of the config assets dir to make sure
        # that the policy is using the same normalization stats as the original training process.
        if data_config.asset_id is None:
            raise ValueError("Asset id is required to load norm stats.")
        norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

    # Determine the device to use for PyTorch models
    if is_pytorch and pytorch_device is None:
        try:
            import torch  # noqa: PLC0415

            pytorch_device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            pytorch_device = "cpu"

    return _policy.Policy(
        model,
        transforms=[
            *repack_transforms.inputs,
            transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
            *repack_transforms.outputs,
        ],
        sample_kwargs=sample_kwargs,
        metadata=train_config.policy_metadata,
        is_pytorch=is_pytorch,
        pytorch_device=pytorch_device if is_pytorch else None,
    )

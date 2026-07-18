# JAX and PyTorch LoRA checkpoints

JAX training keeps the existing full-checkpoint behavior by default:

```python
TrainConfig(
    ...,
    lora_save=False,
)
```

Set `lora_save=True` to save only parameters selected by
`config.trainable_filter`. The checkpoint records the base params path in
`assets/adapter_metadata.json`. Resume and inference first load that base and
then overlay the saved trainable parameters.

```python
TrainConfig(
    ...,
    weight_loader=CheckpointWeightLoader("/path/to/pi05_base/params"),
    lora_save=True,
    ema_decay=None,
)
```

Adapter checkpoints are not standalone: the base params path recorded in the
metadata must remain accessible. `lora_save` currently requires
`ema_decay=None` so the checkpoint does not need two adapter parameter sets.

## Vision training mode

`Pi0Config.vision_train_mode` supports two modes:

- `"lora"`: adds LoRA to SigLIP q/k/v/out attention projections and both MLP
  projections, and freezes the base vision parameters.
- `"full"`: keeps the previous behavior and trains the full vision encoder.

```python
Pi0Config(
    pi05=True,
    paligemma_variant="gemma_2b_lora",
    action_expert_variant="gemma_300m_lora",
    vision_train_mode="lora",  # or "full"
    vision_lora_rank=16,
    vision_lora_alpha=16.0,
)
```

For configs with `auto_freeze_filter=True`, a CLI model override also updates
the freeze filter:

```bash
# Vision LoRA (the dual-shoe config default)
.venv/bin/python scripts/train.py \
  pi05_robotwin_place_dual_shoes_lora \
  --exp-name=dual_shoe_lora \
  --overwrite

# Full vision fine-tuning while retaining adapter-only checkpoint packaging
.venv/bin/python scripts/train.py \
  pi05_robotwin_place_dual_shoes_lora \
  --model.vision-train-mode=full \
  --exp-name=dual_shoe_vision_full \
  --overwrite
```

With the current dual-shoe config, the expected trainable parameter payload is
approximately 0.23 GiB for vision LoRA and 1.74 GiB for full vision training.
Optimizer state and Orbax metadata are additional to these values.

## PyTorch parity implementation

The PyTorch implementation does not use PEFT's conventional 2-D adapters.
OpenPI JAX Gemma applies LoRA to multi-dimensional einsum weights, including
head-wise factors and its contracted-head output equation. The Torch modules in
`openpi.models_pytorch.lora_pytorch` reproduce those equations directly.

For the dual-shoe configuration the JAX and Torch models have exactly the same
parameter partition in both supported vision modes:

- Vision LoRA: total `3,412,116,752`, trainable `60,848,672`.
- Full vision: total `3,403,421,456`, trainable `466,957,072`.

The parity tests copy identical parameters into JAX and Torch and compare the
inference outputs of grouped Gemma q/k/v projections, the contracted-head
output projection, the Gemma FFN, Vision dense projections, and a complete
Vision attention layer. CUDA JAX is forced to FP32 for this comparison because
its default TF32 reduction is a backend precision choice rather than a model
equation difference.

With `lora_save=True`, PyTorch checkpoints contain `adapter.safetensors`, the
optimizer state, metadata, and normalization assets. The full base model is not
copied. Resume and inference load `base_model_path` from
`assets/adapter_metadata.json` before applying the adapter.
`--resume` reuses this recorded base path, so it does not require the base path
to be repeated on the command line. Like the JAX checkpoint manager, Torch
keeps only the latest non-periodic checkpoint plus `keep_period` checkpoints.

The RoboTwin PiSF model server folds both JAX and PyTorch LoRA adapters into
their base weights once at startup by default. PyTorch LoRA modules are replaced
with ordinary `nn.Linear` layers after fusion. This avoids adapter operations in
the inference graph and does not write a full merged checkpoint to disk. Set
`PISF_MERGE_LORA_FOR_INFERENCE=false` when launching `eval_double_env.sh` to
retain the unfused path for an A/B comparison.

PyTorch needs a converted base safetensors model. Conversion writes several
GiB, so do this only after enough disk has been provisioned:

```bash
.venv/bin/python examples/convert_jax_model_to_pytorch.py \
  --checkpoint-dir /root/autodl-tmp/data/openpi/openpi-assets/checkpoints/pi05_base \
  --config-name pi05_robotwin_place_dual_shoes_lora \
  --output-path /path/with/enough/space/pi05_base_torch \
  --precision bfloat16
```

Alternatively, PyTorch can restore the local JAX Orbax base directly in
memory, without writing `model.safetensors`:

```python
TrainConfig(
    ...,
    weight_loader=CheckpointWeightLoader("/path/to/pi05_base/params"),
    pytorch_load_from_jax=True,
    # Optional; defaults to CheckpointWeightLoader.params_path.
    pytorch_jax_weight_path=None,
)
```

The loader accepts a local checkpoint root or its `params/` directory. It
validates that every non-LoRA Torch base parameter was mapped; it permits only
the injected LoRA parameters and the tied PaliGemma LM head to be absent. It
does not download remote checkpoints and does not support JAX adapter-only
checkpoints. Restoring is disk-efficient but temporarily holds the restored
JAX arrays and mapped Torch tensors in host memory.

Standard PI05 Torch LoRA training:

```bash
.venv/bin/python scripts/train_pytorch.py \
  pi05_robotwin_place_dual_shoes_lora \
  --exp-name dual_shoe_torch_lora \
  --pytorch-weight-path /path/with/enough/space/pi05_base_torch
```

The same training directly from a JAX base is:

```bash
.venv/bin/python scripts/train_pytorch.py \
  pi05_robotwin_place_dual_shoes_lora \
  --exp-name dual_shoe_torch_lora \
  --pytorch-load-from-jax
```

Spatial-Forcing Torch LoRA training additionally requires local VGGT weights:

```bash
.venv/bin/python scripts/train_align_pytorch.py \
  pi05_robotwin_place_dual_shoes_lora_torch_sf \
  --exp-name dual_shoe_torch_sf_lora \
  --vggt-weight-path /path/to/vggt
```

`pi05_robotwin_place_dual_shoes_lora_torch_sf` enables
`pytorch_load_from_jax` and points at the existing local PI05 base by default.
Adapter metadata records `base_model_format: jax_orbax`, so resume and policy
inference restore the same JAX base before applying `adapter.safetensors`.

No model or VGGT weights are downloaded automatically by these commands.

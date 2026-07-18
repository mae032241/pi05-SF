import pathlib

import safetensors.torch
import torch
from torch import nn

from openpi.models_pytorch import jax_weights_pytorch
from openpi.models_pytorch import lora_pytorch
from openpi.training import checkpoints_pytorch


class _TinyAdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = lora_pytorch.LoRALinear(4, 5, rank=2, alpha=2, bias=True, zero_b=True)
        self.head = nn.Linear(5, 3)
        self.proj.weight.requires_grad_(requires_grad=False)
        self.proj.bias.requires_grad_(requires_grad=False)


def test_pytorch_adapter_round_trip(tmp_path: pathlib.Path):
    base = _TinyAdapterModel()
    base_state = {name: value.detach().clone() for name, value in base.state_dict().items() if "lora_" not in name}
    base_dir = tmp_path / "base"
    base_dir.mkdir()
    safetensors.torch.save_file(base_state, str(base_dir / checkpoints_pytorch.FULL_MODEL_FILENAME))

    trained = _TinyAdapterModel()
    checkpoints_pytorch.load_full_weights(trained, base_dir, strict=False)
    with torch.no_grad():
        for _name, parameter in trained.named_parameters():
            if parameter.requires_grad:
                parameter.add_(0.25)

    checkpoint_dir = tmp_path / "adapter"
    checkpoint_dir.mkdir()
    checkpoints_pytorch.save_adapter_weights(
        trained,
        checkpoint_dir / checkpoints_pytorch.ADAPTER_FILENAME,
    )
    checkpoints_pytorch.write_adapter_metadata(
        checkpoint_dir,
        {
            "format": checkpoints_pytorch.PYTORCH_ADAPTER_FORMAT,
            "base_model_path": str(base_dir),
            "train_config_name": "tiny",
            "vision_train_mode": "lora",
        },
    )

    restored = _TinyAdapterModel()
    checkpoints_pytorch.load_adapter_model_weights(restored, checkpoint_dir)
    for name, expected in trained.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected)

    inputs = torch.randn(2, 4)
    with torch.no_grad():
        expected_output = restored.head(restored.proj(inputs))
    assert lora_pytorch.merge_lora_modules_(restored) == 1
    assert type(restored.proj) is nn.Linear
    assert not any("lora_" in name for name, _ in restored.named_parameters())
    with torch.no_grad():
        merged_output = restored.head(restored.proj(inputs))
    torch.testing.assert_close(merged_output, expected_output, rtol=2e-5, atol=2e-5)

    saved_adapter = safetensors.torch.load_file(str(checkpoint_dir / checkpoints_pytorch.ADAPTER_FILENAME))
    assert set(saved_adapter) == lora_pytorch.trainable_parameter_names(trained)


def test_prune_checkpoints_matches_orbax_retention(tmp_path: pathlib.Path):
    for step in (5, 10, 15, 20, 23):
        (tmp_path / str(step)).mkdir()
    (tmp_path / "tmp_24").mkdir()

    removed = checkpoints_pytorch.prune_checkpoints(
        tmp_path,
        latest_step=23,
        keep_period=10,
    )

    assert removed == [5, 15]
    assert {path.name for path in tmp_path.iterdir()} == {"10", "20", "23", "tmp_24"}


def test_pytorch_adapter_can_restore_a_jax_base(tmp_path: pathlib.Path, monkeypatch):
    base = _TinyAdapterModel()
    base_state = {name: value.detach().clone() for name, value in base.state_dict().items() if "lora_" not in name}
    trained = _TinyAdapterModel()
    trained.load_state_dict(base_state, strict=False)
    with torch.no_grad():
        for _name, parameter in trained.named_parameters():
            if parameter.requires_grad:
                parameter.add_(0.5)

    checkpoint_dir = tmp_path / "jax_base_adapter"
    checkpoint_dir.mkdir()
    checkpoints_pytorch.save_adapter_weights(trained, checkpoint_dir / checkpoints_pytorch.ADAPTER_FILENAME)
    checkpoints_pytorch.write_adapter_metadata(
        checkpoint_dir,
        {
            "format": checkpoints_pytorch.PYTORCH_ADAPTER_FORMAT,
            "base_model_path": str(tmp_path / "jax_params"),
            "base_model_format": jax_weights_pytorch.JAX_ORBAX_BASE_FORMAT,
            "train_config_name": "tiny",
        },
    )

    calls = []

    def fake_load_jax_weights(model, path):
        calls.append(path)
        model.load_state_dict(base_state, strict=False)

    monkeypatch.setattr(jax_weights_pytorch, "load_jax_weights", fake_load_jax_weights)
    restored = _TinyAdapterModel()
    metadata = checkpoints_pytorch.load_adapter_model_weights(restored, checkpoint_dir)

    assert calls == [str(tmp_path / "jax_params")]
    assert metadata["base_model_format"] == jax_weights_pytorch.JAX_ORBAX_BASE_FORMAT
    for name, expected in trained.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], expected)

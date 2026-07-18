import pathlib

import pytest
import torch
from torch import nn

from openpi.models_pytorch import jax_weights_pytorch
from openpi.models_pytorch import lora_pytorch


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = object()
        self.proj = lora_pytorch.LoRALinear(3, 4, rank=2, alpha=2, bias=True, zero_b=True)


def test_load_jax_weights_validates_and_allows_only_lora_missing(tmp_path: pathlib.Path, monkeypatch):
    expected = _TinyModel()
    mapped = {
        name: torch.full_like(value, 0.25)
        for name, value in expected.state_dict().items()
        if "lora_" not in name
    }
    monkeypatch.setattr(
        jax_weights_pytorch,
        "build_torch_state_dict_from_jax",
        lambda _path, _config: mapped,
    )

    jax_weights_pytorch.load_jax_weights(expected, tmp_path)

    assert torch.all(expected.proj.weight == 0.25)
    assert torch.all(expected.proj.bias == 0.25)


def test_load_jax_weights_rejects_missing_base_parameter(tmp_path: pathlib.Path, monkeypatch):
    model = _TinyModel()
    monkeypatch.setattr(
        jax_weights_pytorch,
        "build_torch_state_dict_from_jax",
        lambda _path, _config: {"proj.weight": torch.zeros_like(model.proj.weight)},
    )

    with pytest.raises(ValueError, match=r"proj\.bias"):
        jax_weights_pytorch.load_jax_weights(model, tmp_path)

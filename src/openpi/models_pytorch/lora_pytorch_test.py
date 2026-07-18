import dataclasses

import flax.linen as nn
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
from torch import nn as torch_nn
import torch.nn.functional as F  # noqa: N812

from openpi.models import gemma
from openpi.models import lora as jax_lora
from openpi.models import siglip
from openpi.models_pytorch import lora_pytorch
from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
from openpi.training import config as training_config


def _random(shape, seed):
    return np.random.default_rng(seed).normal(size=shape).astype(np.float32)


def test_grouped_output_matches_jax_gemma_einsum():
    batch, sequence, groups, in_features, group_out, rank = 2, 3, 2, 5, 4, 3
    alpha = 6.0
    x = _random((batch, sequence, in_features), 0)
    weight = _random((groups, in_features, group_out), 1)
    lora_a = _random((groups, in_features, rank), 2)
    lora_b = _random((groups, rank, group_out), 3)

    jax_layer = jax_lora.Einsum(
        weight.shape,
        lora_config=jax_lora.LoRAConfig(rank=rank, alpha=alpha),
    )
    jax_output = jax_layer.apply(
        {"params": {"w": weight, "lora_a": lora_a, "lora_b": lora_b}},
        "BTD,NDH->BTNH",
        jnp.asarray(x),
    )

    torch_layer = lora_pytorch.GroupedOutputLoRALinear(
        in_features,
        groups * group_out,
        groups=groups,
        group_out_features=group_out,
        rank=rank,
        alpha=alpha,
        bias=False,
    )
    with torch.no_grad():
        torch_layer.weight.copy_(torch.from_numpy(weight.transpose(0, 2, 1).reshape(groups * group_out, in_features)))
        torch_layer.lora_a.copy_(torch.from_numpy(lora_a))
        torch_layer.lora_b.copy_(torch.from_numpy(lora_b))
    torch_output = torch_layer(torch.from_numpy(x)).unflatten(-1, (groups, group_out))

    np.testing.assert_allclose(np.asarray(jax_output), torch_output.detach().numpy(), rtol=1e-5, atol=1e-5)


def test_grouped_input_matches_jax_gemma_einsum():
    batch, sequence, groups, group_in, out_features, rank = 2, 3, 2, 4, 5, 3
    alpha = 6.0
    x = _random((batch, sequence, groups, group_in), 4)
    weight = _random((groups, group_in, out_features), 5)
    lora_a = _random((groups, group_in, rank), 6)
    lora_b = _random((groups, rank, out_features), 7)

    jax_layer = jax_lora.Einsum(
        weight.shape,
        lora_config=jax_lora.LoRAConfig(rank=rank, alpha=alpha),
    )
    jax_output = jax_layer.apply(
        {"params": {"w": weight, "lora_a": lora_a, "lora_b": lora_b}},
        "BTNH,NHD->BTD",
        jnp.asarray(x),
    )

    torch_layer = lora_pytorch.GroupedInputLoRALinear(
        groups * group_in,
        out_features,
        groups=groups,
        group_in_features=group_in,
        rank=rank,
        alpha=alpha,
        bias=False,
    )
    with torch.no_grad():
        torch_layer.weight.copy_(torch.from_numpy(weight.transpose(2, 0, 1).reshape(out_features, -1)))
        torch_layer.lora_a.copy_(torch.from_numpy(lora_a))
        torch_layer.lora_b.copy_(torch.from_numpy(lora_b))
    torch_output = torch_layer(torch.from_numpy(x.reshape(batch, sequence, -1)))

    np.testing.assert_allclose(np.asarray(jax_output), torch_output.detach().numpy(), rtol=1e-5, atol=1e-5)


def test_inference_merge_replaces_all_lora_linear_variants():
    layers_and_inputs = [
        (
            lora_pytorch.LoRALinear(5, 7, rank=3, alpha=6.0, bias=True, zero_b=False),
            torch.from_numpy(_random((2, 4, 5), 70)),
        ),
        (
            lora_pytorch.GroupedOutputLoRALinear(
                5,
                8,
                groups=2,
                group_out_features=4,
                rank=3,
                alpha=6.0,
                bias=False,
            ),
            torch.from_numpy(_random((2, 4, 5), 71)),
        ),
        (
            lora_pytorch.GroupedInputLoRALinear(
                8,
                5,
                groups=2,
                group_in_features=4,
                rank=3,
                alpha=6.0,
                bias=False,
            ),
            torch.from_numpy(_random((2, 4, 8), 72)),
        ),
    ]

    for layer, inputs in layers_and_inputs:
        model = torch_nn.Sequential(layer)
        with torch.no_grad():
            expected = model(inputs)
        assert lora_pytorch.merge_lora_modules_(model) == 1
        assert type(model[0]) is torch_nn.Linear
        assert not any("lora_" in name for name, _ in model.named_parameters())
        with torch.no_grad():
            actual = model(inputs)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


def test_bfloat16_inference_merge_has_only_expected_rounding_error():
    torch.manual_seed(0)
    model = torch_nn.Sequential(
        lora_pytorch.LoRALinear(
            32,
            48,
            rank=4,
            alpha=4.0,
            bias=True,
            zero_b=False,
            dtype=torch.bfloat16,
        )
    )
    inputs = torch.randn(2, 5, 32, dtype=torch.bfloat16)
    with torch.no_grad():
        unfused = model(inputs).float()
    assert lora_pytorch.merge_lora_modules_(model) == 1
    with torch.no_grad():
        fused = model(inputs).float()

    max_abs = torch.max(torch.abs(fused - unfused))
    cosine = F.cosine_similarity(fused.flatten(), unfused.flatten(), dim=0)
    assert max_abs <= 0.0078125
    assert cosine >= 0.99999


def test_feed_forward_matches_jax_gemma_lora():
    batch, sequence, features, hidden, rank = 2, 3, 5, 7, 3
    config = jax_lora.LoRAConfig(rank=rank, alpha=rank)
    x = _random((batch, sequence, features), 8)
    gating = _random((2, features, hidden), 9)
    linear = _random((hidden, features), 10)
    gating_a = _random((2, features, rank), 11)
    gating_b = _random((2, rank, hidden), 12)
    linear_a = _random((hidden, rank), 13)
    linear_b = _random((rank, features), 14)

    jax_layer = jax_lora.FeedForward(features=features, hidden_dim=hidden, lora_config=config)
    jax_output = jax_layer.apply(
        {
            "params": {
                "gating_einsum": gating,
                "linear": linear,
                "gating_einsum_lora_a": gating_a,
                "gating_einsum_lora_b": gating_b,
                "linear_lora_a": linear_a,
                "linear_lora_b": linear_b,
            }
        },
        jnp.asarray(x),
    )

    layers = [
        lora_pytorch.LoRALinear(features, hidden, rank=rank, alpha=rank, bias=False, zero_b=False, apply_scaling=False),
        lora_pytorch.LoRALinear(features, hidden, rank=rank, alpha=rank, bias=False, zero_b=False, apply_scaling=False),
        lora_pytorch.LoRALinear(hidden, features, rank=rank, alpha=rank, bias=False, zero_b=False, apply_scaling=False),
    ]
    with torch.no_grad():
        for index, layer in enumerate(layers[:2]):
            layer.weight.copy_(torch.from_numpy(gating[index].T))
            layer.lora_a.copy_(torch.from_numpy(gating_a[index]))
            layer.lora_b.copy_(torch.from_numpy(gating_b[index]))
        layers[2].weight.copy_(torch.from_numpy(linear.T))
        layers[2].lora_a.copy_(torch.from_numpy(linear_a))
        layers[2].lora_b.copy_(torch.from_numpy(linear_b))
    torch_x = torch.from_numpy(x)
    torch_output = layers[2](F.gelu(layers[0](torch_x), approximate="tanh") * layers[1](torch_x))

    np.testing.assert_allclose(np.asarray(jax_output), torch_output.detach().numpy(), rtol=2e-5, atol=2e-5)


def test_vision_dense_matches_jax_and_starts_at_zero_delta():
    batch, sequence, in_features, out_features, rank = 2, 3, 5, 7, 3
    alpha = 6.0
    x = _random((batch, sequence, in_features), 15)
    kernel = _random((in_features, out_features), 16)
    bias = _random((out_features,), 17)
    lora_a = _random((in_features, rank), 18)
    lora_b = _random((rank, out_features), 19)

    jax_layer = siglip.LoRADense(
        out_features,
        lora_config=jax_lora.LoRAConfig(rank=rank, alpha=alpha),
    )
    jax_output = jax_layer.apply(
        {"params": {"kernel": kernel, "bias": bias, "lora_a": lora_a, "lora_b": lora_b}},
        jnp.asarray(x),
    )

    torch_layer = lora_pytorch.LoRALinear(
        in_features,
        out_features,
        rank=rank,
        alpha=alpha,
        bias=True,
        zero_b=True,
    )
    assert torch.count_nonzero(torch_layer.lora_b) == 0
    with torch.no_grad():
        torch_layer.weight.copy_(torch.from_numpy(kernel.T))
        torch_layer.bias.copy_(torch.from_numpy(bias))
        torch_layer.lora_a.copy_(torch.from_numpy(lora_a))
        torch_layer.lora_b.copy_(torch.from_numpy(lora_b))
    torch_output = torch_layer(torch.from_numpy(x))

    np.testing.assert_allclose(np.asarray(jax_output), torch_output.detach().numpy(), rtol=1e-5, atol=1e-5)


def test_full_vision_attention_output_matches_jax():
    batch, sequence, width, heads, rank = 2, 3, 8, 2, 3
    head_dim = width // heads
    alpha = 6.0
    x = _random((batch, sequence, width), 21)

    def small_random(shape, seed):
        return _random(shape, seed) * 0.1

    params = {}
    torch_projections = {}
    for index, name in enumerate(("query", "key", "value"), start=22):
        kernel = small_random((width, heads, head_dim), index)
        bias = small_random((heads, head_dim), index + 10)
        lora_a = small_random((width, rank), index + 20)
        lora_b = small_random((rank, heads, head_dim), index + 30)
        params[name] = {
            "kernel": kernel,
            "bias": bias,
            "lora_a": lora_a,
            "lora_b": lora_b,
        }
        projection = lora_pytorch.LoRALinear(
            width,
            width,
            rank=rank,
            alpha=alpha,
            bias=True,
            zero_b=True,
        )
        with torch.no_grad():
            projection.weight.copy_(torch.from_numpy(kernel.reshape(width, width).T))
            projection.bias.copy_(torch.from_numpy(bias.reshape(width)))
            projection.lora_a.copy_(torch.from_numpy(lora_a))
            projection.lora_b.copy_(torch.from_numpy(lora_b.reshape(rank, width)))
        torch_projections[name] = projection

    out_kernel = small_random((heads, head_dim, width), 60)
    out_bias = small_random((width,), 61)
    out_lora_a = small_random((heads, head_dim, rank), 62)
    out_lora_b = small_random((rank, width), 63)
    params["out"] = {
        "kernel": out_kernel,
        "bias": out_bias,
        "lora_a": out_lora_a,
        "lora_b": out_lora_b,
    }

    jax_attention = siglip.LoRAMultiHeadDotProductAttention(
        num_heads=heads,
        lora_config=jax_lora.LoRAConfig(rank=rank, alpha=alpha),
    )
    # CUDA JAX defaults to TF32 for these einsums while this reference Torch
    # path runs on CPU in FP32. Force the same arithmetic precision so this
    # test measures the parameter layout and layer math, not backend defaults.
    with jax.default_matmul_precision("float32"):
        jax_output = jax_attention.apply({"params": params}, jnp.asarray(x))

    out_projection = lora_pytorch.LoRALinear(
        width,
        width,
        rank=rank,
        alpha=alpha,
        bias=True,
        zero_b=True,
    )
    with torch.no_grad():
        out_projection.weight.copy_(torch.from_numpy(out_kernel.reshape(width, width).T))
        out_projection.bias.copy_(torch.from_numpy(out_bias))
        out_projection.lora_a.copy_(torch.from_numpy(out_lora_a.reshape(width, rank)))
        out_projection.lora_b.copy_(torch.from_numpy(out_lora_b))

    torch_x = torch.from_numpy(x)
    query = torch_projections["query"](torch_x).unflatten(-1, (heads, head_dim))
    key = torch_projections["key"](torch_x).unflatten(-1, (heads, head_dim))
    value = torch_projections["value"](torch_x).unflatten(-1, (heads, head_dim))
    # Flax scales query before the dot product (rather than scaling logits).
    logits = torch.einsum("bqhd,bkhd->bhqk", query / np.sqrt(head_dim), key)
    attended = torch.einsum("bhqk,bkhd->bqhd", torch.softmax(logits, dim=-1), value)
    torch_output = out_projection(attended.flatten(start_dim=-2))

    np.testing.assert_allclose(np.asarray(jax_output), torch_output.detach().numpy(), rtol=2e-5, atol=2e-5)


def test_jax_gelu_assumption_is_tanh_approximation():
    x = jnp.asarray(_random((32,), 20))
    np.testing.assert_allclose(
        np.asarray(nn.gelu(x)),
        F.gelu(torch.from_numpy(np.asarray(x).copy()), approximate="tanh").numpy(),
        rtol=1e-5,
        atol=1e-5,
    )


class _FakeAttention(torch_nn.Module):
    def __init__(self, config):
        super().__init__()
        self.q_proj = torch_nn.Linear(config.width, config.num_heads * config.head_dim, bias=False)
        self.k_proj = torch_nn.Linear(config.width, config.num_kv_heads * config.head_dim, bias=False)
        self.v_proj = torch_nn.Linear(config.width, config.num_kv_heads * config.head_dim, bias=False)
        self.o_proj = torch_nn.Linear(config.num_heads * config.head_dim, config.width, bias=False)


class _FakeMLP(torch_nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = torch_nn.Linear(config.width, config.mlp_dim, bias=False)
        self.up_proj = torch_nn.Linear(config.width, config.mlp_dim, bias=False)
        self.down_proj = torch_nn.Linear(config.mlp_dim, config.width, bias=False)


class _FakeGemmaLayer(torch_nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self_attn = _FakeAttention(config)
        self.mlp = _FakeMLP(config)


class _FakeGemma(torch_nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = torch_nn.ModuleList([_FakeGemmaLayer(config)])


class _FakeVisionLayer(torch_nn.Module):
    def __init__(self, width=8, hidden=12):
        super().__init__()
        self.self_attn = torch_nn.Module()
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(self.self_attn, name, torch_nn.Linear(width, width))
        self.mlp = torch_nn.Module()
        self.mlp.fc1 = torch_nn.Linear(width, hidden)
        self.mlp.fc2 = torch_nn.Linear(hidden, width)


def test_injection_uses_grouped_gemma_and_standard_vision_lora():
    config = gemma.Config(
        width=8,
        depth=1,
        mlp_dim=12,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        lora_configs={
            "attn": jax_lora.LoRAConfig(rank=2, alpha=2),
            "ffn": jax_lora.LoRAConfig(rank=3, alpha=3),
        },
    )
    gemma_model = _FakeGemma(config)
    lora_pytorch.inject_gemma_lora(gemma_model, config)
    layer = gemma_model.layers[0]
    assert isinstance(layer.self_attn.q_proj, lora_pytorch.GroupedOutputLoRALinear)
    assert isinstance(layer.self_attn.o_proj, lora_pytorch.GroupedInputLoRALinear)
    assert isinstance(layer.mlp.gate_proj, lora_pytorch.LoRALinear)
    assert torch.count_nonzero(layer.self_attn.q_proj.lora_b) > 0
    assert torch.count_nonzero(layer.mlp.gate_proj.lora_b) > 0
    assert "layers.0.self_attn.q_proj.weight" in gemma_model.state_dict()

    paligemma = torch_nn.Module()
    paligemma.model = torch_nn.Module()
    paligemma.model.vision_tower = torch_nn.Module()
    paligemma.model.vision_tower.vision_model = torch_nn.Module()
    paligemma.model.vision_tower.vision_model.encoder = torch_nn.Module()
    paligemma.model.vision_tower.vision_model.encoder.layers = torch_nn.ModuleList([_FakeVisionLayer()])
    lora_pytorch.inject_siglip_lora(paligemma, jax_lora.LoRAConfig(rank=2, alpha=2))
    vision_layer = paligemma.model.vision_tower.vision_model.encoder.layers[0]
    assert isinstance(vision_layer.self_attn.q_proj, lora_pytorch.LoRALinear)
    assert isinstance(vision_layer.mlp.fc2, lora_pytorch.LoRALinear)
    assert torch.count_nonzero(vision_layer.self_attn.q_proj.lora_b) == 0


def test_trainability_matches_jax_partition():
    config = gemma.Config(
        width=8,
        depth=1,
        mlp_dim=12,
        num_heads=2,
        num_kv_heads=1,
        head_dim=4,
        lora_configs={
            "attn": jax_lora.LoRAConfig(rank=2, alpha=2),
            "ffn": jax_lora.LoRAConfig(rank=2, alpha=2),
        },
    )
    pi0 = torch_nn.Module()
    pi0.action_head = torch_nn.Linear(8, 4)
    pi0.paligemma_with_expert = torch_nn.Module()
    nested = pi0.paligemma_with_expert
    nested.paligemma = torch_nn.Module()
    nested.paligemma.language_model = _FakeGemma(config)
    lora_pytorch.inject_gemma_lora(nested.paligemma.language_model, config)
    nested.gemma_expert = torch_nn.Module()
    nested.gemma_expert.model = _FakeGemma(config)
    lora_pytorch.inject_gemma_lora(nested.gemma_expert.model, config)
    nested.paligemma.model = torch_nn.Module()
    nested.paligemma.model.vision_tower = torch_nn.Module()
    nested.paligemma.model.vision_tower.vision_model = torch_nn.Module()
    nested.paligemma.model.vision_tower.vision_model.encoder = torch_nn.Module()
    nested.paligemma.model.vision_tower.vision_model.encoder.layers = torch_nn.ModuleList([_FakeVisionLayer()])
    nested.paligemma.model.multi_modal_projector = torch_nn.Linear(8, 8)
    lora_pytorch.inject_siglip_lora(nested.paligemma, jax_lora.LoRAConfig(rank=2, alpha=2))

    model_config = type(
        "Config",
        (),
        {
            "paligemma_variant": "gemma_2b_lora",
            "action_expert_variant": "gemma_300m_lora",
            "vision_train_mode": "lora",
        },
    )()
    lora_pytorch.configure_pi0_trainability(pi0, model_config)

    trainable = lora_pytorch.trainable_parameter_names(pi0)
    assert "action_head.weight" in trainable
    assert all("lora_" in name for name in trainable if "language_model" in name or "gemma_expert" in name)
    assert all("lora_" in name for name in trainable if "vision_tower" in name)
    assert not any("multi_modal_projector" in name for name in trainable)


def test_real_pi05_torch_parameter_partition_matches_jax_for_both_vision_modes():
    config = training_config.get_config("pi05_robotwin_place_dual_shoes_lora")
    for vision_train_mode in ("lora", "full"):
        model_config = dataclasses.replace(config.model, vision_train_mode=vision_train_mode)
        jax_state = nnx.state(nnx.eval_shape(model_config.create, jax.random.key(0)))
        jax_trainable_filter = nnx.All(nnx.Param, nnx.Not(model_config.get_freeze_filter()))
        jax_trainable = jax_state.filter(jax_trainable_filter)
        jax_total_count = sum(variable.value.size for variable in jax_state.flat_state().values())
        jax_trainable_count = sum(variable.value.size for variable in jax_trainable.flat_state().values())

        with torch.device("meta"):
            torch_model = PI0Pytorch(model_config)
        torch_total_count = sum(parameter.numel() for parameter in torch_model.parameters())
        torch_trainable_count = sum(
            parameter.numel() for parameter in torch_model.parameters() if parameter.requires_grad
        )

        assert torch_total_count == jax_total_count, vision_train_mode
        assert torch_trainable_count == jax_trainable_count, vision_train_mode

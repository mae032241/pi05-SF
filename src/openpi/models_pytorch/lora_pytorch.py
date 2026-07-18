"""PyTorch LoRA layers that are mathematically compatible with OpenPI JAX LoRA.

The JAX Gemma implementation applies LoRA directly to multi-dimensional einsum
weights.  In particular, every attention head owns an independent pair of
low-rank matrices.  A conventional PEFT ``Linear`` adapter would therefore not
be equivalent.  The grouped layers below preserve those einsum semantics while
keeping the original PyTorch ``weight`` and ``bias`` state-dict names.
"""

from __future__ import annotations

from collections.abc import Iterable
import logging
import math
from typing import Protocol

import torch
from torch import nn
import torch.nn.functional as F  # noqa: N812


class LoRAConfigLike(Protocol):
    rank: int
    alpha: float


def _validate_lora(rank: int, alpha: float) -> None:
    if rank <= 0:
        raise ValueError(f"LoRA rank must be positive, got {rank}")
    if not math.isfinite(alpha):
        raise ValueError(f"LoRA alpha must be finite, got {alpha}")


def _copy_base_linear(target: nn.Linear, source: nn.Linear) -> None:
    """Move an existing Linear's parameters without changing state-dict names."""
    target.weight = source.weight
    target.bias = source.bias
    target.train(source.training)


def _init_lora_pair(lora_a: torch.Tensor, lora_b: torch.Tensor, *, zero_b: bool) -> None:
    # Matches openpi.models.lora.LoRAConfig.init_fn.
    nn.init.normal_(lora_a, std=0.01)
    if zero_b:
        nn.init.zeros_(lora_b)
    else:
        nn.init.normal_(lora_b, std=0.01)


class LoRALinear(nn.Linear):
    """A 2-D LoRA Linear with JAX-compatible parameter orientation.

    ``lora_a`` is ``[in_features, rank]`` and ``lora_b`` is
    ``[rank, out_features]``.  The base parameters remain named ``weight`` and
    ``bias``, so a non-LoRA safetensors checkpoint can be loaded with
    ``strict=False`` without remapping its keys.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        rank: int,
        alpha: float,
        bias: bool,
        zero_b: bool,
        apply_scaling: bool = True,
        device=None,
        dtype=None,
    ):
        _validate_lora(rank, alpha)
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank if apply_scaling else 1.0
        self.lora_a = nn.Parameter(torch.empty(in_features, rank, device=device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.empty(rank, out_features, device=device, dtype=torch.float32))
        _init_lora_pair(self.lora_a, self.lora_b, zero_b=zero_b)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        rank: int,
        alpha: float,
        zero_b: bool,
        apply_scaling: bool = True,
    ) -> LoRALinear:
        result = cls(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            bias=linear.bias is not None,
            zero_b=zero_b,
            apply_scaling=apply_scaling,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        _copy_base_linear(result, linear)
        return result

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.linear(inputs, self.weight, self.bias)
        lora_a = self.lora_a.to(dtype=inputs.dtype)
        lora_b = self.lora_b.to(dtype=inputs.dtype)
        delta = torch.matmul(torch.matmul(inputs, lora_a), lora_b)
        return base + delta.to(dtype=base.dtype) * self.scaling


class GroupedOutputLoRALinear(nn.Linear):
    """LoRA for ``...D,NDH->...NH`` Gemma attention projections."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        groups: int,
        group_out_features: int,
        rank: int,
        alpha: float,
        bias: bool,
        device=None,
        dtype=None,
    ):
        _validate_lora(rank, alpha)
        if out_features != groups * group_out_features:
            raise ValueError(
                f"out_features={out_features} does not equal groups*group_out_features={groups * group_out_features}"
            )
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.groups = groups
        self.group_out_features = group_out_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_a = nn.Parameter(torch.empty(groups, in_features, rank, device=device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.empty(groups, rank, group_out_features, device=device, dtype=torch.float32))
        # OpenPI Gemma initializes both factors from N(0, 0.01).
        _init_lora_pair(self.lora_a, self.lora_b, zero_b=False)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        groups: int,
        group_out_features: int,
        rank: int,
        alpha: float,
    ) -> GroupedOutputLoRALinear:
        result = cls(
            linear.in_features,
            linear.out_features,
            groups=groups,
            group_out_features=group_out_features,
            rank=rank,
            alpha=alpha,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        _copy_base_linear(result, linear)
        return result

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.linear(inputs, self.weight, self.bias)
        lora_a = self.lora_a.to(dtype=inputs.dtype)
        lora_b = self.lora_b.to(dtype=inputs.dtype)
        low_rank = torch.einsum("...d,gdr->...gr", inputs, lora_a)
        delta = torch.einsum("...gr,grh->...gh", low_rank, lora_b).flatten(start_dim=-2)
        return base + delta.to(dtype=base.dtype) * self.scaling


class GroupedInputLoRALinear(nn.Linear):
    """LoRA for ``...NH,NHD->...D`` Gemma attention output projections.

    OpenPI's JAX equation rewriting removes the contracted head label from the
    intermediate equation.  Consequently it computes ``...NH,NHL->...L`` and
    then ``...L,NLD->...D``.  This is intentionally reproduced here instead of
    using the more usual per-head ``...NL`` intermediate.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        groups: int,
        group_in_features: int,
        rank: int,
        alpha: float,
        bias: bool,
        device=None,
        dtype=None,
    ):
        _validate_lora(rank, alpha)
        if in_features != groups * group_in_features:
            raise ValueError(
                f"in_features={in_features} does not equal groups*group_in_features={groups * group_in_features}"
            )
        super().__init__(in_features, out_features, bias=bias, device=device, dtype=dtype)
        self.groups = groups
        self.group_in_features = group_in_features
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.lora_a = nn.Parameter(torch.empty(groups, group_in_features, rank, device=device, dtype=torch.float32))
        self.lora_b = nn.Parameter(torch.empty(groups, rank, out_features, device=device, dtype=torch.float32))
        _init_lora_pair(self.lora_a, self.lora_b, zero_b=False)

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        *,
        groups: int,
        group_in_features: int,
        rank: int,
        alpha: float,
    ) -> GroupedInputLoRALinear:
        result = cls(
            linear.in_features,
            linear.out_features,
            groups=groups,
            group_in_features=group_in_features,
            rank=rank,
            alpha=alpha,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )
        _copy_base_linear(result, linear)
        return result

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        base = F.linear(inputs, self.weight, self.bias)
        grouped_inputs = inputs.unflatten(-1, (self.groups, self.group_in_features))
        lora_a = self.lora_a.to(dtype=inputs.dtype)
        lora_b = self.lora_b.to(dtype=inputs.dtype)
        low_rank = torch.einsum("...gh,ghr->...r", grouped_inputs, lora_a)
        delta = torch.einsum("...r,gro->...o", low_rank, lora_b)
        return base + delta.to(dtype=base.dtype) * self.scaling


_LoRALinearType = LoRALinear | GroupedOutputLoRALinear | GroupedInputLoRALinear


def _merged_plain_linear(module: _LoRALinearType) -> nn.Linear:
    """Fold one LoRA module into its base weight without duplicating that weight."""
    with torch.no_grad():
        lora_a = module.lora_a.to(dtype=torch.float32)
        lora_b = module.lora_b.to(dtype=torch.float32)
        if isinstance(module, GroupedOutputLoRALinear):
            # [G,I,R] @ [G,R,H] -> nn.Linear weight [G*H,I].
            delta_weight = torch.einsum("gir,grh->ghi", lora_a, lora_b).flatten(0, 1)
        elif isinstance(module, GroupedInputLoRALinear):
            # OpenPI's output projection sums B over its group/head axis.
            # [G,H,R], [M,R,O] -> nn.Linear weight [O,G*H].
            delta_weight = torch.einsum("ghr,mro->ogh", lora_a, lora_b).flatten(1, 2)
        else:
            # Forward uses x @ A @ B; nn.Linear stores the transposed weight.
            delta_weight = torch.matmul(lora_a, lora_b).T
        if delta_weight.shape != module.weight.shape:
            raise ValueError(
                f"Cannot merge {type(module).__name__}: delta={tuple(delta_weight.shape)} "
                f"weight={tuple(module.weight.shape)}"
            )
        module.weight.add_(delta_weight.to(dtype=module.weight.dtype), alpha=float(module.scaling))

    # Construct on the meta device to avoid allocating another potentially
    # multi-hundred-MiB base matrix, then reuse the merged parameters.
    merged = nn.Linear(
        module.in_features,
        module.out_features,
        bias=module.bias is not None,
        device="meta",
        dtype=module.weight.dtype,
    )
    _copy_base_linear(merged, module)
    return merged


def merge_lora_modules_(model: nn.Module) -> int:
    """Recursively fold all OpenPI LoRA modules into ordinary ``nn.Linear`` layers."""

    def merge_children(parent: nn.Module) -> int:
        count = 0
        for name, child in list(parent.named_children()):
            if isinstance(child, (LoRALinear, GroupedOutputLoRALinear, GroupedInputLoRALinear)):
                setattr(parent, name, _merged_plain_linear(child))
                count += 1
            else:
                count += merge_children(child)
        return count

    merged_count = merge_children(model)
    leftover = [name for name, _ in model.named_parameters() if "lora_" in name]
    if leftover:
        raise ValueError(f"Unmerged PyTorch LoRA parameters remain: {leftover[:5]}")
    logging.info("Folded %d PyTorch LoRA modules into ordinary nn.Linear layers", merged_count)
    return merged_count


def inject_gemma_lora(model: nn.Module, config) -> None:
    """Inject the exact JAX Gemma attention and FFN LoRA factorization."""
    attn_config: LoRAConfigLike | None = config.lora_configs.get("attn")
    ffn_config: LoRAConfigLike | None = config.lora_configs.get("ffn")
    if attn_config is None and ffn_config is None:
        return

    for layer in model.layers:
        if attn_config is not None:
            attn = layer.self_attn
            attn.q_proj = GroupedOutputLoRALinear.from_linear(
                attn.q_proj,
                groups=config.num_heads,
                group_out_features=config.head_dim,
                rank=attn_config.rank,
                alpha=attn_config.alpha,
            )
            for name in ("k_proj", "v_proj"):
                setattr(
                    attn,
                    name,
                    GroupedOutputLoRALinear.from_linear(
                        getattr(attn, name),
                        groups=config.num_kv_heads,
                        group_out_features=config.head_dim,
                        rank=attn_config.rank,
                        alpha=attn_config.alpha,
                    ),
                )
            attn.o_proj = GroupedInputLoRALinear.from_linear(
                attn.o_proj,
                groups=config.num_heads,
                group_in_features=config.head_dim,
                rank=attn_config.rank,
                alpha=attn_config.alpha,
            )

        if ffn_config is not None:
            mlp = layer.mlp
            # JAX openpi.models.lora.FeedForward intentionally does not apply
            # alpha/rank, so apply_scaling=False is required for parity.
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(
                    mlp,
                    name,
                    LoRALinear.from_linear(
                        getattr(mlp, name),
                        rank=ffn_config.rank,
                        alpha=ffn_config.alpha,
                        zero_b=False,
                        apply_scaling=False,
                    ),
                )


def inject_siglip_lora(paligemma: nn.Module, config: LoRAConfigLike) -> None:
    """Inject Vision LoRA matching openpi.models.siglip."""
    vision_model = paligemma.model.vision_tower.vision_model
    for layer in vision_model.encoder.layers:
        attention = layer.self_attn
        for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
            setattr(
                attention,
                name,
                LoRALinear.from_linear(
                    getattr(attention, name),
                    rank=config.rank,
                    alpha=config.alpha,
                    zero_b=True,
                ),
            )
        for name in ("fc1", "fc2"):
            setattr(
                layer.mlp,
                name,
                LoRALinear.from_linear(
                    getattr(layer.mlp, name),
                    rank=config.rank,
                    alpha=config.alpha,
                    zero_b=True,
                ),
            )


def _freeze_non_lora(module: nn.Module) -> None:
    for name, parameter in module.named_parameters():
        parameter.requires_grad_("lora_" in name)


def configure_pi0_trainability(model: nn.Module, config) -> None:
    """Apply the same trainable/frozen partition as Pi0Config.get_freeze_filter."""
    nested = model.paligemma_with_expert
    if "lora" in config.paligemma_variant:
        _freeze_non_lora(nested.paligemma.language_model)
    if "lora" in config.action_expert_variant:
        _freeze_non_lora(nested.gemma_expert.model)
    if config.vision_train_mode == "lora":
        _freeze_non_lora(nested.paligemma.model.vision_tower)
        # JAX's PaliGemma/img filter also freezes the multimodal image head.
        _freeze_non_lora(nested.paligemma.model.multi_modal_projector)


def trainable_parameters(model: nn.Module) -> Iterable[nn.Parameter]:
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def trainable_parameter_names(model: nn.Module) -> set[str]:
    return {name for name, parameter in model.named_parameters() if parameter.requires_grad}


def adapter_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    names = trainable_parameter_names(model)
    return {name: value.detach() for name, value in model.state_dict().items() if name in names}

import flax
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import lora
from openpi.models import lora_merge
from openpi.models import pi0_config
from openpi.models import siglip


def test_vision_lora_preserves_base_parameter_paths_and_zero_delta():
    image = jnp.ones((1, 32, 32, 3), dtype=jnp.float32)
    base = siglip.Module(num_classes=64, variant="mu/16", pool_type="none", scan=True)
    adapted = siglip.Module(
        num_classes=64,
        variant="mu/16",
        pool_type="none",
        scan=True,
        lora_config=lora.LoRAConfig(rank=2, alpha=2.0),
    )

    base_variables = base.init(jax.random.key(0), image)
    adapted_variables = adapted.init(jax.random.key(1), image)
    base_flat = flax.traverse_util.flatten_dict(base_variables["params"], sep="/")
    adapted_flat = flax.traverse_util.flatten_dict(adapted_variables["params"], sep="/")

    assert set(base_flat) <= set(adapted_flat)
    assert all(base_flat[key].shape == adapted_flat[key].shape for key in base_flat)
    assert any("lora" in key for key in set(adapted_flat) - set(base_flat))

    merged_flat = dict(adapted_flat)
    merged_flat.update(base_flat)
    merged_variables = {"params": flax.traverse_util.unflatten_dict(merged_flat, sep="/")}
    base_output, _ = base.apply(base_variables, image)
    adapted_output, _ = adapted.apply(merged_variables, image)
    np.testing.assert_allclose(base_output, adapted_output, rtol=0, atol=0)

    def loss(params):
        _, outputs = adapted.apply({"params": params}, image)
        return outputs["encoded"][0, 0, 0]

    grads = jax.grad(loss)(merged_variables["params"])
    grad_flat = flax.traverse_util.flatten_dict(grads, sep="/")
    assert any(jnp.any(value != 0) for key, value in grad_flat.items() if key.endswith("lora_b"))


def test_full_vision_module_matches_after_inference_merge():
    image = jax.random.normal(jax.random.key(2), (1, 32, 32, 3))
    base = siglip.Module(num_classes=64, variant="mu/16", pool_type="none", scan=True)
    adapted = siglip.Module(
        num_classes=64,
        variant="mu/16",
        pool_type="none",
        scan=True,
        lora_config=lora.LoRAConfig(rank=2, alpha=2.0),
    )
    adapted_variables = adapted.init(jax.random.key(3), image)
    adapted_flat = flax.traverse_util.flatten_dict(adapted_variables["params"], sep="/")
    rng = np.random.default_rng(4)
    for key, value in adapted_flat.items():
        if key.endswith(("lora_a", "lora_b")):
            adapted_flat[key] = rng.normal(scale=0.01, size=value.shape).astype(np.float32)
    adapted_params = flax.traverse_util.unflatten_dict(adapted_flat, sep="/")

    model_config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        vision_train_mode="lora",
        vision_variant="mu/16",
        vision_lora_rank=2,
        vision_lora_alpha=2.0,
    )
    merged, _ = lora_merge.merge_lora_params({"PaliGemma": {"img": adapted_params}}, model_config)
    merged_img_params = merged["PaliGemma"]["img"]

    adapted_output, _ = adapted.apply({"params": adapted_params}, image)
    merged_output, _ = base.apply({"params": merged_img_params}, image)
    np.testing.assert_allclose(merged_output, adapted_output, rtol=2e-5, atol=2e-5)

import flax.traverse_util
import numpy as np

from openpi.models import lora_merge
from openpi.models import pi0_config


def _random(shape, seed):
    return np.random.default_rng(seed).normal(scale=0.1, size=shape).astype(np.float32)


def test_standard_scanned_merge_matches_separate_lora():
    weight = _random((3, 5, 7), 0)
    lora_a = _random((3, 5, 2), 1)
    lora_b = _random((3, 2, 7), 2)
    merged = lora_merge._merge_pair(weight.copy(), lora_a, lora_b, scale=0.75, rule="standard")  # noqa: SLF001

    expected = weight + 0.75 * np.einsum("gir,gro->gio", lora_a, lora_b)
    np.testing.assert_allclose(merged, expected, rtol=1e-6, atol=1e-6)


def test_vision_projection_merges_match_separate_lora():
    inputs = _random((2, 4, 6), 3)

    qkv_weight = _random((6, 3, 5), 4)
    qkv_a = _random((6, 2), 5)
    qkv_b = _random((2, 3, 5), 6)
    merged_qkv = lora_merge._merge_pair(  # noqa: SLF001
        qkv_weight.copy(), qkv_a, qkv_b, scale=1.25, rule="vision_qkv"
    )
    separate_qkv = np.einsum("bti,ihd->bthd", inputs, qkv_weight)
    separate_qkv += 1.25 * np.einsum("btr,rhd->bthd", np.einsum("bti,ir->btr", inputs, qkv_a), qkv_b)
    folded_qkv = np.einsum("bti,ihd->bthd", inputs, merged_qkv)
    np.testing.assert_allclose(folded_qkv, separate_qkv, rtol=1e-5, atol=1e-6)

    out_inputs = _random((2, 4, 3, 5), 7)
    out_weight = _random((3, 5, 6), 8)
    out_a = _random((3, 5, 2), 9)
    out_b = _random((2, 6), 10)
    merged_out = lora_merge._merge_pair(  # noqa: SLF001
        out_weight.copy(), out_a, out_b, scale=0.5, rule="vision_out"
    )
    separate_out = np.einsum("bthd,hdo->bto", out_inputs, out_weight)
    separate_out += 0.5 * np.einsum("btr,ro->bto", np.einsum("bthd,hdr->btr", out_inputs, out_a), out_b)
    folded_out = np.einsum("bthd,hdo->bto", out_inputs, merged_out)
    np.testing.assert_allclose(folded_out, separate_out, rtol=1e-5, atol=1e-6)


def test_gemma_output_merge_preserves_openpi_cross_head_semantics():
    inputs = _random((2, 4, 3, 5), 11)
    weight = _random((3, 5, 7), 12)
    lora_a = _random((3, 5, 2), 13)
    lora_b = _random((3, 2, 7), 14)
    merged = lora_merge._merge_pair(weight.copy(), lora_a, lora_b, scale=0.8, rule="gemma_out")  # noqa: SLF001

    separate = np.einsum("btnh,nhd->btd", inputs, weight)
    low_rank = np.einsum("btnh,nhl->btl", inputs, lora_a)
    separate += 0.8 * np.einsum("btl,nld->btd", low_rank, lora_b)
    folded = np.einsum("btnh,nhd->btd", inputs, merged)
    np.testing.assert_allclose(folded, separate, rtol=1e-5, atol=1e-6)


def test_merge_lora_params_removes_adapters_and_preserves_other_trainables():
    config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        vision_train_mode="lora",
        vision_lora_rank=2,
        vision_lora_alpha=4.0,
    )
    vision_weight = _random((4, 6), 15)
    vision_a = _random((4, 2), 16)
    vision_b = _random((2, 6), 17)
    gemma_weight = _random((3, 5), 18)
    gemma_a = _random((3, 16), 19)
    gemma_b = _random((16, 5), 20)
    ordinary_trainable = _random((2, 3), 21)
    params = {
        "PaliGemma": {
            "img": {
                "Transformer": {
                    "Dense_0": {
                        "kernel": vision_weight.copy(),
                        "lora_a": vision_a,
                        "lora_b": vision_b,
                    }
                }
            },
            "llm": {
                "layers": {
                    "attn": {
                        "q_einsum": {
                            "w": gemma_weight.copy(),
                            "lora_a": gemma_a,
                            "lora_b": gemma_b,
                        }
                    }
                }
            },
        },
        "action_in_proj": {"kernel": ordinary_trainable},
    }

    merged, inference_config = lora_merge.merge_lora_params(params, config)
    flat = flax.traverse_util.flatten_dict(merged, sep="/")
    assert not any("lora" in key for key in flat)
    np.testing.assert_allclose(
        flat["PaliGemma/img/Transformer/Dense_0/kernel"],
        vision_weight + 2.0 * vision_a @ vision_b,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        flat["PaliGemma/llm/layers/attn/q_einsum/w"],
        gemma_weight + gemma_a @ gemma_b,
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(flat["action_in_proj/kernel"], ordinary_trainable)
    assert inference_config.paligemma_variant == "gemma_2b"
    assert inference_config.action_expert_variant == "gemma_300m"
    assert inference_config.vision_train_mode == "full"

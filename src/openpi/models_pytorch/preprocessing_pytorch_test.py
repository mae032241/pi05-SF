from types import SimpleNamespace

import torch

from openpi.models_pytorch import preprocessing_pytorch


def _observation(images):
    batch = next(iter(images.values())).shape[0]
    height, width = 8, 10
    return SimpleNamespace(
        images=images,
        image_padding_mask={name: torch.ones(batch, height, width, dtype=torch.bool) for name in images},
        image_masks={name: torch.ones(batch, dtype=torch.bool) for name in images},
        state=torch.zeros(batch, 32),
        tokenized_prompt=torch.ones(batch, 4, dtype=torch.int32),
        tokenized_prompt_mask=torch.ones(batch, 4, dtype=torch.bool),
        token_ar_mask=None,
        token_loss_mask=None,
    )


def test_preprocessing_always_returns_nchw_for_torch_vision():
    image_keys = preprocessing_pytorch.IMAGE_KEYS
    hwc = torch.arange(2 * 8 * 10 * 3, dtype=torch.float32).reshape(2, 8, 10, 3)
    chw = hwc.permute(0, 3, 1, 2).contiguous()

    from_hwc = preprocessing_pytorch.preprocess_observation_pytorch(
        _observation({name: hwc.clone() for name in image_keys}),
        image_resolution=(8, 10),
    )
    from_chw = preprocessing_pytorch.preprocess_observation_pytorch(
        _observation({name: chw.clone() for name in image_keys}),
        image_resolution=(8, 10),
    )

    for name in image_keys:
        assert from_hwc.images[name].shape == (2, 3, 8, 10)
        torch.testing.assert_close(from_hwc.images[name], from_chw.images[name])

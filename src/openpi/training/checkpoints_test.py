import json

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from openpi.models import model as model_lib
from openpi.models import pi0_config
from openpi.policies import policy_config
from openpi.shared import nnx_utils
from openpi.training import checkpoints
from openpi.training import config
from openpi.training import utils
from openpi.training import weight_loaders


class _DataLoader:
    def data_config(self):
        return config.DataConfig(repo_id="fake")


def _save_params(path, params):
    with ocp.PyTreeCheckpointer() as checkpointer:
        checkpointer.save(path, {"params": params})


def _make_config(base_params_path):
    model = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        vision_train_mode="lora",
        vision_variant="mu/16",
        vision_lora_rank=2,
        vision_lora_alpha=2.0,
    )
    freeze_filter = nnx.All(nnx.Param, nnx.Not(nnx_utils.PathRegex(".*lora.*")))
    return config.TrainConfig(
        name="adapter_test",
        exp_name="run",
        model=model,
        data=config.FakeDataConfig(),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(base_params_path)),
        freeze_filter=freeze_filter,
        lora_save=True,
        ema_decay=None,
        wandb_enabled=False,
    )


def _make_state(train_config, *, perturb_adapters):
    model = train_config.model.create(jax.random.key(1))
    model_def, params = nnx.split(model)
    params.replace_by_pure_dict(train_config.weight_loader.load(params.to_pure_dict()))
    trainable_paths = set(params.filter(train_config.trainable_filter).flat_state())
    if perturb_adapters:
        params = params.map(lambda path, leaf: leaf.replace(leaf.value + 0.25) if path in trainable_paths else leaf)
    tx = optax.adam(1e-3)
    return utils.TrainState(
        step=jnp.asarray(7),
        params=params,
        model_def=model_def,
        tx=tx,
        opt_state=tx.init(params.filter(train_config.trainable_filter)),
        ema_decay=None,
        ema_params=None,
    )


def _flat_values(state):
    values = {}

    def collect(path, leaf):
        values[path] = np.asarray(leaf.value)
        return leaf

    state.map(collect)
    return values


def test_adapter_checkpoint_save_restore_and_inference_merge(tmp_path):
    base_model_config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        vision_train_mode="full",
        vision_variant="mu/16",
    )
    base_model = base_model_config.create(jax.random.key(0))
    base_params_path = tmp_path / "base_params"
    _save_params(base_params_path, nnx.state(base_model).to_pure_dict())

    train_config = _make_config(base_params_path)
    saved_state = _make_state(train_config, perturb_adapters=True)
    manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "adapter_run", keep_period=None, overwrite=False, resume=False
    )
    metadata = {
        "format": "openpi_trainable_adapter_v1",
        "base_params_path": str(base_params_path),
        "train_config_name": train_config.name,
        "vision_train_mode": "lora",
    }
    checkpoints.save_state(
        manager,
        saved_state,
        _DataLoader(),
        7,
        trainable_filter=train_config.trainable_filter,
        adapter_metadata=metadata,
    )
    manager.wait_until_finished()

    checkpoint_dir = tmp_path / "adapter_run" / "7"
    assert json.loads((checkpoint_dir / "assets" / "adapter_metadata.json").read_text()) == metadata
    saved_adapter = model_lib.restore_params(checkpoint_dir / "params", restore_type=np.ndarray)
    assert saved_adapter
    assert all("lora" in key for key in _flatten_keys(saved_adapter))

    restored_state = checkpoints.restore_state(
        manager,
        _make_state(train_config, perturb_adapters=False),
        _DataLoader(),
        trainable_filter=train_config.trainable_filter,
    )
    saved_values = _flat_values(saved_state.params.filter(train_config.trainable_filter))
    restored_values = _flat_values(restored_state.params.filter(train_config.trainable_filter))
    assert saved_values.keys() == restored_values.keys()
    for key in saved_values:
        np.testing.assert_array_equal(saved_values[key], restored_values[key])

    inference_model = policy_config._load_adapter_model(train_config, checkpoint_dir)  # noqa: SLF001
    inference_values = _flat_values(nnx.state(inference_model).filter(train_config.trainable_filter))
    assert saved_values.keys() == inference_values.keys()
    for key in saved_values:
        np.testing.assert_array_equal(saved_values[key], inference_values[key])

    fused_model = policy_config._load_adapter_model(  # noqa: SLF001
        train_config,
        checkpoint_dir,
        merge_lora_for_inference=True,
    )
    fused_values = _flat_values(nnx.state(fused_model))
    assert not any("lora" in "/".join(map(str, key)) for key in fused_values)
    manager.close()


def test_default_checkpoint_still_saves_full_params(tmp_path):
    base_model_config = pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="dummy",
        action_expert_variant="dummy",
        vision_train_mode="full",
        vision_variant="mu/16",
    )
    base_model = base_model_config.create(jax.random.key(0))
    base_params_path = tmp_path / "base_params"
    _save_params(base_params_path, nnx.state(base_model).to_pure_dict())
    train_config = _make_config(base_params_path)
    state = _make_state(train_config, perturb_adapters=True)
    manager, _ = checkpoints.initialize_checkpoint_dir(
        tmp_path / "full_run", keep_period=None, overwrite=False, resume=False
    )
    checkpoints.save_state(manager, state, _DataLoader(), 7)
    manager.wait_until_finished()
    checkpoint_dir = tmp_path / "full_run" / "7"
    assert not (checkpoint_dir / "assets" / "adapter_metadata.json").exists()
    full_params = model_lib.restore_params(checkpoint_dir / "params", restore_type=np.ndarray)
    assert _flatten_keys(full_params) == _flatten_keys(state.params.to_pure_dict())
    manager.close()


def _flatten_keys(tree):
    leaves, _ = jax.tree.flatten_with_path(tree)
    return {jax.tree_util.keystr(path) for path, _ in leaves}

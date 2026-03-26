"""Debug Policy with detailed logging for state inputs and action outputs."""
from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy

# Configure numpy printing
np.set_printoptions(linewidth=200, suppress=True, precision=4)


class DebugPolicy(BasePolicy):
    """Policy with detailed debug logging for inputs and outputs."""

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
        debug_interval: int = 1,  # Print every N steps
        debug_enabled: bool = True,
    ):
        """Initialize the Debug Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
            debug_interval: Print debug info every N steps (default: 1 = every step).
            debug_enabled: Enable/disable debug printing (default: True).
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device
        self._debug_interval = debug_interval
        self._debug_enabled = debug_enabled
        self._step_count = 0

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    def _print_debug_info(self, stage: str, data: dict | np.ndarray, prefix: str = ""):
        """Print debug information in a formatted way."""
        if not self._debug_enabled or self._step_count % self._debug_interval != 0:
            return

        print(f"\n{'='*80}")
        print(f"[DEBUG] Step {self._step_count} - {stage}")
        print(f"{'='*80}")

        if isinstance(data, dict):
            for key, value in data.items():
                if isinstance(value, (np.ndarray, jnp.ndarray, torch.Tensor)):
                    if isinstance(value, torch.Tensor):
                        value = value.detach().cpu().numpy()
                    elif isinstance(value, jnp.ndarray):
                        value = np.asarray(value)

                    print(f"{prefix}{key}:")
                    print(f"  shape: {value.shape}")
                    print(f"  dtype: {value.dtype}")
                    print(f"  range: [{value.min():.4f}, {value.max():.4f}]")
                    print(f"  mean: {value.mean():.4f}, std: {value.std():.4f}")

                    # Print actual values for small arrays or first few elements
                    if value.size <= 20:
                        print(f"  values: {value.flatten()}")
                    elif len(value.shape) == 1:
                        print(f"  values (first 10): {value[:10]}")
                    elif len(value.shape) == 2:
                        print(f"  values (first row): {value[0]}")
                    else:
                        print(f"  values (first element): {value.flatten()[:10]}")
                elif isinstance(value, dict):
                    print(f"{prefix}{key}: (nested dict)")
                    self._print_debug_info(stage, value, prefix=f"  {prefix}")
                else:
                    print(f"{prefix}{key}: {value}")
        elif isinstance(data, (np.ndarray, jnp.ndarray, torch.Tensor)):
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            elif isinstance(data, jnp.ndarray):
                data = np.asarray(data)

            print(f"{prefix}Array:")
            print(f"  shape: {data.shape}")
            print(f"  dtype: {data.dtype}")
            print(f"  range: [{data.min():.4f}, {data.max():.4f}]")
            print(f"  mean: {data.mean():.4f}, std: {data.std():.4f}")
            print(f"  values: {data}")

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)

        # Debug: Print raw inputs
        if self._debug_enabled and self._step_count % self._debug_interval == 0:
            print(f"\n{'='*80}")
            print(f"[DEBUG] Step {self._step_count} - RAW INPUTS (before transform)")
            print(f"{'='*80}")
            if "state" in inputs:
                state = inputs["state"]
                if isinstance(state, (np.ndarray, jnp.ndarray, torch.Tensor)):
                    state_np = np.asarray(state)
                    print(f"state shape: {state_np.shape}")
                    print(f"state dtype: {state_np.dtype}")
                    print(f"state values: {state_np}")
                    print(f"state range: [{state_np.min():.4f}, {state_np.max():.4f}]")
                    print(f"state mean: {state_np.mean():.4f}, std: {state_np.std():.4f}")

        inputs = self._input_transform(inputs)

        # Debug: Print transformed inputs
        if self._debug_enabled and self._step_count % self._debug_interval == 0:
            print(f"\n{'='*80}")
            print(f"[DEBUG] Step {self._step_count} - TRANSFORMED INPUTS (after input_transform)")
            print(f"{'='*80}")
            if "state" in inputs:
                state = inputs["state"]
                if isinstance(state, (np.ndarray, jnp.ndarray, torch.Tensor)):
                    state_np = np.asarray(state)
                    print(f"state shape: {state_np.shape}")
                    print(f"state dtype: {state_np.dtype}")
                    print(f"state values: {state_np}")
                    print(f"state range: [{state_np.min():.4f}, {state_np.max():.4f}]")
                    print(f"state mean: {state_np.mean():.4f}, std: {state_np.std():.4f}")

        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Debug: Print batched inputs
        if self._debug_enabled and self._step_count % self._debug_interval == 0:
            print(f"\n{'='*80}")
            print(f"[DEBUG] Step {self._step_count} - BATCHED INPUTS (ready for model)")
            print(f"{'='*80}")
            if "state" in inputs:
                state = inputs["state"]
                if isinstance(state, (np.ndarray, jnp.ndarray, torch.Tensor)):
                    if isinstance(state, torch.Tensor):
                        state_np = state.detach().cpu().numpy()
                    else:
                        state_np = np.asarray(state)
                    print(f"state shape: {state_np.shape}")
                    print(f"state dtype: {state_np.dtype}")
                    print(f"state values: {state_np}")
                    print(f"state range: [{state_np.min():.4f}, {state_np.max():.4f}]")

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()

        # Call model
        actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        model_time = time.monotonic() - start_time

        # Debug: Print raw model outputs (before unbatching)
        if self._debug_enabled and self._step_count % self._debug_interval == 0:
            print(f"\n{'='*80}")
            print(f"[DEBUG] Step {self._step_count} - RAW MODEL OUTPUT (before unbatching)")
            print(f"{'='*80}")
            if isinstance(actions, (np.ndarray, jnp.ndarray, torch.Tensor)):
                if isinstance(actions, torch.Tensor):
                    actions_np = actions.detach().cpu().numpy()
                else:
                    actions_np = np.asarray(actions)
                print(f"actions shape: {actions_np.shape}")
                print(f"actions dtype: {actions_np.dtype}")
                print(f"actions range: [{actions_np.min():.4f}, {actions_np.max():.4f}]")
                print(f"actions mean: {actions_np.mean():.4f}, std: {actions_np.std():.4f}")

                # Print first action in batch
                if len(actions_np.shape) >= 2:
                    print(f"actions[0] (first in batch): {actions_np[0]}")

        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        # Debug: Print unbatched outputs (before output transform)
        if self._debug_enabled and self._step_count % self._debug_interval == 0:
            print(f"\n{'='*80}")
            print(f"[DEBUG] Step {self._step_count} - UNBATCHED OUTPUT (before output_transform)")
            print(f"{'='*80}")
            if "actions" in outputs:
                actions_out = outputs["actions"]
                print(f"actions shape: {actions_out.shape}")
                print(f"actions dtype: {actions_out.dtype}")
                print(f"actions values: {actions_out}")
                print(f"actions range: [{actions_out.min():.4f}, {actions_out.max():.4f}]")
                print(f"actions mean: {actions_out.mean():.4f}, std: {actions_out.std():.4f}")

        outputs = self._output_transform(outputs)

        # Debug: Print final outputs (after output transform)
        if self._debug_enabled and self._step_count % self._debug_interval == 0:
            print(f"\n{'='*80}")
            print(f"[DEBUG] Step {self._step_count} - FINAL OUTPUT (after output_transform)")
            print(f"{'='*80}")
            if "actions" in outputs:
                actions_final = outputs["actions"]
                print(f"actions shape: {actions_final.shape}")
                print(f"actions dtype: {actions_final.dtype}")
                print(f"actions values: {actions_final}")
                print(f"actions range: [{actions_final.min():.4f}, {actions_final.max():.4f}]")
                print(f"actions mean: {actions_final.mean():.4f}, std: {actions_final.std():.4f}")

                # Analyze left vs right arm if action dim is 14
                if actions_final.shape[-1] == 14:
                    print(f"\nLeft arm actions [0-6]:  {actions_final[..., :7]}")
                    print(f"Right arm actions [7-13]: {actions_final[..., 7:14]}")
                    print(f"Left gripper (idx 6):  {actions_final[..., 6]}")
                    print(f"Right gripper (idx 13): {actions_final[..., 13]}")

            print(f"\nInference time: {model_time * 1000:.2f} ms")
            print(f"{'='*80}\n")

        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }

        self._step_count += 1
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

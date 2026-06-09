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
from openpi.models import pi0 as _pi0
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
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
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transforms = tuple(output_transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(
                model.sample_actions,
                static_argnames=("rtc_prefix_attention_schedule",),
            )
            self._rng = rng or jax.random.key(0)

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
        rtc_context: dict[str, Any] | None = None,
        return_model_actions: bool = False,
    ) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        if rtc_context is not None:
            if self._is_pytorch_model:
                raise NotImplementedError("Inference-time RTC is currently only supported for JAX Pi0 policies.")
            if not isinstance(self._model, _pi0.Pi0):
                raise NotImplementedError("Inference-time RTC is currently only supported for JAX Pi0 policies.")

            prev_action_chunk = rtc_context.get("prev_action_chunk")
            if prev_action_chunk is None:
                raise ValueError("rtc_context must include prev_action_chunk")
            relative_action_mask = self._get_rtc_relative_action_mask(rtc_context)
            if relative_action_mask is not None:
                prev_action_chunk = _transforms.reanchor_relative_rtc_prefix(
                    np.asarray(prev_action_chunk), np.asarray(inputs["state"]), relative_action_mask
                )
            prev_action_chunk = jnp.asarray(prev_action_chunk)
            if prev_action_chunk.ndim == 2:
                prev_action_chunk = prev_action_chunk[None, ...]
            if prev_action_chunk.ndim != 3:
                raise ValueError(f"prev_action_chunk must be rank 2 or 3, got shape {prev_action_chunk.shape}")

            action_horizon = prev_action_chunk.shape[-2]
            inference_delay = int(rtc_context.get("inference_delay", 0))
            execution_horizon = int(
                rtc_context.get("execution_horizon", rtc_context.get("prefix_attention_horizon", action_horizon))
            )
            if inference_delay < 0:
                raise ValueError(f"inference_delay must be non-negative, got {inference_delay}")
            if not 0 <= execution_horizon <= action_horizon:
                raise ValueError(
                    f"execution_horizon must be in [0, {action_horizon}], got {execution_horizon}"
                )
            if inference_delay > execution_horizon:
                raise ValueError(
                    "inference_delay cannot exceed execution_horizon: " f"{inference_delay} > {execution_horizon}"
                )

            schedule = rtc_context.get("prefix_attention_schedule", "exp")
            sample_kwargs["rtc_prev_action_chunk"] = prev_action_chunk
            sample_kwargs["rtc_inference_delay"] = inference_delay
            sample_kwargs["rtc_execution_horizon"] = execution_horizon
            sample_kwargs["rtc_prefix_attention_horizon"] = execution_horizon
            sample_kwargs["rtc_prefix_attention_schedule"] = schedule
            sample_kwargs["rtc_max_guidance_weight"] = float(rtc_context.get("max_guidance_weight", 10.0))

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        model_actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        outputs = {
            "state": inputs["state"],
            "actions": model_actions,
        }
        model_time = time.monotonic() - start_time
        model_outputs = None
        if return_model_actions:
            model_outputs = jax.tree.map(
                lambda x: np.asarray(x[0, ...].detach().cpu()) if self._is_pytorch_model else np.asarray(x[0, ...]),
                {"actions": model_actions},
            )
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        if model_outputs is not None:
            outputs["_rtc_model_actions"] = model_outputs["actions"]
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata

    def _get_rtc_relative_action_mask(self, rtc_context: dict[str, Any]) -> Sequence[bool] | np.ndarray | None:
        """Resolve whether an absolute RTC prefix should be reanchored into delta action space."""
        if "relative_action_mask" in rtc_context:
            return rtc_context["relative_action_mask"]

        prefix_space = rtc_context.get("prev_action_chunk_space")
        should_reanchor = rtc_context.get("reanchor_relative_prefix", False) or prefix_space == "absolute"
        if not should_reanchor:
            return None

        for transform in _iter_transforms(self._output_transforms):
            if isinstance(transform, _transforms.AbsoluteActions) and transform.mask is not None:
                return transform.mask
        return None


def _iter_transforms(transforms: Sequence[_transforms.DataTransformFn]):
    for transform in transforms:
        if isinstance(transform, _transforms.CompositeTransform):
            yield from _iter_transforms(transform.transforms)
        else:
            yield transform


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict, **kwargs) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs, **kwargs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results

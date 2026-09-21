#!/usr/bin/env python3
"""Benchmark the two LIFT2 Pi0 checkpoints on a small real training-data subset.

This script intentionally benchmarks one checkpoint per process. That keeps the GPU
memory state and JAX compilation cache independent between the discrete and standard
models. It reports both policy-level timings and model-internal timings.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path
import statistics
import time
from typing import Any

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image
import pyarrow.parquet as pq

from openpi.models import model as _model
from openpi.models import pi0 as _pi0
from openpi.policies import policy_config
from openpi.training import config as _config

DISCRETE_CHECKPOINT = (
    "/soft/wangxi/discrete_forcing_openpi/checkpoints/pi0_lift2_df_image_discrete/lift2_df_image_discrete_full/99999"
)
STANDARD_CHECKPOINT = "/soft/wangxi/openpi-lift2/checkpoints/pi0_lift2_df_image/pi0_lift2_df_image_full/99999"
DATA_ROOT = Path("/soft/wangxi/.cache/huggingface/lerobot/lerobot_lift2_df_image")


@dataclasses.dataclass
class Sample:
    observation: dict[str, Any]
    source_file: str
    row: int
    episode_index: int
    frame_index: int


def now_ns() -> int:
    return time.perf_counter_ns()


def elapsed_ms(start_ns: int, end_ns: int | None = None) -> float:
    return (end_ns if end_ns is not None else now_ns() - start_ns) / 1e6


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values, dtype=np.float64), p))


def summary(values: list[float]) -> dict[str, float | int]:
    values = [float(x) for x in values]
    return {
        "n": len(values),
        "mean_ms": float(statistics.mean(values)) if values else float("nan"),
        "median_ms": float(statistics.median(values)) if values else float("nan"),
        "p90_ms": percentile(values, 90),
        "p95_ms": percentile(values, 95),
        "min_ms": float(min(values)) if values else float("nan"),
        "max_ms": float(max(values)) if values else float("nan"),
        "std_ms": float(statistics.stdev(values)) if len(values) > 1 else 0.0,
    }


def decode_image(encoded: dict[str, Any]) -> np.ndarray:
    # The current LeRobot cache stores PNG bytes in the parquet struct.
    image = Image.open(__import__("io").BytesIO(encoded["bytes"])).convert("RGB")
    return np.asarray(image, dtype=np.uint8)


def load_samples(num_samples: int, stride: int) -> tuple[list[Sample], dict[str, Any]]:
    if not DATA_ROOT.exists():
        raise FileNotFoundError(f"Training dataset cache not found: {DATA_ROOT}")

    parquet_files = sorted((DATA_ROOT / "data" / "chunk-000").glob("episode_*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files under {DATA_ROOT / 'data'}")

    samples: list[Sample] = []
    decode_times: list[float] = []
    rows_scanned = 0
    for parquet_file in parquet_files:
        table = pq.read_table(parquet_file)
        row_count = table.num_rows
        rows_scanned += row_count
        # Pick separated rows from different episodes/files where possible. The dataset
        # has one episode per parquet file, so this gives a diverse small subset.
        for row in range(0, row_count, max(1, stride)):
            t0 = now_ns()
            head = decode_image(table["observation.images.head"][row].as_py())
            left = decode_image(table["observation.images.left_wrist"][row].as_py())
            right = decode_image(table["observation.images.right_wrist"][row].as_py())
            decode_times.append(elapsed_ms(t0))
            state = np.asarray(table["observation.state"][row].as_py(), dtype=np.float32)
            samples.append(
                Sample(
                    observation={
                        "observation.images.head": head,
                        "observation.images.left_wrist": left,
                        "observation.images.right_wrist": right,
                        "observation.state": state,
                        "prompt": "perform task",
                    },
                    source_file=str(parquet_file),
                    row=row,
                    episode_index=int(table["episode_index"][row].as_py()),
                    frame_index=int(table["frame_index"][row].as_py()),
                )
            )
            if len(samples) >= num_samples:
                return samples, {
                    "dataset_root": str(DATA_ROOT),
                    "source_files": len({s.source_file for s in samples}),
                    "rows_scanned": rows_scanned,
                    "decoded_samples": len(samples),
                    "png_decode_ms": summary(decode_times),
                    "image_shape": list(head.shape),
                    "state_shape": list(state.shape),
                }
    raise ValueError(f"Only found {len(samples)} samples, requested {num_samples}")


def make_policy(model_name: str):
    if model_name == "discrete":
        config_name = "pi0_lift2_df_image_discrete"
        checkpoint = DISCRETE_CHECKPOINT
    else:
        config_name = "pi0_lift2_df_image"
        checkpoint = STANDARD_CHECKPOINT

    cfg = _config.get_config(config_name, require_discrete_stats=False)
    # Use an explicit zero noise in the benchmark so both models see the same input
    # observation and the benchmark does not include RNG generation variability.
    policy = policy_config.create_trained_policy(
        cfg,
        checkpoint,
        default_prompt="perform task",
        sample_kwargs={"num_steps": 10} if model_name == "standard" else None,
    )
    return cfg, policy


def prepare_policy_input(policy, raw_observation: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    # This mirrors Policy.infer but exposes the transform and host-to-device portions.
    # Do not mutate the raw sample because it is reused for every model call.
    raw = jax.tree.map(lambda x: x.copy() if isinstance(x, np.ndarray) else x, raw_observation)
    t0 = now_ns()
    transformed = policy._input_transform(raw)  # noqa: SLF001 - benchmark intentionally profiles internals.
    transform_ms = elapsed_ms(t0)

    t0 = now_ns()
    batched = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], transformed)
    device_input_ms = elapsed_ms(t0)
    return batched, {"input_transform_ms": transform_ms, "host_to_device_ms": device_input_ms}


def make_noise(model_name: str) -> np.ndarray:
    # A fixed noise array removes random generation from the measured model path.
    # The standard and discrete models both consume the same Pi0 padded shape.
    rng = np.random.default_rng(20260921)
    return rng.standard_normal((30, 32), dtype=np.float32)


def sync_tree(tree: Any) -> None:
    leaves = jax.tree.leaves(tree)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def time_jax_call(fn, *args, **kwargs) -> tuple[Any, float]:
    t0 = now_ns()
    result = fn(*args, **kwargs)
    sync_tree(result)
    return result, elapsed_ms(t0)


def model_internal_benchmarks(
    policy, model_name: str, batched_input: dict[str, Any], noise: np.ndarray
) -> dict[str, Any]:
    """Measure internal model pieces with separate JIT functions.

    The wrappers reproduce the corresponding code in Pi0.sample_actions and
    Pi0Discrete.sample_actions. They are only for timing; policy.infer remains the
    end-to-end correctness path.
    """
    model = policy._model  # noqa: SLF001
    observation = _model.Observation.from_dict(batched_input)
    noise_jax = jnp.asarray(noise)[None, ...]
    batch = 1
    ones = jnp.ones(batch)
    zeros = jnp.zeros(batch)
    graphdef, state = nnx.split(model)

    if model_name == "standard":

        def prefix_fun(state, observation):
            module = nnx.merge(graphdef, state)
            observation = _model.preprocess_observation(None, observation, train=False)
            prefix_tokens, prefix_mask, prefix_ar_mask = module.embed_prefix(observation)
            prefix_attn_mask = _pi0.make_attn_mask(prefix_mask, prefix_ar_mask)
            positions = jnp.cumsum(prefix_mask, axis=1) - 1
            _, kv_cache = module.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
            return observation, prefix_tokens, prefix_mask, kv_cache

        def step_fun(state, observation, prefix_tokens, prefix_mask, kv_cache, x_t, time):
            module = nnx.merge(graphdef, state)
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = module.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch)
            )
            suffix_attn_mask = _pi0.make_attn_mask(suffix_mask, suffix_ar_mask)
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
            (prefix_out, suffix_out), _ = module.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            del prefix_out
            velocity = module.action_out_proj(suffix_out[:, -module.action_horizon :])
            return x_t - velocity / 10.0

        prefix_jit = jax.jit(prefix_fun)
        step_jit = jax.jit(step_fun)

        # Compile and warm up the separated wrappers.
        (prepared_obs, prefix_tokens, prefix_mask, kv_cache), prefix_compile_ms = time_jax_call(
            prefix_jit, state, observation
        )
        x = noise_jax
        t = jnp.asarray(1.0)
        _, step_compile_ms = time_jax_call(step_jit, state, prepared_obs, prefix_tokens, prefix_mask, kv_cache, x, t)

        prefix_times: list[float] = []
        step_times: list[float] = []

        for _ in range(10):
            _, dt = time_jax_call(prefix_jit, state, observation)
            prefix_times.append(dt)
            (prepared_obs, prefix_tokens, prefix_mask, kv_cache), _ = time_jax_call(prefix_jit, state, observation)
            x = noise_jax
            t = jnp.asarray(1.0)
            _, dt = time_jax_call(step_jit, state, prepared_obs, prefix_tokens, prefix_mask, kv_cache, x, t)
            step_times.append(dt)

        # The actual policy uses a compiled 10-step while_loop. Measure it separately
        # through the policy's exact jitted sampler.
        sampler = policy._sample_actions  # noqa: SLF001
        rng = jax.random.key(0)
        _, full_compile_ms = time_jax_call(sampler, rng, observation, num_steps=10, noise=noise_jax)
        full_times: list[float] = []
        for _ in range(20):
            _, dt = time_jax_call(sampler, rng, observation, num_steps=10, noise=noise_jax)
            full_times.append(dt)

        return {
            "prefix_cache": summary(prefix_times),
            "flow_step_one": summary(step_times),
            "sampler_10_steps": summary(full_times),
            "compile_ms": {
                "prefix_cache": prefix_compile_ms,
                "flow_step_one": step_compile_ms,
                "sampler_10_steps": full_compile_ms,
            },
            "derived_10_flow_steps_ms": {
                "prefix_plus_10_steps_mean": summary(prefix_times)["mean_ms"] + 10 * summary(step_times)["mean_ms"],
                "sampler_10_steps_mean": summary(full_times)["mean_ms"],
            },
        }

    # Discrete internal wrappers.
    def prefix_fun(state, observation):
        module = nnx.merge(graphdef, state)
        observation = _model.preprocess_observation(None, observation, train=False)
        cache, prefix_mask = module._cache(observation)  # noqa: SLF001
        return observation, cache, prefix_mask

    def discrete_fun(state, observation, cache, prefix_mask, noise):
        module = nnx.merge(graphdef, state)
        tokens = jnp.full((batch, module.codec.coarse_horizon * module.codec.dim), 256, jnp.int32)
        logits = module._predict(  # noqa: SLF001
            observation, noise, tokens, ones, ones, cache, prefix_mask, "discrete"
        )
        return jnp.argmax(logits, axis=-1)

    def decode_fun(state, noise, tokens):
        module = nnx.merge(graphdef, state)
        return module._source(noise, tokens)  # noqa: SLF001

    def continuous_fun(state, observation, cache, prefix_mask, source, tokens):
        module = nnx.merge(graphdef, state)
        return module._predict(  # noqa: SLF001
            observation, source, tokens, ones, zeros, cache, prefix_mask, "continuous"
        )

    prefix_jit = jax.jit(prefix_fun)
    discrete_jit = jax.jit(discrete_fun)
    decode_jit = jax.jit(decode_fun)
    continuous_jit = jax.jit(continuous_fun)

    (prepared_obs, cache, prefix_mask), prefix_compile_ms = time_jax_call(prefix_jit, state, observation)
    tokens, discrete_compile_ms = time_jax_call(discrete_jit, state, prepared_obs, cache, prefix_mask, noise_jax)
    source, decode_compile_ms = time_jax_call(decode_jit, state, noise_jax, tokens)
    _, continuous_compile_ms = time_jax_call(continuous_jit, state, prepared_obs, cache, prefix_mask, source, tokens)

    prefix_times: list[float] = []
    discrete_times: list[float] = []
    decode_times: list[float] = []
    continuous_times: list[float] = []
    for _ in range(20):
        (prepared_obs, cache, prefix_mask), dt = time_jax_call(prefix_jit, state, observation)
        prefix_times.append(dt)
        tokens, dt = time_jax_call(discrete_jit, state, prepared_obs, cache, prefix_mask, noise_jax)
        discrete_times.append(dt)
        source, dt = time_jax_call(decode_jit, state, noise_jax, tokens)
        decode_times.append(dt)
        _, dt = time_jax_call(continuous_jit, state, prepared_obs, cache, prefix_mask, source, tokens)
        continuous_times.append(dt)

    sampler = policy._sample_actions  # noqa: SLF001
    rng = jax.random.key(0)
    _, full_compile_ms = time_jax_call(sampler, rng, observation, noise=noise_jax)
    full_times: list[float] = []
    for _ in range(20):
        _, dt = time_jax_call(sampler, rng, observation, noise=noise_jax)
        full_times.append(dt)

    return {
        "prefix_cache": summary(prefix_times),
        "discrete_argmax_pass": summary(discrete_times),
        "decode_and_mix": summary(decode_times),
        "continuous_flow_pass": summary(continuous_times),
        "sampler_fixed_two_passes": summary(full_times),
        "compile_ms": {
            "prefix_cache": prefix_compile_ms,
            "discrete_argmax_pass": discrete_compile_ms,
            "decode_and_mix": decode_compile_ms,
            "continuous_flow_pass": continuous_compile_ms,
            "sampler_fixed_two_passes": full_compile_ms,
        },
    }


def benchmark_policy(policy, samples: list[Sample], model_name: str, warmup: int, repeats: int) -> dict[str, Any]:
    # Prepare all samples once, outside the steady-state inference timing. This lets
    # us report input preprocessing separately while keeping model comparisons fair.
    prepared: list[tuple[dict[str, Any], dict[str, float], np.ndarray]] = []
    input_timing: dict[str, list[float]] = {"input_transform_ms": [], "host_to_device_ms": []}
    for sample in samples:
        batched, timing = prepare_policy_input(policy, sample.observation)
        prepared.append((batched, timing, make_noise(model_name)))
        for key, value in timing.items():
            input_timing[key].append(value)

    # Compile the exact Policy sampler once and then exclude it from steady state.
    sample0, _, noise0 = prepared[0]
    observation0 = _model.Observation.from_dict(sample0)
    rng0 = jax.random.key(0)
    t0 = now_ns()
    result0 = policy._sample_actions(rng0, observation0, noise=jnp.asarray(noise0)[None, ...], **policy._sample_kwargs)  # noqa: SLF001
    sync_tree(result0)
    sampler_compile_ms = elapsed_ms(t0)

    # Warm-up calls use real samples, warming up all paths:
    # 1) sample_actions
    # 2) device-to-host transfer
    # 3) output transforms
    # 4) public Policy.infer
    for i in range(warmup):
        sample, _, noise = prepared[i % len(prepared)]
        obs = _model.Observation.from_dict(sample)
        out = policy._sample_actions(  # noqa: SLF001
            jax.random.key(i + 1),
            obs,
            noise=jnp.asarray(noise)[None, ...],
            **policy._sample_kwargs,  # noqa: SLF001
        )
        sync_tree(out)
        out_host = np.asarray(out[0])
        _ = policy._output_transform({"state": sample["state"], "actions": out_host})  # noqa: SLF001
        raw = samples[i % len(samples)].observation
        res = policy.infer(raw, noise=noise)
        _ = np.asarray(res["actions"])

    stage_input_transform: list[float] = []
    stage_host_to_device: list[float] = []
    stage_model_sampler: list[float] = []
    stage_device_to_host: list[float] = []
    stage_output_transform: list[float] = []
    end_to_end_infer: list[float] = []
    result_shapes: list[list[int]] = []

    for i in range(repeats):
        raw_obs = samples[i % len(samples)].observation
        noise = prepared[i % len(prepared)][2]
        raw = jax.tree.map(lambda x: x.copy() if isinstance(x, np.ndarray) else x, raw_obs)

        # 1. Input transform
        t0 = now_ns()
        inputs = policy._input_transform(raw)  # noqa: SLF001
        stage_input_transform.append(elapsed_ms(t0))

        # 2. Host to device transfer
        t0 = now_ns()
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        sync_tree(inputs)
        stage_host_to_device.append(elapsed_ms(t0))

        obs = _model.Observation.from_dict(inputs)
        rng = jax.random.key(1000 + i)

        # 3. Model execution
        t0 = now_ns()
        actions = policy._sample_actions(  # noqa: SLF001
            rng,
            obs,
            noise=jnp.asarray(noise)[None, ...],
            **policy._sample_kwargs,  # noqa: SLF001
        )
        sync_tree(actions)
        stage_model_sampler.append(elapsed_ms(t0))

        # 4. Device to host transfer
        t0 = now_ns()
        actions_host = np.asarray(actions[0])
        stage_device_to_host.append(elapsed_ms(t0))

        # 5. Output transform
        outputs = {"state": inputs["state"], "actions": actions_host}
        t0 = now_ns()
        outputs = policy._output_transform(outputs)  # noqa: SLF001
        stage_output_transform.append(elapsed_ms(t0))
        result_shapes.append(list(np.asarray(outputs["actions"]).shape))

        # 6. Public Policy.infer end-to-end
        t0 = now_ns()
        public_result = policy.infer(raw_obs, noise=noise)
        _ = np.asarray(public_result["actions"])
        end_to_end_infer.append(elapsed_ms(t0))

    return {
        "breakdown": {
            "input_transform_ms": summary(stage_input_transform),
            "host_to_device_ms": summary(stage_host_to_device),
            "model_sampler_ms": summary(stage_model_sampler),
            "device_to_host_ms": summary(stage_device_to_host),
            "output_transform_ms": summary(stage_output_transform),
        },
        "public_policy_infer_end_to_end_ms": summary(end_to_end_infer),
        "exact_sampler_compile_first_call_ms": sampler_compile_ms,
        "result_shapes": sorted({tuple(x) for x in result_shapes}),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["discrete", "standard"], required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--sample-stride", type=int, default=37)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    print(f"jax version: {jax.__version__}")
    print(f"jax devices: {jax.devices()}")
    if not any(device.platform == "gpu" for device in jax.devices()):
        raise RuntimeError("A CUDA JAX device is required for this benchmark.")

    t0 = now_ns()
    samples, dataset_info = load_samples(args.num_samples, args.sample_stride)
    dataset_load_ms = elapsed_ms(t0)
    print(
        f"loaded {len(samples)} samples in {dataset_load_ms:.2f} ms from {dataset_info['source_files']} parquet files"
    )

    t0 = now_ns()
    cfg, policy = make_policy(args.model)
    model_load_ms = elapsed_ms(t0)
    print(f"loaded {args.model} policy in {model_load_ms / 1000:.2f} s")

    # The first model-internal benchmark also compiles the separated timing wrappers.
    first_prepared, _ = prepare_policy_input(policy, samples[0].observation)
    first_noise = make_noise(args.model)
    t0 = now_ns()
    internal = model_internal_benchmarks(policy, args.model, first_prepared, first_noise)
    internal_total_ms = elapsed_ms(t0)
    print(f"internal timing wrappers completed in {internal_total_ms / 1000:.2f} s")

    t0 = now_ns()
    policy_level = benchmark_policy(policy, samples, args.model, args.warmup, args.repeats)
    policy_level_total_ms = elapsed_ms(t0)
    print(f"policy-level benchmark completed in {policy_level_total_ms / 1000:.2f} s")

    output = {
        "benchmark": {
            "timestamp_unix": time.time(),
            "model": args.model,
            "checkpoint": DISCRETE_CHECKPOINT if args.model == "discrete" else STANDARD_CHECKPOINT,
            "config": cfg.name,
            "gpu_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jax_devices": [str(x) for x in jax.devices()],
            "warmup": args.warmup,
            "repeats": args.repeats,
            "fixed_noise": True,
            "noise_shape": [30, 32],
            "sample_count": len(samples),
            "sample_metadata": [
                {
                    "file": sample.source_file,
                    "row": sample.row,
                    "episode_index": sample.episode_index,
                    "frame_index": sample.frame_index,
                }
                for sample in samples
            ],
        },
        "dataset": dataset_info | {"parquet_read_and_decode_total_ms": dataset_load_ms},
        "model_load_ms": model_load_ms,
        "internal_model_timing": internal,
        "policy_level_timing": policy_level,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True))
    print(json.dumps(output["internal_model_timing"], indent=2, sort_keys=True))
    print(json.dumps(output["policy_level_timing"], indent=2, sort_keys=True))
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()

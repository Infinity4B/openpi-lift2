"""Evaluate a trained LIFT2 policy on its LeRobot training dataset.

This evaluator uses the same dataset and model transforms as training, compares the
sampled 30-step action chunks against the dataset actions, ignores episode-end
padding, and measures synchronized JAX model latency.

Example:
    CUDA_VISIBLE_DEVICES=0 uv run scripts/eval_lift2_dataset.py \
        --checkpoint checkpoints/pi05_lift2_df_image/lift2_df_image_full/99999 \
        --max-samples 1024
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import dataclasses
import json
import logging
import pathlib
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import torch

import openpi.models.model as _model
import openpi.policies.policy_config as _policy_config
import openpi.training.checkpoints as _checkpoints
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader
import openpi.transforms as _transforms

ACTION_LABELS = (
    "L_dx",
    "L_dy",
    "L_dz",
    "L_droll",
    "L_dpitch",
    "L_dyaw",
    "L_gripper",
    "R_dx",
    "R_dy",
    "R_dz",
    "R_droll",
    "R_dpitch",
    "R_dyaw",
    "R_gripper",
)
GROUPS = {
    "Left XYZ": slice(0, 3),
    "Left RPY": slice(3, 6),
    "Left Gripper": slice(6, 7),
    "Right XYZ": slice(7, 10),
    "Right RPY": slice(10, 13),
    "Right Gripper": slice(13, 14),
}


class EvalDataset(torch.utils.data.Dataset):
    """Apply the training input transforms while retaining the padding mask."""

    def __init__(self, dataset: _data_loader.Dataset, data_config: _config.DataConfig):
        self._dataset = dataset
        norm_stats = data_config.norm_stats
        self._transform = _transforms.compose(
            [
                *data_config.repack_transforms.inputs,
                *data_config.data_transforms.inputs,
                _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
                *data_config.model_transforms.inputs,
            ]
        )

    def __len__(self) -> int:
        return len(self._dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._dataset[index]
        action_is_pad = np.asarray(sample.pop("action_is_pad"), dtype=bool)
        episode_index = int(np.asarray(sample["episode_index"]).item())
        task_index = int(np.asarray(sample["task_index"]).item())
        transformed = self._transform(sample)
        return {
            "model": transformed,
            "action_is_pad": action_is_pad,
            "episode_index": episode_index,
            "task_index": task_index,
        }


def collate_eval(items: list[dict[str, Any]]) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    models = [item["model"] for item in items]
    model_batch = jax.tree.map(
        lambda *values: np.stack([np.asarray(value) for value in values], axis=0),
        *models,
    )
    return (
        model_batch,
        np.stack([item["action_is_pad"] for item in items], axis=0),
        np.asarray([item["episode_index"] for item in items], dtype=np.int64),
        np.asarray([item["task_index"] for item in items], dtype=np.int64),
    )


@dataclasses.dataclass
class ErrorAccumulator:
    sum_abs: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(14, dtype=np.float64))
    sum_sq: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(14, dtype=np.float64))
    max_abs: np.ndarray = dataclasses.field(default_factory=lambda: np.zeros(14, dtype=np.float64))
    count: int = 0

    def update(self, error: np.ndarray, valid: np.ndarray) -> None:
        flat_error = error.reshape(-1, 14)
        flat_valid = valid.reshape(-1)
        if not np.any(flat_valid):
            return
        selected = flat_error[flat_valid]
        abs_error = np.abs(selected)
        self.sum_abs += abs_error.sum(axis=0)
        self.sum_sq += np.square(selected).sum(axis=0)
        self.max_abs = np.maximum(self.max_abs, abs_error.max(axis=0))
        self.count += int(selected.shape[0])

    def to_dict(self) -> dict[str, Any]:
        if self.count == 0:
            return {"count": 0}
        mae = self.sum_abs / self.count
        rmse = np.sqrt(self.sum_sq / self.count)
        return {
            "count": self.count,
            "mae": {label: float(value) for label, value in zip(ACTION_LABELS, mae, strict=True)},
            "rmse": {label: float(value) for label, value in zip(ACTION_LABELS, rmse, strict=True)},
            "max_abs": {label: float(value) for label, value in zip(ACTION_LABELS, self.max_abs, strict=True)},
            "groups_mae": {
                name: float(mae[indices].mean()) for name, indices in GROUPS.items()
            },
            "overall_mae": float(mae.mean()),
            "overall_rmse": float(rmse.mean()),
        }


def _make_observation(model_batch: Mapping[str, Any]) -> _model.Observation:
    observation_dict = {key: value for key, value in model_batch.items() if key != "actions"}
    observation = _model.Observation.from_dict(observation_dict)
    return jax.tree.map(jnp.asarray, observation)


def _slice_batch(model_batch: Mapping[str, Any], count: int) -> dict[str, Any]:
    return jax.tree.map(lambda value: np.asarray(value)[:count], model_batch)


def _unnormalize_quantile(actions: np.ndarray, norm_stats: Mapping[str, _transforms.NormStats]) -> np.ndarray:
    stats = norm_stats["actions"]
    if stats.q01 is None or stats.q99 is None:
        raise ValueError("Expected q01/q99 action statistics for PI05 quantile normalization")
    q01 = np.asarray(stats.q01, dtype=np.float32)[:14]
    q99 = np.asarray(stats.q99, dtype=np.float32)[:14]
    return (actions[..., :14] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01


def _summary(accumulator: ErrorAccumulator) -> dict[str, Any]:
    result = accumulator.to_dict()
    if result["count"] == 0:
        return result
    rmse_values = np.asarray([result["rmse"][label] for label in ACTION_LABELS])
    result["groups_rmse"] = {
        name: float(rmse_values[indices].mean()) for name, indices in GROUPS.items()
    }
    return result


def _timing_summary(times: list[float], sample_count: int) -> dict[str, float | int]:
    if not times:
        return {"timed_batches": 0, "sample_count": sample_count}
    values = np.asarray(times, dtype=np.float64) * 1000.0
    return {
        "timed_batches": len(times),
        "sample_count": sample_count,
        "average_batch_ms": float(values.mean()),
        "median_batch_ms": float(np.median(values)),
        "p95_batch_ms": float(np.percentile(values, 95)),
        "average_ms_per_sample": float(values.mean() / sample_count),
        "samples_per_second": float(sample_count / (values.sum() / 1000.0)),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_lift2_df_image")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/pi05_lift2_df_image/lift2_df_image_full/99999",
    )
    parser.add_argument("--dataset", default="lerobot_lift2_df_image")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--warmup-batches", type=int, default=2)
    parser.add_argument("--speed-repeats", type=int, default=20)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--max-samples", type=int, default=0, help="0 evaluates the full dataset")
    parser.add_argument(
        "--start-sample",
        type=int,
        default=0,
        help="Global dataset sample index at which to start evaluation",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=pathlib.Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    checkpoint = pathlib.Path(args.checkpoint).resolve()

    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers must be non-negative")
    if args.num_steps < 1:
        raise ValueError("--num-steps must be positive")
    if args.start_sample < 0:
        raise ValueError("--start-sample must be non-negative")

    if args.speed_repeats < 1:
        raise ValueError("--speed-repeats must be positive")
    base_config = _config.get_config(args.config)
    config = dataclasses.replace(base_config, batch_size=args.batch_size, num_workers=args.num_workers)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.asset_id is None:
        raise ValueError("The data config must define an asset id")
    norm_stats = _checkpoints.load_norm_stats(str(checkpoint / "assets"), data_config.asset_id)
    if norm_stats is None:
        raise ValueError(f"No normalization statistics found in {checkpoint / 'assets' / data_config.asset_id}")
    data_config = dataclasses.replace(data_config, norm_stats=norm_stats)

    logging.info("Loading policy from %s", checkpoint)
    policy = _policy_config.create_trained_policy(
        config,
        checkpoint,
        default_prompt=None,
        norm_stats=norm_stats,
    )

    logging.info("Creating dataset %s", args.dataset)
    dataset_config = dataclasses.replace(data_config, repo_id=args.dataset)
    raw_dataset = _data_loader.create_torch_dataset(dataset_config, config.model.action_horizon, config.model)
    eval_dataset = EvalDataset(raw_dataset, dataset_config)
    total_dataset_samples = len(eval_dataset)
    if args.start_sample >= total_dataset_samples:
        raise ValueError(
            f"--start-sample ({args.start_sample}) must be less than dataset size ({total_dataset_samples})"
        )
    end_sample = total_dataset_samples if args.max_samples <= 0 else min(
        args.start_sample + args.max_samples, total_dataset_samples
    )
    selected_dataset = torch.utils.data.Subset(eval_dataset, range(args.start_sample, end_sample))
    loader = torch.utils.data.DataLoader(
        selected_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        multiprocessing_context="spawn" if args.num_workers > 0 else None,
        collate_fn=collate_eval,
        drop_last=False,
    )

    requested_samples = len(selected_dataset)
    logging.info(
        "Evaluating samples [%d, %d) (%d/%d) with batch size %d, action horizon %d, diffusion steps %d",
        args.start_sample,
        end_sample,
        requested_samples,
        total_dataset_samples,
        args.batch_size,
        config.model.action_horizon,
        args.num_steps,
    )

    rng = jax.random.key(args.seed)
    overall = ErrorAccumulator()
    overall_normalized = ErrorAccumulator()
    first_step = ErrorAccumulator()
    first_step_normalized = ErrorAccumulator()
    by_task: dict[int, ErrorAccumulator] = {}
    by_task_normalized: dict[int, ErrorAccumulator] = {}
    timing: list[float] = []
    timed_samples = 0
    first_observation: _model.Observation | None = None
    processed_samples = 0

    for batch_index, (raw_model_batch, raw_action_is_pad, _episode_indices, raw_task_indices) in enumerate(loader):
        remaining = requested_samples - processed_samples
        if remaining <= 0:
            break
        take = min(remaining, raw_model_batch["actions"].shape[0])
        model_batch = _slice_batch(raw_model_batch, take)
        action_is_pad = raw_action_is_pad[:take]
        task_indices = raw_task_indices[:take]
        observation = _make_observation(model_batch)
        if first_observation is None:
            first_observation = jax.tree.map(lambda x: x[:1], observation)

        rng, sample_rng = jax.random.split(rng)
        start_time = time.perf_counter()
        predictions = policy._sample_actions(  # noqa: SLF001
            sample_rng,
            observation,
            num_steps=args.num_steps,
        )
        predictions = jax.block_until_ready(predictions)
        elapsed = time.perf_counter() - start_time
        predictions = np.asarray(predictions)[..., :14]

        targets_normalized = np.asarray(model_batch["actions"])[..., :14]
        valid = ~action_is_pad
        normalized_error = predictions - targets_normalized
        predictions_raw = _unnormalize_quantile(predictions, norm_stats)
        targets_raw = _unnormalize_quantile(targets_normalized, norm_stats)
        error = predictions_raw - targets_raw

        overall.update(error, valid)
        overall_normalized.update(normalized_error, valid)
        first_step.update(error[:, 0], valid[:, 0])
        first_step_normalized.update(normalized_error[:, 0], valid[:, 0])
        for task_index in np.unique(task_indices):
            task_mask = task_indices == task_index
            task_acc = by_task.setdefault(int(task_index), ErrorAccumulator())
            task_acc.update(error[task_mask], valid[task_mask])
            task_acc_normalized = by_task_normalized.setdefault(int(task_index), ErrorAccumulator())
            task_acc_normalized.update(normalized_error[task_mask], valid[task_mask])

        if batch_index >= args.warmup_batches:
            timing.append(elapsed)
            timed_samples += take
        processed_samples += take
        if batch_index % 20 == 0 or processed_samples == requested_samples:
            logging.info(
                "progress: %d/%d samples, last batch %.2f ms, overall raw MAE %.6f",
                processed_samples,
                requested_samples,
                elapsed * 1000.0,
                overall.to_dict().get("overall_mae", float("nan")),
            )

    if processed_samples == 0 or first_observation is None:
        raise RuntimeError("No samples were evaluated")

    online_times: list[float] = []
    rng, online_rng = jax.random.split(rng)
    warmup_prediction = policy._sample_actions(  # noqa: SLF001
        online_rng,
        first_observation,
        num_steps=args.num_steps,
    )
    jax.block_until_ready(warmup_prediction)
    for _ in range(args.speed_repeats):
        rng, online_rng = jax.random.split(rng)
        start_time = time.perf_counter()
        prediction = policy._sample_actions(  # noqa: SLF001
            online_rng,
            first_observation,
            num_steps=args.num_steps,
        )
        jax.block_until_ready(prediction)
        online_times.append(time.perf_counter() - start_time)

    online_ms = np.asarray(online_times, dtype=np.float64) * 1000.0
    task_names = {
        0: "Grasp the moving cube and place it on the plate.",
        1: "Identify and pick up the illuminated red light from the rotating turntable, then place it aside.",
        2: "Pick up the four randomly placed cylinders and insert each one into the matching hole according to its size.",
        3: "Transfer the test tube from the right rack to the left rack.",
    }
    result = {
        "config": args.config,
        "checkpoint": str(checkpoint),
        "dataset": args.dataset,
        "dataset_samples": total_dataset_samples,
        "start_sample": args.start_sample,
        "end_sample": end_sample,
        "evaluated_samples": processed_samples,
        "valid_action_steps": overall.count,
        "action_horizon": config.model.action_horizon,
        "diffusion_steps": args.num_steps,
        "batch_size": args.batch_size,
        "metrics_raw_action_units": {
            "all_horizon": _summary(overall),
            "first_action": _summary(first_step),
            "by_task_all_horizon": {
                str(task_index): {
                    "task": task_names.get(task_index, f"task_{task_index}"),
                    "metrics": _summary(accumulator),
                }
                for task_index, accumulator in sorted(by_task.items())
            },
        },
        "metrics_normalized_model_units": {
            "all_horizon": _summary(overall_normalized),
            "first_action": _summary(first_step_normalized),
            "by_task_all_horizon": {
                str(task_index): _summary(by_task_normalized[task_index])
                for task_index in sorted(by_task_normalized)
            },
        },
        "timing_model_only": {
            "batch": _timing_summary(timing, timed_samples),
            "batch_1_online": {
                "repeats": len(online_times),
                "average_ms": float(online_ms.mean()),
                "median_ms": float(np.median(online_ms)),
                "p95_ms": float(np.percentile(online_ms, 95)),
                "inferences_per_second": float(1000.0 / online_ms.mean()),
            },
            "note": "Timing includes synchronized JAX sample_actions only; dataset decoding and preprocessing are excluded.",
        },
    }

    output = args.output or pathlib.Path("logs") / "pi05_lift2_df_image_train_eval.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    logging.info("Saved evaluation result to %s", output)


if __name__ == "__main__":
    main()

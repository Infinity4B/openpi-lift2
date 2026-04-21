"""Analyze HDF5 trajectory quality, hard-filter broken data, then rank by anomaly score.

This script follows a multi-signal quality filtering pipeline:
1. Extract per-trajectory metrics:
   - structure integrity
   - relative length anomaly
   - action change
   - image change
   - state change
2. Hard-drop clearly broken / misgenerated trajectories.
3. Rank remaining trajectories by anomaly score.
4. Split into keep / drop, preview the statistics, and optionally move keep files.

Examples:
    uv run python scripts/filter_dataset_by_length.py \
        --data-dir ./dataset_0410_towel

    uv run python scripts/filter_dataset_by_length.py \
        --data-dir ./dataset_0410_towel \
        --target-count 150

    uv run python scripts/filter_dataset_by_length.py \
        --data-dir ./dataset_0410_towel \
        --target-count 150 \
        --output-dir ./dataset_0410_towel_keep_150
"""

import argparse
import shutil
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import h5py
import numpy as np


CAMERA_NAMES = ("head", "left_wrist", "right_wrist")
EPS = 1e-8


@dataclass
class TrajectoryMetrics:
    episode_id: int
    path: Path
    length: int = 0
    action_mean: float = np.nan
    action_p90: float = np.nan
    zero_action_ratio: float = np.nan
    total_motion: float = np.nan
    state_span: float = np.nan
    static_ratio: float = np.nan
    image_change: float = np.nan
    image_std: float = np.nan
    structure_reasons: list[str] = field(default_factory=list)
    hard_reasons: list[str] = field(default_factory=list)
    anomaly_components: dict[str, float] = field(default_factory=dict)
    anomaly_score: float = np.nan
    split: str = "unassigned"


@dataclass
class MetricReference:
    median: float
    scale: float
    p05: float
    p50: float
    p95: float


def parse_episode_id(path: Path) -> int:
    return int(path.stem.split("_")[1])


def get_episode_paths(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.glob("episode_*.hdf5"), key=parse_episode_id)
    if not paths:
        raise FileNotFoundError(f"No episode_*.hdf5 files found in {data_dir}")
    return paths


def robust_scale(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return max(1.4826 * mad, EPS)


def build_reference(values: np.ndarray) -> MetricReference:
    return MetricReference(
        median=float(np.median(values)),
        scale=robust_scale(values),
        p05=float(np.percentile(values, 5)),
        p50=float(np.percentile(values, 50)),
        p95=float(np.percentile(values, 95)),
    )


def summary(values: np.ndarray) -> str:
    return (
        f"p05={np.percentile(values, 5):.4f}, "
        f"p50={np.percentile(values, 50):.4f}, "
        f"p95={np.percentile(values, 95):.4f}"
    )


def safe_log(value: float) -> float:
    return float(np.log(max(value, EPS)))


def two_sided_score(value: float, ref: MetricReference) -> float:
    return abs(value - ref.median) / ref.scale


def upper_score(value: float, ref: MetricReference) -> float:
    return max(0.0, value - ref.median) / ref.scale


def lower_score(value: float, ref: MetricReference) -> float:
    return max(0.0, ref.median - value) / ref.scale


def sample_frame_indices(length: int, sample_frames: int) -> np.ndarray:
    if length <= 0:
        return np.array([], dtype=int)
    return np.unique(np.linspace(0, length - 1, min(sample_frames, length), dtype=int))


def load_gray_image(encoded: np.ndarray, image_size: int) -> np.ndarray | None:
    image = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    return cv2.resize(image, (image_size, image_size), interpolation=cv2.INTER_AREA)


def compute_action_metrics(eef: np.ndarray) -> tuple[float, float, float, float, float]:
    if len(eef) < 2:
        return 0.0, 0.0, 1.0, 0.0, 1.0

    eef_diff = np.diff(eef, axis=0)
    xyz_step = np.linalg.norm(eef_diff[:, :3], axis=1) + np.linalg.norm(eef_diff[:, 7:10], axis=1)
    rpy_step = np.linalg.norm(eef_diff[:, 3:6], axis=1) + np.linalg.norm(eef_diff[:, 10:13], axis=1)
    grip_step = np.abs(eef_diff[:, 6]) + np.abs(eef_diff[:, 13])
    action_mag = xyz_step + 0.1 * rpy_step + 0.02 * grip_step

    return (
        float(np.mean(action_mag)),
        float(np.percentile(action_mag, 90)),
        float(np.mean(action_mag < 0.0015)),
        float(np.sum(xyz_step)),
        float(np.mean(action_mag < 0.0015)),
    )


def compute_state_span(eef: np.ndarray) -> float:
    if len(eef) == 0:
        return 0.0
    left_span = np.linalg.norm(eef[-1, :3] - eef[0, :3])
    right_span = np.linalg.norm(eef[-1, 7:10] - eef[0, 7:10])
    return float(left_span + right_span)


def compute_image_metrics(f: h5py.File, length: int, sample_frames: int, image_size: int) -> tuple[float, float, list[str]]:
    frame_indices = sample_frame_indices(length, sample_frames)
    image_change_values = []
    image_std_values = []
    reasons: list[str] = []

    for camera_name in CAMERA_NAMES:
        dataset_key = f"observations/images/{camera_name}"
        if dataset_key not in f:
            reasons.append(f"missing_{camera_name}_images")
            continue

        dataset = f[dataset_key]
        previous = None
        camera_changes = []
        camera_stds = []

        for frame_idx in frame_indices:
            image = load_gray_image(dataset[frame_idx], image_size)
            if image is None:
                reasons.append(f"decode_failed_{camera_name}")
                break

            camera_stds.append(float(image.std()))
            if previous is not None:
                diff = float(np.mean(np.abs(image.astype(np.float32) - previous.astype(np.float32))))
                camera_changes.append(diff)
            previous = image

        if camera_stds:
            image_std_values.append(float(np.mean(camera_stds)))
        if camera_changes:
            image_change_values.append(float(np.mean(camera_changes)))

    if reasons:
        return np.nan, np.nan, reasons

    return float(np.mean(image_change_values)), float(np.mean(image_std_values)), []


def collect_metrics(path: Path, sample_frames: int, image_size: int) -> TrajectoryMetrics:
    metrics = TrajectoryMetrics(episode_id=parse_episode_id(path), path=path)

    try:
        with h5py.File(path, "r") as f:
            if "observations/eef" not in f:
                metrics.structure_reasons.append("missing_observations_eef")
                return metrics

            eef = f["observations/eef"][:]
            if eef.ndim != 2 or eef.shape[1] != 14:
                metrics.structure_reasons.append("invalid_eef_shape")
                return metrics
            if not np.isfinite(eef).all():
                metrics.structure_reasons.append("non_finite_eef")
                return metrics

            metrics.length = int(eef.shape[0])
            if metrics.length < 2:
                metrics.structure_reasons.append("too_short_to_analyze")
                return metrics

            required_images = [
                "observations/images/head",
                "observations/images/left_wrist",
                "observations/images/right_wrist",
            ]
            image_lengths = []
            for key in required_images:
                if key not in f:
                    metrics.structure_reasons.append(f"missing_{key.replace('/', '_')}")
                    continue
                dataset = f[key]
                if dataset.ndim != 2:
                    metrics.structure_reasons.append(f"invalid_shape_{key.replace('/', '_')}")
                    continue
                image_lengths.append(int(dataset.shape[0]))

            if metrics.structure_reasons:
                return metrics

            if any(image_length != metrics.length for image_length in image_lengths):
                metrics.structure_reasons.append("image_length_mismatch")
                return metrics

            if "action_eef" in f:
                action_eef = f["action_eef"]
                if action_eef.ndim != 2 or action_eef.shape[1] != 14:
                    metrics.structure_reasons.append("invalid_action_eef_shape")
                    return metrics
                if action_eef.shape[0] != metrics.length:
                    metrics.structure_reasons.append("action_eef_length_mismatch")
                    return metrics

            (
                metrics.action_mean,
                metrics.action_p90,
                metrics.zero_action_ratio,
                metrics.total_motion,
                metrics.static_ratio,
            ) = compute_action_metrics(eef)
            metrics.state_span = compute_state_span(eef)
            metrics.image_change, metrics.image_std, image_reasons = compute_image_metrics(
                f, metrics.length, sample_frames, image_size
            )
            metrics.structure_reasons.extend(image_reasons)
    except OSError:
        metrics.structure_reasons.append("hdf5_open_failed")
    except Exception as exc:  # pragma: no cover - defensive reporting
        metrics.structure_reasons.append(f"unexpected_error_{type(exc).__name__}")

    return metrics


def build_metric_references(records: list[TrajectoryMetrics]) -> dict[str, MetricReference]:
    return {
        "log_length": build_reference(np.array([safe_log(record.length) for record in records])),
        "log_action_mean": build_reference(np.array([safe_log(record.action_mean) for record in records])),
        "log_action_p90": build_reference(np.array([safe_log(record.action_p90) for record in records])),
        "log_total_motion": build_reference(np.array([safe_log(record.total_motion) for record in records])),
        "static_ratio": build_reference(np.array([record.static_ratio for record in records])),
        "log_image_change": build_reference(np.array([safe_log(record.image_change) for record in records])),
        "log_image_std": build_reference(np.array([safe_log(record.image_std) for record in records])),
    }


def apply_hard_rules(records: list[TrajectoryMetrics]) -> tuple[list[TrajectoryMetrics], dict[str, float]]:
    if not records:
        return records, {}

    total_motion = np.array([record.total_motion for record in records])
    action_p90 = np.array([record.action_p90 for record in records])
    static_ratio = np.array([record.static_ratio for record in records])
    image_change = np.array([record.image_change for record in records])
    image_std = np.array([record.image_std for record in records])

    thresholds = {
        "low_total_motion": max(0.1, float(np.percentile(total_motion, 5)) * 0.2),
        "low_action_p90": max(0.001, float(np.percentile(action_p90, 5)) * 0.2),
        "high_static_ratio": min(0.98, max(0.85, float(np.percentile(static_ratio, 95)))),
        "low_image_change": max(2.0, float(np.percentile(image_change, 5)) * 0.2),
        "low_image_std": max(8.0, float(np.percentile(image_std, 5)) * 0.5),
    }

    for record in records:
        if record.total_motion <= thresholds["low_total_motion"] and record.image_change <= thresholds["low_image_change"]:
            record.hard_reasons.append("near_static_state_and_image")
        if record.action_p90 <= thresholds["low_action_p90"] and record.static_ratio >= thresholds["high_static_ratio"]:
            record.hard_reasons.append("near_zero_action_and_high_static_ratio")
        if record.image_std <= thresholds["low_image_std"] and record.image_change <= thresholds["low_image_change"]:
            record.hard_reasons.append("low_information_images")

    return records, thresholds


def compute_anomaly_scores(records: list[TrajectoryMetrics]) -> dict[str, MetricReference]:
    refs = build_metric_references(records)
    for record in records:
        components = {
            "length": 0.4 * two_sided_score(safe_log(record.length), refs["log_length"]),
            "action_mean": 1.0 * two_sided_score(safe_log(record.action_mean), refs["log_action_mean"]),
            "action_p90": 1.0 * two_sided_score(safe_log(record.action_p90), refs["log_action_p90"]),
            "total_motion": 1.2 * two_sided_score(safe_log(record.total_motion), refs["log_total_motion"]),
            "static_ratio": 1.0 * upper_score(record.static_ratio, refs["static_ratio"]),
            "image_change": 1.0 * two_sided_score(safe_log(record.image_change), refs["log_image_change"]),
            "image_std": 0.8 * lower_score(safe_log(record.image_std), refs["log_image_std"]),
        }
        record.anomaly_components = components
        record.anomaly_score = float(sum(components.values()))
    return refs


def split_keep_drop(
    records: list[TrajectoryMetrics],
    target_count: int | None,
    score_mad_multiplier: float,
) -> tuple[list[TrajectoryMetrics], list[TrajectoryMetrics], float, str]:
    if not records:
        return [], [], np.nan, "no_valid_trajectories"

    ranked = sorted(records, key=lambda record: record.anomaly_score)
    scores = np.array([record.anomaly_score for record in ranked])

    if target_count is not None:
        keep_count = min(max(target_count, 0), len(ranked))
        score_threshold = ranked[keep_count - 1].anomaly_score if keep_count > 0 else float("-inf")
        mode = f"target_count={target_count}"
        keep = ranked[:keep_count]
        drop = ranked[keep_count:]
    else:
        median_score = float(np.median(scores))
        scale = robust_scale(scores)
        score_threshold = median_score + score_mad_multiplier * scale
        mode = f"median+{score_mad_multiplier:.2f}*MAD"
        keep = [record for record in ranked if record.anomaly_score <= score_threshold]
        drop = [record for record in ranked if record.anomaly_score > score_threshold]

    for record in keep:
        record.split = "keep"
    for record in drop:
        record.split = "drop"

    return keep, drop, float(score_threshold), mode


def reason_counts(records: list[TrajectoryMetrics], attribute: str) -> Counter:
    counter: Counter = Counter()
    for record in records:
        for reason in getattr(record, attribute):
            counter[reason] += 1
    return counter


def print_reason_summary(title: str, records: list[TrajectoryMetrics], attribute: str) -> None:
    counter = reason_counts(records, attribute)
    print(f"\n{title}: {len(records)}")
    if not counter:
        print("  (none)")
        return
    for reason, count in sorted(counter.items()):
        print(f"  {reason}: {count}")


def metric_array(records: list[TrajectoryMetrics], name: str) -> np.ndarray:
    return np.array([getattr(record, name) for record in records], dtype=np.float64)


def print_metric_summary(title: str, records: list[TrajectoryMetrics]) -> None:
    print(f"\n{title}: {len(records)}")
    if not records:
        print("  (none)")
        return
    for name in [
        "length",
        "action_mean",
        "action_p90",
        "zero_action_ratio",
        "total_motion",
        "state_span",
        "static_ratio",
        "image_change",
        "image_std",
    ]:
        values = metric_array(records, name)
        print(f"  {name:<16} {summary(values)}")


def print_rank_preview(title: str, records: list[TrajectoryMetrics], limit: int) -> None:
    print(f"\n{title}:")
    if not records:
        print("  (none)")
        return

    print(
        "  rank  score    length  act_mean  act_p90   total_m   static   img_chg  img_std"
    )
    for idx, record in enumerate(records[:limit], start=1):
        print(
            f"  {idx:>4}  {record.anomaly_score:>6.3f}  {record.length:>6d}  "
            f"{record.action_mean:>8.4f}  {record.action_p90:>8.4f}  {record.total_motion:>8.3f}  "
            f"{record.static_ratio:>7.3f}  {record.image_change:>8.3f}  {record.image_std:>7.3f}"
        )


def print_component_summary(records: list[TrajectoryMetrics], title: str) -> None:
    print(f"\n{title}:")
    if not records:
        print("  (none)")
        return

    keys = [
        "length",
        "action_mean",
        "action_p90",
        "total_motion",
        "static_ratio",
        "image_change",
        "image_std",
    ]
    for key in keys:
        values = np.array([record.anomaly_components.get(key, 0.0) for record in records], dtype=np.float64)
        print(f"  {key:<16} mean={values.mean():.4f}, p90={np.percentile(values, 90):.4f}")


def confirm_move(output_dir: Path, keep: list[TrajectoryMetrics], hard_drop: list[TrajectoryMetrics], soft_drop: list[TrajectoryMetrics]) -> bool:
    response = input(
        f"\nMove {len(keep)} keep trajectories into {output_dir}? "
        f"Hard drop={len(hard_drop)}, ranked drop={len(soft_drop)}. Type 'yes' to continue: "
    ).strip()
    return response == "yes"


def move_keep_records(keep: list[TrajectoryMetrics], output_dir: Path) -> None:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in keep:
        shutil.move(str(record.path), str(output_dir / record.path.name))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze trajectory quality with hard rules and anomaly scoring"
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="Directory with episode_*.hdf5")
    parser.add_argument(
        "--target-count",
        type=int,
        default=None,
        help="Keep this many lowest-anomaly trajectories after hard filtering",
    )
    parser.add_argument(
        "--score-mad-multiplier",
        type=float,
        default=2.5,
        help="Auto split threshold when --target-count is not provided",
    )
    parser.add_argument(
        "--sample-frames",
        type=int,
        default=12,
        help="How many frames per camera to decode for image metrics",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=64,
        help="Resize sampled grayscale images to this square size before comparison",
    )
    parser.add_argument(
        "--preview-rows",
        type=int,
        default=10,
        help="How many lowest/highest anomaly trajectories to preview",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Destination directory for moved keep trajectories",
    )
    args = parser.parse_args()

    all_records = [collect_metrics(path, args.sample_frames, args.image_size) for path in get_episode_paths(args.data_dir)]

    structural_invalid = [record for record in all_records if record.structure_reasons]
    structurally_valid = [record for record in all_records if not record.structure_reasons]

    print(f"Dataset: {args.data_dir}")
    print(f"Episodes: {len(all_records)}")
    print_reason_summary("Structure invalid trajectories", structural_invalid, "structure_reasons")

    if not structurally_valid:
        print("\nNo structurally valid trajectories left after integrity checks.")
        return

    structurally_valid, hard_thresholds = apply_hard_rules(structurally_valid)
    hard_drop = [record for record in structurally_valid if record.hard_reasons]
    score_pool = [record for record in structurally_valid if not record.hard_reasons]

    print("\nHard-rule thresholds:")
    for key, value in hard_thresholds.items():
        print(f"  {key}: {value:.4f}")
    print_reason_summary("Hard-drop trajectories", hard_drop, "hard_reasons")

    if not score_pool:
        print("\nNo trajectories left after hard filtering.")
        return

    refs = compute_anomaly_scores(score_pool)
    keep, ranked_drop, score_threshold, split_mode = split_keep_drop(
        score_pool,
        target_count=args.target_count,
        score_mad_multiplier=args.score_mad_multiplier,
    )

    print("\nReference metric ranges for scoring pool:")
    for name, ref in refs.items():
        print(f"  {name:<16} p05={ref.p05:.4f}, p50={ref.p50:.4f}, p95={ref.p95:.4f}")

    all_drops = hard_drop + ranked_drop
    print(
        f"\nSplit mode: {split_mode}, score_threshold={score_threshold:.4f}, "
        f"keep={len(keep)}, drop={len(all_drops)} (hard={len(hard_drop)}, ranked={len(ranked_drop)})"
    )

    print_metric_summary("Keep metric summary", keep)
    print_metric_summary("Ranked-drop metric summary", ranked_drop)
    print_metric_summary("All-drop metric summary", all_drops)

    ranked_keep = sorted(keep, key=lambda record: record.anomaly_score)
    ranked_drop_desc = sorted(ranked_drop, key=lambda record: record.anomaly_score, reverse=True)
    hard_drop_desc = sorted(
        hard_drop,
        key=lambda record: (record.total_motion, record.image_change, record.action_p90),
    )

    print_rank_preview("Lowest-anomaly keep preview", ranked_keep, args.preview_rows)
    print_rank_preview("Highest-anomaly ranked-drop preview", ranked_drop_desc, args.preview_rows)
    print_rank_preview("Hard-drop preview", hard_drop_desc, args.preview_rows)

    print_component_summary(ranked_keep, "Keep anomaly-component summary")
    print_component_summary(ranked_drop, "Ranked-drop anomaly-component summary")

    if args.output_dir is None:
        print("\nNo files moved. Re-run with --output-dir to move keep trajectories after confirmation.")
        return

    if not confirm_move(args.output_dir, keep, hard_drop, ranked_drop):
        print("Move cancelled.")
        return

    move_keep_records(keep, args.output_dir)
    print(f"Moved {len(keep)} files to {args.output_dir}")


if __name__ == "__main__":
    main()

"""
Convert LIFT2 HDF5 dataset to LeRobot format with EEF delta representation.

Uses the pre-computed observations/eef field directly from HDF5 (absolute EEF poses),
then computes delta actions for training.

Data format:
- observation.state: 14-dim absolute EEF [left_xyz(3), left_rpy(3), left_gripper_norm(1),
                                           right_xyz(3), right_rpy(3), right_gripper_norm(1)]
- action: 14-dim delta EEF [left_delta_xyz(3), left_delta_rpy(3), left_gripper_norm(1),
                             right_delta_xyz(3), right_delta_rpy(3), right_gripper_norm(1)]
  - xyz and rpy: delta (next - current)
  - gripper: absolute normalized [0, 1] (NOT delta)

Gripper normalization: raw [0, 5] -> normalized [0, 1]
  0 = fully closed, 5 = fully open

Usage:
uv run convert_hdf5_to_lerobot_eef.py \
    --data-dir ./datasets_mytask \
    --repo-id mytask_eef \
    --task-description "describe your task here"

Convert 60Hz recordings to a 30Hz LeRobot dataset:
uv run convert_hdf5_to_lerobot_eef.py \
    --data-dir ./dataset0319 \
    --repo-id 0319_pick_and_place_block_30hz \
    --task-description "Put the block on the plate." \
    --source-fps 60 \
    --fps 30
"""

import shutil
from pathlib import Path
import h5py
import cv2
import numpy as np
import tyro

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# Gripper normalization constants (matching X-VLA LIFT2 convention)
GRIPPER_MIN = 0.0  # Fully closed
GRIPPER_MAX = 5.0  # Fully open


def normalize_gripper(raw_value: float) -> float:
    """Normalize gripper from raw [0, 5] to [0, 1]. Clips out-of-range values."""
    return np.clip(raw_value, GRIPPER_MIN, GRIPPER_MAX) / GRIPPER_MAX


def build_eef_state(eef_raw: np.ndarray) -> np.ndarray:
    """
    Convert raw EEF observation to state with normalized gripper.

    Args:
        eef_raw: (14,) [left_xyz(3), left_rpy(3), left_gripper_raw(1),
                         right_xyz(3), right_rpy(3), right_gripper_raw(1)]

    Returns:
        eef_state: (14,) same layout but with normalized gripper [0, 1]
    """
    state = eef_raw.copy().astype(np.float32)
    state[6] = normalize_gripper(eef_raw[6])
    state[13] = normalize_gripper(eef_raw[13])
    return state


def compute_delta_action(current_eef_raw: np.ndarray, next_eef_raw: np.ndarray) -> np.ndarray:
    """
    Compute delta action from two consecutive raw EEF observations.

    Delta = next - current for xyz and rpy dimensions.
    Gripper = absolute normalized value of the NEXT frame (target gripper state).

    Args:
        current_eef_raw: (14,) current absolute EEF (raw gripper)
        next_eef_raw: (14,) next absolute EEF (raw gripper)

    Returns:
        action: (14,) [left_delta_xyz(3), left_delta_rpy(3), left_gripper_norm(1),
                        right_delta_xyz(3), right_delta_rpy(3), right_gripper_norm(1)]
    """
    action = np.zeros(14, dtype=np.float32)

    # Left arm: delta xyz + delta rpy
    action[0:3] = next_eef_raw[0:3] - current_eef_raw[0:3]
    action[3:6] = next_eef_raw[3:6] - current_eef_raw[3:6]
    # Left gripper: absolute normalized (target state)
    action[6] = normalize_gripper(next_eef_raw[6])

    # Right arm: delta xyz + delta rpy
    action[7:10] = next_eef_raw[7:10] - current_eef_raw[7:10]
    action[10:13] = next_eef_raw[10:13] - current_eef_raw[10:13]
    # Right gripper: absolute normalized (target state)
    action[13] = normalize_gripper(next_eef_raw[13])

    return action


def detect_motion_start(eef_data: np.ndarray, threshold: float = 0.001) -> int:
    """
    Detect the first frame where significant motion begins in either arm.

    Args:
        eef_data: [T, 14] EEF trajectory data
        threshold: Motion threshold in meters (default: 1mm)

    Returns:
        start_frame: Index of first frame with significant motion
    """
    # Compute deltas for both arms
    left_xyz = eef_data[:, :3]      # Left arm xyz
    right_xyz = eef_data[:, 7:10]   # Right arm xyz

    left_deltas = np.diff(left_xyz, axis=0)   # [T-1, 3]
    right_deltas = np.diff(right_xyz, axis=0)  # [T-1, 3]

    left_motion = np.linalg.norm(left_deltas, axis=1)   # [T-1]
    right_motion = np.linalg.norm(right_deltas, axis=1)  # [T-1]

    # Either arm moving counts as motion
    motion_magnitude = np.maximum(left_motion, right_motion)  # [T-1]

    # Find first frame with motion > threshold
    motion_frames = np.where(motion_magnitude > threshold)[0]

    if len(motion_frames) == 0:
        # No significant motion detected, use all frames
        return 0

    start_frame = motion_frames[0]
    return start_frame


def main(
    data_dir: str,
    repo_id: str,
    task_description: str,
    *,
    push_to_hub: bool = False,
    fps: int = 30,
    source_fps: int | None = None,
    skip_static_start: bool = True,
    motion_threshold: float = 0.001,
):
    """
    Convert HDF5 dataset to LeRobot format.

    Args:
        data_dir: Directory containing episode_*.hdf5 files
        repo_id: Dataset repository ID (e.g., "mytask_eef")
        task_description: Task description string (e.g., "pick and place")
        push_to_hub: Whether to push to HuggingFace Hub
        fps: Output LeRobot dataset frame rate (default: 30)
        source_fps: Source HDF5 recording frame rate. If set higher than fps, data is
            downsampled by an integer stride during conversion.
    """
    data_path = Path(data_dir)

    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    effective_source_fps = source_fps if source_fps is not None else fps
    if effective_source_fps <= 0:
        raise ValueError(f"source_fps must be positive, got {effective_source_fps}")
    if effective_source_fps < fps:
        raise ValueError(f"source_fps ({effective_source_fps}) must be >= fps ({fps})")
    if effective_source_fps % fps != 0:
        raise ValueError(
            f"source_fps ({effective_source_fps}) must be an integer multiple of fps ({fps})"
        )
    sample_stride = effective_source_fps // fps

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="bimanual",
        fps=fps,
        features={
            "observation.images.head": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.left_wrist": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.images.right_wrist": {
                "dtype": "image",
                "shape": (480, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["eef_state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (14,),
                "names": ["eef_delta"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    episode_files = sorted(data_path.glob("episode_*.hdf5"))
    print(f"Found {len(episode_files)} episodes")
    print(
        f"Converting from source_fps={effective_source_fps} to output_fps={fps} "
        f"(sample_stride={sample_stride})"
    )

    total_frames = 0
    for episode_file in episode_files:
        print(f"Processing {episode_file.name}...")

        with h5py.File(episode_file, 'r') as f:
            eef_raw = f['observations/eef'][:]  # [T, 14] absolute EEF
            num_frames = eef_raw.shape[0]

            # Detect motion start and skip initial static frames
            start_frame = 0
            if skip_static_start:
                start_frame = detect_motion_start(eef_raw, threshold=motion_threshold)
                if start_frame > 0:
                    print(f"  Skipping first {start_frame} static frames (motion threshold: {motion_threshold*1000:.1f}mm)")

            sampled_indices = list(range(start_frame, num_frames, sample_stride))
            if not sampled_indices:
                print("  No frames remain after sampling, skipping episode")
                continue

            for sampled_pos, frame_idx in enumerate(sampled_indices):
                # Decode images (JPEG bytes -> BGR -> RGB)
                head_img = cv2.cvtColor(
                    cv2.imdecode(f['observations/images/head'][frame_idx], cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                )
                left_wrist_img = cv2.cvtColor(
                    cv2.imdecode(f['observations/images/left_wrist'][frame_idx], cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                )
                right_wrist_img = cv2.cvtColor(
                    cv2.imdecode(f['observations/images/right_wrist'][frame_idx], cv2.IMREAD_COLOR),
                    cv2.COLOR_BGR2RGB,
                )

                # State: absolute EEF with normalized gripper
                eef_state = build_eef_state(eef_raw[frame_idx])

                # Action: delta EEF with normalized gripper
                if sampled_pos < len(sampled_indices) - 1:
                    next_frame_idx = sampled_indices[sampled_pos + 1]
                    action = compute_delta_action(eef_raw[frame_idx], eef_raw[next_frame_idx])
                else:
                    # Last frame: zero delta, keep current gripper
                    action = np.zeros(14, dtype=np.float32)
                    action[6] = normalize_gripper(eef_raw[frame_idx][6])
                    action[13] = normalize_gripper(eef_raw[frame_idx][13])

                dataset.add_frame(
                    {
                        "observation.images.head": head_img,
                        "observation.images.left_wrist": left_wrist_img,
                        "observation.images.right_wrist": right_wrist_img,
                        "observation.state": eef_state,
                        "action": action,
                        "task": task_description,
                    }
                )

            dataset.save_episode()
            frames_used = len(sampled_indices)
            total_frames += frames_used
            print(
                f"  Saved episode with {frames_used} frames "
                f"(skipped {start_frame}, stride {sample_stride})"
            )

    print(f"\nDataset saved to {output_path}")
    print(f"Total episodes: {len(episode_files)}, total frames: {total_frames}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["lift2", "bimanual", "eef", "delta"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)

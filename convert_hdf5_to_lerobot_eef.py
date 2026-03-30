#convert_hdf5_to_lerobot_eef.py

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

Convert 60Hz recordings to a 120Hz LeRobot dataset (upsample):
uv run convert_hdf5_to_lerobot_eef.py \
    --data-dir ./dataset0319 \
    --repo-id 0319_pick_and_place_block_120hz \
    --task-description "Put the block on the plate." \
    --source-fps 60 \
    --fps 120

Convert 60Hz recordings to a 30Hz LeRobot dataset (downsample):
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


def interpolate_eef(eef_a: np.ndarray, eef_b: np.ndarray, alpha: float) -> np.ndarray:
    """
    线性插值两帧之间的 EEF 状态（原始值，未归一化）。

    Args:
        eef_a: (14,) 起始帧原始 EEF
        eef_b: (14,) 结束帧原始 EEF
        alpha: 插值系数，0.0 = eef_a，1.0 = eef_b

    Returns:
        interpolated: (14,) 插值后的原始 EEF
    """
    return ((1.0 - alpha) * eef_a + alpha * eef_b).astype(np.float32)


def interpolate_image(img_a: np.ndarray, img_b: np.ndarray, alpha: float) -> np.ndarray:
    """
    线性混合两帧图像（加权平均）。

    Args:
        img_a: (H, W, 3) uint8 起始帧 RGB 图像
        img_b: (H, W, 3) uint8 结束帧 RGB 图像
        alpha: 插值系数，0.0 = img_a，1.0 = img_b

    Returns:
        blended: (H, W, 3) uint8 混合图像
    """
    return cv2.addWeighted(img_a, 1.0 - alpha, img_b, alpha, 0)


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
    left_xyz = eef_data[:, :3]
    right_xyz = eef_data[:, 7:10]

    left_deltas = np.diff(left_xyz, axis=0)
    right_deltas = np.diff(right_xyz, axis=0)

    left_motion = np.linalg.norm(left_deltas, axis=1)
    right_motion = np.linalg.norm(right_deltas, axis=1)

    motion_magnitude = np.maximum(left_motion, right_motion)

    motion_frames = np.where(motion_magnitude > threshold)[0]

    if len(motion_frames) == 0:
        return 0

    return motion_frames[0]


def build_interpolated_frames(
    eef_raw: np.ndarray,
    source_indices: list[int],
    interp_factor: int,
) -> list[tuple[int, int, float]]:
    """
    生成上采样后的帧列表，每个元素为 (frame_a_idx, frame_b_idx, alpha)。

    对于最后一帧，alpha=0（直接使用最后一帧，无后继帧可插值）。

    Args:
        eef_raw: [T, 14] 原始 EEF 数据（用于确定边界）
        source_indices: 原始帧索引列表（降采样后的）
        interp_factor: 上采样倍率（target_fps / source_fps）

    Returns:
        frames: [(idx_a, idx_b, alpha), ...]
    """
    frames = []
    for i, idx_a in enumerate(source_indices):
        if i < len(source_indices) - 1:
            idx_b = source_indices[i + 1]
            # 在两帧之间生成 interp_factor 个子帧
            for sub in range(interp_factor):
                alpha = sub / interp_factor  # 0, 1/N, 2/N, ...
                frames.append((idx_a, idx_b, alpha))
        else:
            # 最后一帧：无后继帧，直接输出一次
            frames.append((idx_a, idx_a, 0.0))
    return frames


def main(
    data_dir: str,
    repo_id: str,
    task_description: str,
    *,
    push_to_hub: bool = False,
    fps: int = 120,
    source_fps: int | None = None,
    skip_static_start: bool = True,
    motion_threshold: float = 0.001,
):
    """
    Convert HDF5 dataset to LeRobot format.

    支持上采样（如 60Hz -> 120Hz）和下采样（如 60Hz -> 30Hz）。

    Args:
        data_dir: Directory containing episode_*.hdf5 files
        repo_id: Dataset repository ID (e.g., "mytask_eef_120hz")
        task_description: Task description string (e.g., "pick and place")
        push_to_hub: Whether to push to HuggingFace Hub
        fps: Output LeRobot dataset frame rate (default: 120)
        source_fps: Source HDF5 recording frame rate.
            - 若 fps > source_fps：上采样（线性插值）
            - 若 fps < source_fps：下采样（整数步长抽帧）
            - 若 fps == source_fps：直接转换
        skip_static_start: 是否跳过开头静止帧
        motion_threshold: 运动检测阈值（米）
    """
    data_path = Path(data_dir)

    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    effective_source_fps = source_fps if source_fps is not None else fps
    if effective_source_fps <= 0:
        raise ValueError(f"source_fps must be positive, got {effective_source_fps}")

    # -----------------------------------------------------------------------
    # 判断模式：上采样 / 下采样 / 等频
    # -----------------------------------------------------------------------
    if fps > effective_source_fps:
        # 上采样：fps 必须是 source_fps 的整数倍
        if fps % effective_source_fps != 0:
            raise ValueError(
                f"For upsampling, fps ({fps}) must be an integer multiple of "
                f"source_fps ({effective_source_fps})"
            )
        mode = "upsample"
        interp_factor = fps // effective_source_fps   # e.g. 120/60 = 2
        sample_stride = 1                              # 原始帧不跳过
    elif fps < effective_source_fps:
        # 下采样：source_fps 必须是 fps 的整数倍
        if effective_source_fps % fps != 0:
            raise ValueError(
                f"For downsampling, source_fps ({effective_source_fps}) must be an "
                f"integer multiple of fps ({fps})"
            )
        mode = "downsample"
        interp_factor = 1
        sample_stride = effective_source_fps // fps   # e.g. 60/30 = 2
    else:
        mode = "passthrough"
        interp_factor = 1
        sample_stride = 1

    print(f"Mode: {mode} | source_fps={effective_source_fps} -> output_fps={fps}")
    if mode == "upsample":
        print(f"  Interpolation factor: {interp_factor}x (linear interpolation)")
    elif mode == "downsample":
        print(f"  Sample stride: {sample_stride}")

    # -----------------------------------------------------------------------
    # 初始化 LeRobot 数据集
    # -----------------------------------------------------------------------
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

    total_frames = 0

    for episode_file in episode_files:
        print(f"Processing {episode_file.name}...")

        with h5py.File(episode_file, 'r') as f:
            eef_raw = f['observations/eef'][:]  # [T, 14]
            num_frames = eef_raw.shape[0]

            # 跳过静止开头
            start_frame = 0
            if skip_static_start:
                start_frame = detect_motion_start(eef_raw, threshold=motion_threshold)
                if start_frame > 0:
                    print(
                        f"  Skipping first {start_frame} static frames "
                        f"(motion threshold: {motion_threshold * 1000:.1f}mm)"
                    )

            # 原始帧索引（下采样 / 等频 / 上采样 均先生成等间隔的原始帧列表）
            source_indices = list(range(start_frame, num_frames, sample_stride))
            if not source_indices:
                print("  No frames remain after sampling, skipping episode")
                continue

            # ------------------------------------------------------------------
            # 生成最终帧列表
            # ------------------------------------------------------------------
            if mode in ("downsample", "passthrough"):
                # 每个原始索引直接对应一个输出帧，alpha=0
                frame_tuples = [(idx, idx, 0.0) for idx in source_indices]
            else:
                # 上采样：在相邻原始帧间线性插值
                frame_tuples = build_interpolated_frames(
                    eef_raw, source_indices, interp_factor
                )

            # ------------------------------------------------------------------
            # 逐帧写入数据集
            # ------------------------------------------------------------------
            # 预解码所有需要的原始图像帧（避免重复解码）
            # 上采样时相邻两帧都需要，缓存上一帧图像减少 IO
            prev_raw_idx = None
            prev_imgs = None  # (head, left_wrist, right_wrist)

            for out_pos, (idx_a, idx_b, alpha) in enumerate(frame_tuples):

                # --- 解码图像 ---
                def decode_rgb(ds_key: str, idx: int) -> np.ndarray:
                    return cv2.cvtColor(
                        cv2.imdecode(f[ds_key][idx], cv2.IMREAD_COLOR),
                        cv2.COLOR_BGR2RGB,
                    )

                if alpha == 0.0 or idx_a == idx_b:
                    # 直接用 idx_a（无需插值）
                    head_img       = decode_rgb('observations/images/head', idx_a)
                    left_wrist_img = decode_rgb('observations/images/left_wrist', idx_a)
                    right_wrist_img= decode_rgb('observations/images/right_wrist', idx_a)
                else:
                    # 需要 idx_a 和 idx_b 两帧图像
                    # 利用缓存：如果上一次的 idx_a 和这次相同，复用
                    if prev_raw_idx == idx_a and prev_imgs is not None:
                        head_a, left_a, right_a = prev_imgs
                    else:
                        head_a       = decode_rgb('observations/images/head', idx_a)
                        left_a       = decode_rgb('observations/images/left_wrist', idx_a)
                        right_a      = decode_rgb('observations/images/right_wrist', idx_a)

                    head_b       = decode_rgb('observations/images/head', idx_b)
                    left_b       = decode_rgb('observations/images/left_wrist', idx_b)
                    right_b      = decode_rgb('observations/images/right_wrist', idx_b)

                    head_img       = interpolate_image(head_a,  head_b,  alpha)
                    left_wrist_img = interpolate_image(left_a,  left_b,  alpha)
                    right_wrist_img= interpolate_image(right_a, right_b, alpha)

                    # 缓存本次 idx_a 对应的图像，供下一轮复用
                    prev_raw_idx = idx_a
                    prev_imgs    = (head_a, left_a, right_a)

                # --- EEF 状态（插值后归一化）---
                if alpha == 0.0 or idx_a == idx_b:
                    eef_interp = eef_raw[idx_a]
                else:
                    eef_interp = interpolate_eef(eef_raw[idx_a], eef_raw[idx_b], alpha)

                eef_state = build_eef_state(eef_interp)

                # --- Action（当前插值帧 -> 下一插值帧的 delta）---
                if out_pos < len(frame_tuples) - 1:
                    next_idx_a, next_idx_b, next_alpha = frame_tuples[out_pos + 1]
                    if next_alpha == 0.0 or next_idx_a == next_idx_b:
                        next_eef_interp = eef_raw[next_idx_a]
                    else:
                        next_eef_interp = interpolate_eef(
                            eef_raw[next_idx_a], eef_raw[next_idx_b], next_alpha
                        )
                    action = compute_delta_action(eef_interp, next_eef_interp)
                else:
                    # 最后一帧：零 delta，保持当前夹爪
                    action = np.zeros(14, dtype=np.float32)
                    action[6]  = normalize_gripper(eef_interp[6])
                    action[13] = normalize_gripper(eef_interp[13])

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
            frames_used = len(frame_tuples)
            total_frames += frames_used
            print(
                f"  Saved episode with {frames_used} frames "
                f"(source frames used: {len(source_indices)}, "
                f"interp_factor: {interp_factor})"
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

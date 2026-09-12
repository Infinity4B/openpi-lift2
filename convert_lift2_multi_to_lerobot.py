"""Convert multiple LIFT2 HDF5 directories into one LeRobot dataset.

Each ``episode_*.hdf5`` file is written as one episode in the output dataset.
The conversion logic intentionally matches ``convert_hdf5_to_lerobot_eef.py``:
EEF states are kept as absolute values, xyz/rpy actions are converted to deltas,
grippers are normalized to [0, 1], and images/states can optionally be resampled.

The default conversion is 60 Hz -> 30 Hz, matching the reference LIFT2 converter:

    uv run convert_lift2_multi_to_lerobot.py

The defaults read the four LIFT2 task directories and write:
    ~/.cache/huggingface/lerobot/lerobot_lift2_df_image

To override a path or conversion option, for example:
    uv run convert_lift2_multi_to_lerobot.py \
        --fps 30 \
        --source-fps 60 \
        --overwrite
"""

from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import cast

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
import tyro

from convert_hdf5_to_lerobot_eef import build_eef_state
from convert_hdf5_to_lerobot_eef import build_interpolated_frames
from convert_hdf5_to_lerobot_eef import compute_delta_action
from convert_hdf5_to_lerobot_eef import detect_motion_start
from convert_hdf5_to_lerobot_eef import interpolate_eef
from convert_hdf5_to_lerobot_eef import interpolate_image
from convert_hdf5_to_lerobot_eef import normalize_gripper

EEF_DIM = 14
IMAGE_SHAPE = (480, 640, 3)
IMAGE_KEYS = (
    "observations/images/head",
    "observations/images/left_wrist",
    "observations/images/right_wrist",
)


@dataclass(frozen=True)
class TaskSource:
    """One HDF5 directory and the task prompt assigned to its episodes."""

    name: str
    data_dir: Path
    task_description: str


def create_empty_dataset(repo_id: str, fps: int) -> LeRobotDataset:
    """Create the combined LeRobot dataset with the same schema as the existing converter."""
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        raise FileExistsError(
            f"Output dataset already exists: {output_path}. Use --overwrite to remove it before conversion."
        )

    return LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="bimanual",
        fps=fps,
        features={
            "observation.images.head": {
                "dtype": "image",
                "shape": IMAGE_SHAPE,
                "names": ["height", "width", "channel"],
            },
            "observation.images.left_wrist": {
                "dtype": "image",
                "shape": IMAGE_SHAPE,
                "names": ["height", "width", "channel"],
            },
            "observation.images.right_wrist": {
                "dtype": "image",
                "shape": IMAGE_SHAPE,
                "names": ["height", "width", "channel"],
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (EEF_DIM,),
                "names": ["eef_state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (EEF_DIM,),
                "names": ["eef_delta"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )


def find_episode_files(source: TaskSource) -> list[Path]:
    """Find the HDF5 episodes for one task source."""
    if not source.data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {source.data_dir}")

    episode_files = sorted(source.data_dir.glob("episode_*.hdf5"))
    if not episode_files:
        raise FileNotFoundError(f"No episode_*.hdf5 files found directly under {source.data_dir}")
    return episode_files


def validate_episode_file(episode_file: Path) -> None:
    """Validate the fields required by the LIFT2 EEF converter."""
    required_paths = ("observations/eef", *IMAGE_KEYS)
    with h5py.File(episode_file, "r") as file:
        missing_paths = [path for path in required_paths if path not in file]
        if missing_paths:
            raise ValueError(f"{episode_file} is missing required dataset(s): {missing_paths}")

        eef_dataset = cast(h5py.Dataset, file["observations/eef"])
        eef_shape = eef_dataset.shape
        if len(eef_shape) != 2 or eef_shape[1] != EEF_DIM:
            raise ValueError(f"{episode_file}: observations/eef must have shape [T, {EEF_DIM}], got {eef_shape}")

        frame_count = eef_shape[0]
        for image_path in IMAGE_KEYS:
            image_dataset = cast(h5py.Dataset, file[image_path])
            image_shape = image_dataset.shape
            if not image_shape or image_shape[0] != frame_count:
                raise ValueError(
                    f"{episode_file}: {image_path} must contain {frame_count} frames, got shape {image_shape}"
                )


def process_episode(
    dataset: LeRobotDataset,
    episode_file: Path,
    task_description: str,
    *,
    mode: str,
    interp_factor: int,
    sample_stride: int,
    skip_static_start: bool,
    motion_threshold: float,
) -> int:
    """Convert one HDF5 episode and append it to the combined dataset."""
    with h5py.File(episode_file, "r") as file:
        eef_dataset = cast(h5py.Dataset, file["observations/eef"])
        eef_raw = np.asarray(eef_dataset[:])
        num_frames = eef_raw.shape[0]

        start_frame = 0
        if skip_static_start:
            start_frame = detect_motion_start(
                eef_raw,
                threshold=motion_threshold,
            )
            if start_frame > 0:
                print(
                    f"  Skipping first {start_frame} static frames (motion threshold: {motion_threshold * 1000:.1f}mm)"
                )

        source_indices = list(range(start_frame, num_frames, sample_stride))
        if not source_indices:
            print("  No frames remain after sampling, skipping episode")
            return 0

        if mode in ("downsample", "passthrough"):
            frame_tuples = [(index, index, 0.0) for index in source_indices]
        else:
            frame_tuples = build_interpolated_frames(
                eef_raw,
                source_indices,
                interp_factor,
            )

        previous_raw_index = None
        previous_images = None

        for output_position, (index_a, index_b, alpha) in enumerate(frame_tuples):

            def decode_rgb(dataset_path: str, index: int) -> np.ndarray:
                image_dataset = cast(h5py.Dataset, file[dataset_path])
                image = cv2.imdecode(
                    np.asarray(image_dataset[index]),
                    cv2.IMREAD_COLOR,
                )
                if image is None:
                    raise ValueError(f"Failed to decode image {dataset_path}[{index}] in {episode_file}")
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            if alpha == 0.0 or index_a == index_b:
                head_image = decode_rgb("observations/images/head", index_a)
                left_wrist_image = decode_rgb(
                    "observations/images/left_wrist",
                    index_a,
                )
                right_wrist_image = decode_rgb(
                    "observations/images/right_wrist",
                    index_a,
                )
            else:
                if previous_raw_index == index_a and previous_images is not None:
                    head_a, left_a, right_a = previous_images
                else:
                    head_a = decode_rgb("observations/images/head", index_a)
                    left_a = decode_rgb("observations/images/left_wrist", index_a)
                    right_a = decode_rgb("observations/images/right_wrist", index_a)

                head_b = decode_rgb("observations/images/head", index_b)
                left_b = decode_rgb("observations/images/left_wrist", index_b)
                right_b = decode_rgb("observations/images/right_wrist", index_b)

                head_image = interpolate_image(head_a, head_b, alpha)
                left_wrist_image = interpolate_image(left_a, left_b, alpha)
                right_wrist_image = interpolate_image(right_a, right_b, alpha)

                previous_raw_index = index_a
                previous_images = (head_a, left_a, right_a)

            if alpha == 0.0 or index_a == index_b:
                eef_interpolated = eef_raw[index_a]
            else:
                eef_interpolated = interpolate_eef(
                    eef_raw[index_a],
                    eef_raw[index_b],
                    alpha,
                )

            eef_state = build_eef_state(eef_interpolated)

            if output_position < len(frame_tuples) - 1:
                next_index_a, next_index_b, next_alpha = frame_tuples[output_position + 1]
                if next_alpha == 0.0 or next_index_a == next_index_b:
                    next_eef_interpolated = eef_raw[next_index_a]
                else:
                    next_eef_interpolated = interpolate_eef(
                        eef_raw[next_index_a],
                        eef_raw[next_index_b],
                        next_alpha,
                    )
                action = compute_delta_action(
                    eef_interpolated,
                    next_eef_interpolated,
                )
            else:
                action = np.zeros(EEF_DIM, dtype=np.float32)
                action[6] = normalize_gripper(eef_interpolated[6])
                action[13] = normalize_gripper(eef_interpolated[13])

            dataset.add_frame(
                {
                    "observation.images.head": head_image,
                    "observation.images.left_wrist": left_wrist_image,
                    "observation.images.right_wrist": right_wrist_image,
                    "observation.state": eef_state,
                    "action": action,
                    "task": task_description,
                }
            )

        dataset.save_episode()
        return len(frame_tuples)


def main(
    moving_cube_data_dir: str = "/soft/wangxi/lift2_realdata/moving_cube",
    moving_cube_task_description: str = "Grasp the moving cube and place it on the plate.",
    light_data_dir: str = "/soft/wangxi/lift2_realdata/light",
    light_task_description: str = (
        "Identify and pick up the illuminated red light from the rotating turntable, then place it aside."
    ),
    size_data_dir: str = "/soft/wangxi/lift2_realdata/size",
    size_task_description: str = (
        "Pick up the four randomly placed cylinders and insert each one into the matching hole according to its size."
    ),
    tube_data_dir: str = "/soft/wangxi/lift2_realdata/tube",
    tube_task_description: str = "Transfer the test tube from the right rack to the left rack.",
    repo_id: str = "lerobot_lift2_df_image",
    *,
    push_to_hub: bool = False,
    fps: int = 30,
    source_fps: int = 60,
    skip_static_start: bool = True,
    motion_threshold: float = 0.001,
    overwrite: bool = False,
) -> None:
    """Convert four LIFT2 HDF5 directories into one LeRobot dataset.

    Args:
        moving_cube_data_dir: Directory containing moving-cube episodes.
        moving_cube_task_description: Prompt assigned to moving-cube episodes.
        light_data_dir: Directory containing light-task episodes.
        light_task_description: Prompt assigned to light-task episodes.
        size_data_dir: Directory containing size-sorting episodes.
        size_task_description: Prompt assigned to size-sorting episodes.
        tube_data_dir: Directory containing test-tube episodes.
        tube_task_description: Prompt assigned to test-tube episodes.
        repo_id: Output LeRobot repository ID.
        push_to_hub: Push the result to the Hugging Face Hub after conversion.
        fps: Output dataset frame rate.
        source_fps: Source HDF5 frame rate. Defaults to 60 Hz.
        skip_static_start: Remove static frames at the beginning of each episode.
        motion_threshold: Motion threshold in meters for static-frame removal.
        overwrite: Remove an existing output dataset before conversion.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")

    effective_source_fps = source_fps
    if effective_source_fps <= 0:
        raise ValueError(f"source_fps must be positive, got {effective_source_fps}")

    if fps > effective_source_fps:
        if fps % effective_source_fps != 0:
            raise ValueError(
                f"For upsampling, fps ({fps}) must be an integer multiple of source_fps ({effective_source_fps})"
            )
        mode = "upsample"
        interp_factor = fps // effective_source_fps
        sample_stride = 1
    elif fps < effective_source_fps:
        if effective_source_fps % fps != 0:
            raise ValueError(
                f"For downsampling, source_fps ({effective_source_fps}) must be an integer multiple of fps ({fps})"
            )
        mode = "downsample"
        interp_factor = 1
        sample_stride = effective_source_fps // fps
    else:
        mode = "passthrough"
        interp_factor = 1
        sample_stride = 1

    sources = (
        TaskSource(
            name="moving_cube",
            data_dir=Path(moving_cube_data_dir),
            task_description=moving_cube_task_description,
        ),
        TaskSource(
            name="light",
            data_dir=Path(light_data_dir),
            task_description=light_task_description,
        ),
        TaskSource(
            name="size",
            data_dir=Path(size_data_dir),
            task_description=size_task_description,
        ),
        TaskSource(
            name="tube",
            data_dir=Path(tube_data_dir),
            task_description=tube_task_description,
        ),
    )

    source_files = {source.name: find_episode_files(source) for source in sources}
    all_episode_files = [episode_file for files in source_files.values() for episode_file in files]
    print(f"Mode: {mode} | source_fps={effective_source_fps} -> output_fps={fps}")
    if mode == "upsample":
        print(f"  Interpolation factor: {interp_factor}x (linear interpolation)")
    elif mode == "downsample":
        print(f"  Sample stride: {sample_stride}")
    print(f"Found {len(all_episode_files)} episodes across {len(sources)} task datasets")

    for episode_file in all_episode_files:
        validate_episode_file(episode_file)

    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output dataset already exists: {output_path}. Use --overwrite to remove it before conversion."
            )
        shutil.rmtree(output_path)

    dataset = create_empty_dataset(repo_id, fps)
    total_frames = 0
    total_episodes = 0

    for source in sources:
        episode_files = source_files[source.name]
        print(f"\nProcessing {source.name}: {len(episode_files)} episodes with task: {source.task_description}")
        source_frames = 0
        for episode_file in episode_files:
            print(f"Processing {episode_file.name}...")
            frames_used = process_episode(
                dataset,
                episode_file,
                source.task_description,
                mode=mode,
                interp_factor=interp_factor,
                sample_stride=sample_stride,
                skip_static_start=skip_static_start,
                motion_threshold=motion_threshold,
            )
            source_frames += frames_used
            total_frames += frames_used
            total_episodes += int(frames_used > 0)
            print(f"  Saved episode with {frames_used} frames")
        print(f"Finished {source.name}: {source_frames} frames")

    print(f"\nDataset saved to {output_path}")
    print(f"Total episodes: {total_episodes}, total frames: {total_frames}")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["lift2", "bimanual", "eef", "delta", "multi-task"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)

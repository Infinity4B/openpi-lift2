#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert saved LIFT2 camera frames to mp4 videos."""

import argparse
from pathlib import Path

import cv2


CAMERA_VIDEO_NAMES = {
    "camera_h": "camera_h.mp4",
    "camera_l": "camera_l.mp4",
    "camera_r": "camera_r.mp4",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_frame_paths(frame_dir):
    return sorted(
        path for path in frame_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def write_video_from_frames(frame_dir, output_path, fps):
    frame_paths = list_frame_paths(frame_dir)
    if not frame_paths:
        raise ValueError(f"No image frames found in {frame_dir}")

    first_frame = cv2.imread(str(frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read first frame: {frame_paths[0]}")

    height, width = first_frame.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {output_path}")

    try:
        for frame_path in frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Failed to read frame: {frame_path}")
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Frame size mismatch: {frame_path} has {frame.shape[1]}x{frame.shape[0]}, "
                    f"expected {width}x{height}"
                )
            writer.write(frame)
    finally:
        writer.release()

    print(f"Saved {len(frame_paths)} frames -> {output_path} @ {fps:g} FPS")


def find_lift2_camera_dirs(input_dir):
    return {
        camera_dir_name: input_dir / camera_dir_name
        for camera_dir_name in CAMERA_VIDEO_NAMES
        if (input_dir / camera_dir_name).is_dir()
    }


def resolve_single_output(input_dir, output):
    if output is None:
        return input_dir.with_suffix(".mp4")
    if output.suffix.lower() == ".mp4":
        return output
    return output / f"{input_dir.name}.mp4"


def convert(input_dir, output, fps):
    input_dir = input_dir.expanduser().resolve()
    output = output.expanduser().resolve() if output else None

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    camera_dirs = find_lift2_camera_dirs(input_dir)
    if camera_dirs:
        output_dir = output or input_dir.with_name(f"{input_dir.name}_videos")
        for camera_dir_name, camera_dir in camera_dirs.items():
            write_video_from_frames(camera_dir, output_dir / CAMERA_VIDEO_NAMES[camera_dir_name], fps)
        return

    write_video_from_frames(input_dir, resolve_single_output(input_dir, output), fps)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Convert a single image folder, or a LIFT2 recording folder containing "
            "camera_h/camera_l/camera_r subfolders, into mp4 video(s)."
        )
    )
    parser.add_argument("input_dir", type=Path, help="Image folder or LIFT2 pic/{task}/{seq} folder")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output mp4 path or output directory")
    parser.add_argument("--fps", type=float, default=60.0, help="Video FPS (default: 60)")
    args = parser.parse_args()

    convert(args.input_dir, args.output, args.fps)


if __name__ == "__main__":
    main()

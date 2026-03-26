"""
Offline evaluation: replay training dataset frames through the policy server
and compare predicted actions with ground truth.

Usage:
    uv run test_dataset_eval.py \
        --host localhost --port 7777 \
        --data-dir ./dataset0319 \
        --prompt "perform task" \
        --episode 0 \
        --start-frame 0 --num-frames 50 \
        --step 1
"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np

from openpi_client import image_tools, websocket_client_policy

# Gripper normalization (matching convert_hdf5_to_lerobot_eef.py)
GRIPPER_MAX = 5.0


def normalize_gripper(raw_value):
    return np.clip(raw_value, 0.0, GRIPPER_MAX) / GRIPPER_MAX


def build_eef_state(eef_raw):
    state = eef_raw.copy().astype(np.float32)
    state[6] = normalize_gripper(eef_raw[6])
    state[13] = normalize_gripper(eef_raw[13])
    return state


def compute_gt_action(current_eef_raw, next_eef_raw):
    action = np.zeros(14, dtype=np.float32)
    action[0:3] = next_eef_raw[0:3] - current_eef_raw[0:3]
    action[3:6] = next_eef_raw[3:6] - current_eef_raw[3:6]
    action[6] = normalize_gripper(next_eef_raw[6])
    action[7:10] = next_eef_raw[7:10] - current_eef_raw[7:10]
    action[10:13] = next_eef_raw[10:13] - current_eef_raw[10:13]
    action[13] = normalize_gripper(next_eef_raw[13])
    return action


def decode_image(jpeg_bytes):
    img = cv2.imdecode(jpeg_bytes, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def build_observation(f, frame_idx, eef_raw, prompt):
    head_img = decode_image(f["observations/images/head"][frame_idx])
    left_img = decode_image(f["observations/images/left_wrist"][frame_idx])
    right_img = decode_image(f["observations/images/right_wrist"][frame_idx])

    state = build_eef_state(eef_raw[frame_idx])

    return {
        "observation.images.head": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(head_img, 224, 224)
        ),
        "observation.images.left_wrist": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(left_img, 224, 224)
        ),
        "observation.images.right_wrist": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(right_img, 224, 224)
        ),
        "observation.state": state,
        "prompt": prompt,
    }


def print_action_comparison(frame_idx, pred, gt, labels):
    """Print side-by-side comparison of predicted vs ground truth."""
    print(f"\n--- Frame {frame_idx} ---")
    print(f"  {'Component':<20s} {'Predicted':>12s} {'GroundTruth':>12s} {'Error':>12s}")
    for i, label in enumerate(labels):
        err = pred[i] - gt[i]
        print(f"  {label:<20s} {pred[i]:>12.6f} {gt[i]:>12.6f} {err:>12.6f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate policy on training dataset")
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--port", type=int, default=7777)
    parser.add_argument("--data-dir", type=str, default="./dataset0319")
    parser.add_argument("--prompt", type=str, default="perform task")
    parser.add_argument("--episode", type=int, default=0, help="Episode index")
    parser.add_argument("--start-frame", type=int, default=0, help="Start frame index")
    parser.add_argument("--num-frames", type=int, default=20, help="Number of frames to evaluate (0=all)")
    parser.add_argument("--step", type=int, default=1, help="Frame step interval")
    parser.add_argument("--action-index", type=int, default=0,
                        help="Which action in the chunk to compare (0=first)")
    parser.add_argument("--verbose", action="store_true", help="Print per-frame details")
    args = parser.parse_args()

    ACTION_LABELS = [
        "L_dx", "L_dy", "L_dz", "L_droll", "L_dpitch", "L_dyaw", "L_gripper",
        "R_dx", "R_dy", "R_dz", "R_droll", "R_dpitch", "R_dyaw", "R_gripper",
    ]

    episode_file = Path(args.data_dir) / f"episode_{args.episode}.hdf5"
    if not episode_file.exists():
        print(f"Episode file not found: {episode_file}")
        return

    print(f"Connecting to policy server at {args.host}:{args.port}")
    client = websocket_client_policy.WebsocketClientPolicy(host=args.host, port=args.port)

    with h5py.File(episode_file, "r") as f:
        eef_raw = f["observations/eef"][:]
        num_frames = eef_raw.shape[0]
        print(f"Episode {args.episode}: {num_frames} frames")

        # Determine frame range
        start = args.start_frame
        end = num_frames - 1  # need next frame for GT
        if args.num_frames > 0:
            end = min(start + args.num_frames * args.step, end)

        frames = list(range(start, end, args.step))
        print(f"Evaluating {len(frames)} frames [{start}:{end}:{args.step}], "
              f"comparing action_chunk[{args.action_index}] with GT\n")

        all_errors = []

        for frame_idx in frames:
            # Ground truth action (first step: current -> next)
            gt_frame = frame_idx + args.action_index
            if gt_frame >= num_frames - 1:
                break
            gt_action = compute_gt_action(eef_raw[gt_frame], eef_raw[gt_frame + 1])

            # Predict
            obs = build_observation(f, frame_idx, eef_raw, args.prompt)
            result = client.infer(obs)
            pred_chunk = result["actions"]  # (action_horizon, 14)
            pred_action = pred_chunk[args.action_index]

            error = pred_action - gt_action
            all_errors.append(error)

            if args.verbose:
                print_action_comparison(frame_idx, pred_action, gt_action, ACTION_LABELS)

            # Brief progress
            abs_err = np.abs(error)
            print(f"Frame {frame_idx:4d}  "
                  f"L_xyz_mae={abs_err[0:3].mean():.6f}  "
                  f"R_xyz_mae={abs_err[7:10].mean():.6f}  "
                  f"L_grip_err={abs_err[6]:.4f}  "
                  f"R_grip_err={abs_err[13]:.4f}")

        if not all_errors:
            print("No frames evaluated.")
            return

        # Summary statistics
        errors = np.array(all_errors)  # (N, 14)
        abs_errors = np.abs(errors)
        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        print(f"Frames evaluated: {len(errors)}")
        print(f"\n{'Component':<20s} {'MAE':>10s} {'Std':>10s} {'Max':>10s}")
        print("-" * 50)
        for i, label in enumerate(ACTION_LABELS):
            mae = abs_errors[:, i].mean()
            std = abs_errors[:, i].std()
            mx = abs_errors[:, i].max()
            print(f"{label:<20s} {mae:>10.6f} {std:>10.6f} {mx:>10.6f}")

        # Grouped summaries
        print(f"\n{'Group':<20s} {'MAE':>10s}")
        print("-" * 30)
        print(f"{'Left XYZ':<20s} {abs_errors[:, 0:3].mean():>10.6f}")
        print(f"{'Left RPY':<20s} {abs_errors[:, 3:6].mean():>10.6f}")
        print(f"{'Left Gripper':<20s} {abs_errors[:, 6].mean():>10.6f}")
        print(f"{'Right XYZ':<20s} {abs_errors[:, 7:10].mean():>10.6f}")
        print(f"{'Right RPY':<20s} {abs_errors[:, 10:13].mean():>10.6f}")
        print(f"{'Right Gripper':<20s} {abs_errors[:, 13].mean():>10.6f}")
        print(f"{'Overall':<20s} {abs_errors.mean():>10.6f}")


if __name__ == "__main__":
    main()

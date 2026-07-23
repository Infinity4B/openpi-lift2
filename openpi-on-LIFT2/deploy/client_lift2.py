#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenPI LIFT2 Client for ARX R5 Dual-Arm Robot
End-effector delta control (delta_xyz + delta_rpy + gripper) with OpenPI remote inference
"""

import numpy as np
import time
import argparse
import collections
import json
import math
import os
import shutil
import socket
import subprocess
import sys
import statistics
import uuid
from pathlib import Path

import rospy

# Add parent directory to path to find deploy.utils
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
# Prefer the self-contained copy under openpi-on-LIFT2 over any globally installed
# or workspace-level openpi_client package on the robot.
sys.path.insert(0, parent_dir)

from openpi_client import image_tools
from openpi_client import websocket_client_policy

from deploy.utils.rotation import pose_to_eef, apply_eef_delta, denormalize_gripper
from deploy.utils.rosoperator import RosOperator
from deploy.utils.eef_action_executor import EEFInterpolatingExecutor, EEFTrajectoryExecutor

CAMERA_NAMES = ['head', 'left_wrist', 'right_wrist']

DEFAULT_LANGUAGE_INSTRUCTION = 'perform task'
DEFAULT_MAX_PUBLISH_STEP = 1000
DEFAULT_LAUNCH_CONFIG = Path(parent_dir) / 'launch_profiles.yaml'
DEFAULT_RTC_COMPARE_OUTPUT_DIR = Path(parent_dir) / 'rtc_real_compare'
DEFAULT_MAX_DELTA_XYZ = 0.05
DEFAULT_MAX_DELTA_RPY = 0.2
RTC_COMPARE_CAMERA_DIR_NAMES = {
    'head': 'camera_h',
    'left_wrist': 'camera_l',
    'right_wrist': 'camera_r',
}
LEGACY_RUNTIME_DEFAULTS = {
    'host': '192.168.101.101',
    'port': 7777,
    'publish_rate': 30,
    'execute_horizon': 30,
    'action_chunk_size': 30,
    'max_publish_step': DEFAULT_MAX_PUBLISH_STEP,
}
PROFILE_OPTIONAL_KEYS = (
    'client_mode',
    'rtc_action_horizon',
    'rtc_execution_horizon',
    'rtc_inference_delay',
    'rtc_control_period_s',
    'rtc_prefix_attention_schedule',
    'rtc_max_guidance_weight',
)
PROFILE_VALUE_CASTERS = {
    'host': str,
    'client_mode': str,
    'port': int,
    'publish_rate': int,
    'execute_horizon': int,
    'action_chunk_size': int,
    'rtc_action_horizon': int,
    'rtc_execution_horizon': int,
    'rtc_inference_delay': int,
    'rtc_control_period_s': float,
    'rtc_prefix_attention_schedule': str,
    'rtc_max_guidance_weight': float,
}
PROFILE_REQUIRED_KEYS = (
    'host',
    'port',
    'publish_rate',
    'execute_horizon',
    'action_chunk_size',
)
PRESET_TASK_INSTRUCTIONS = {
    'tube': 'Transfer the test tube from the right rack to the left rack.',
    'towel': 'Flatten the towel.',
    'wrench': 'Open the toolbox, check the items inside one by one, and find the wrench.',
    'power_strip': 'Move the power strip with the left arm, and press the button of the power strip with the right arm.',
    'drum': 'Pick up two small drumsticks and hit the small drum.',
    'dice': 'Roll the dice and move the small stand the specified number of squares based on the number rolled.',
    'stack': 'Stack the building blocks one by one with the larger ones at the bottom.',
    'size': 'Pick up the four randomly placed cylinders and insert each one into the matching hole according to its size.',
    'color': 'Pick up each colored cylinder placed in front of the base and insert it into the empty groove at the matching color position on the 4-by-4 board.',
    'cube': 'Put the block on the plate.',
    'light': 'Identify and pick up the illuminated red light from the rotating turntable, then place it aside.',
}


def resolve_language_instruction(args):
    if args.language_instruction is not None:
        return args.language_instruction
    if args.task:
        return PRESET_TASK_INSTRUCTIONS[args.task]
    return DEFAULT_LANGUAGE_INSTRUCTION


def load_launch_profile(config_path, profile_name):
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(f"PyYAML is required to read {config_path}: {exc}") from exc

    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}

    profiles = data.get('profiles')
    if not isinstance(profiles, dict):
        raise ValueError(f"Invalid config: missing 'profiles' mapping in {config_path}")

    profile = profiles.get(profile_name)
    if not isinstance(profile, dict):
        available = ', '.join(sorted(profiles)) or '<none>'
        raise ValueError(f"Unknown profile '{profile_name}'. Available profiles: {available}")

    missing = [key for key in PROFILE_REQUIRED_KEYS if key not in profile]
    if missing:
        raise ValueError(
            f"Profile '{profile_name}' is missing required keys: {', '.join(missing)}"
        )

    return dict(profile)


def apply_launch_profile(args):
    if not getattr(args, 'profile', None):
        return

    config_path = Path(args.config).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    profile = load_launch_profile(config_path, args.profile)
    args.profile_config_path = str(config_path)

    for key in PROFILE_REQUIRED_KEYS + PROFILE_OPTIONAL_KEYS:
        if key in profile and getattr(args, key, None) is None:
            value = profile[key]
            caster = PROFILE_VALUE_CASTERS.get(key, str)
            setattr(args, key, caster(value))

    if args.max_publish_step is None:
        args.max_publish_step = DEFAULT_MAX_PUBLISH_STEP


def finalize_runtime_args(args):
    for key, value in LEGACY_RUNTIME_DEFAULTS.items():
        if getattr(args, key, None) is None:
            setattr(args, key, value)

    if args.client_mode is None:
        args.client_mode = 'standard'
    args.client_mode = args.client_mode.lower()
    if args.client_mode not in ('standard', 'rtc'):
        raise ValueError(f"client_mode must be 'standard' or 'rtc', got {args.client_mode!r}")

    if not (0.0 < args.smooth_alpha <= 1.0):
        raise ValueError(
            "smooth_alpha must be in (0, 1], "
            f"got {args.smooth_alpha}"
        )

    if args.fast:
        args.enable_inference_executor = True
    if args.executor_strategy not in ('trajectory_buffer', 'legacy_interpolate'):
        raise ValueError(
            "executor_strategy must be 'trajectory_buffer' or 'legacy_interpolate', "
            f"got {args.executor_strategy!r}"
        )
    if args.executor_rate_hz <= 0:
        raise ValueError(f"executor_rate_hz must be positive, got {args.executor_rate_hz}")
    if args.executor_max_queue_size <= 0:
        raise ValueError(f"executor_max_queue_size must be positive, got {args.executor_max_queue_size}")
    if args.rtc_action_horizon is None:
        args.rtc_action_horizon = args.action_chunk_size
    if args.rtc_execution_horizon is None:
        args.rtc_execution_horizon = min(args.execute_horizon, args.rtc_action_horizon)
    if args.rtc_inference_delay is None:
        args.rtc_inference_delay = 0
    if args.rtc_control_period_s is None:
        args.rtc_control_period_s = 1.0 / float(args.publish_rate)
    if args.rtc_prefix_attention_schedule is None:
        args.rtc_prefix_attention_schedule = 'exp'
    if args.rtc_max_guidance_weight is None:
        args.rtc_max_guidance_weight = 10.0
    if args.rtc_committed_prefix is None:
        args.rtc_committed_prefix = min(
            args.rtc_execution_horizon,
            max(1, args.rtc_inference_delay + 1),
        )
    else:
        if args.rtc_committed_prefix <= 0:
            raise ValueError(f"rtc_committed_prefix must be positive, got {args.rtc_committed_prefix}")
        args.rtc_committed_prefix = min(args.rtc_committed_prefix, args.rtc_execution_horizon)
    args.language_instruction = resolve_language_instruction(args)
    return args


def format_launcher_summary(args):
    lines = [
        '=' * 50,
        'OpenPI LIFT2 Client Launcher',
        '=' * 50,
    ]

    if getattr(args, 'profile', None):
        lines.append(f"Config File: {getattr(args, 'profile_config_path', args.config)}")
        lines.append(f"Profile: {args.profile}")

    lines.extend([
        f"Policy Server: {args.host}:{args.port}",
        f"Client Mode: {args.client_mode}",
        f"Control Rate: {args.publish_rate} Hz",
        f"Execute Horizon: {args.execute_horizon} frames",
        f"Action Chunk Size: {args.action_chunk_size} frames",
        f"Max Steps: {'Infinite' if args.max_publish_step <= 0 else args.max_publish_step}",
        'Inference Executor: '
        + ('Enabled' if args.enable_inference_executor else 'Disabled')
        + f" (policy {args.publish_rate}Hz -> executor {args.executor_rate_hz:.1f}Hz)",
        f"Executor Strategy: {args.executor_strategy}",
    ])

    if args.client_mode == 'rtc':
        lines.extend([
            f"RTC Action Horizon: {args.rtc_action_horizon} frames",
            f"RTC Execution Horizon: {args.rtc_execution_horizon} frames",
            f"RTC Inference Delay: {args.rtc_inference_delay} frames",
            f"RTC Committed Prefix: {args.rtc_committed_prefix} frames",
            f"RTC Control Period: {args.rtc_control_period_s:.4f} s",
            f"RTC Prefix Attention: {args.rtc_prefix_attention_schedule}, weight={args.rtc_max_guidance_weight}",
        ])

    if args.task:
        lines.append(f"Task Preset: {args.task}")
    lines.append(f"Language Instruction: {args.language_instruction}")

    if args.record_video:
        lines.append(
            f"Video Recording: ON (tmp frames -> /tmp/openpi-lift2/{args.task or 'custom'}/<session_id>/frames, "
            f"videos -> ./video/{args.task or 'custom'}/<next_seq>)"
        )
    if args.rtc_compare_record:
        lines.append(
            "RTC Compare Recording: ON "
            f"(dir -> {args.rtc_compare_output_dir}/{get_rtc_compare_task_name(args)}, "
            f"suffix -> _{get_rtc_compare_mode_suffix(args.client_mode)})"
        )
    if args.debug:
        lines.append('Debug Mode: ON (press Enter each step)')

    lines.append('=' * 50)
    return '\n'.join(lines)


def check_server_connectivity(host, port, timeout_seconds=2.0):
    with socket.create_connection((host, port), timeout=timeout_seconds):
        return


def resolve_recording_output_dirs(args):
    task_name = args.task if args.task else 'custom'
    video_root = os.path.join(parent_dir, 'video', task_name)
    pic_root = os.path.join(parent_dir, 'pic', task_name)
    os.makedirs(video_root, exist_ok=True)
    os.makedirs(pic_root, exist_ok=True)

    next_seq = 1
    for root in (video_root, pic_root):
        for entry in os.listdir(root):
            if entry.isdigit():
                next_seq = max(next_seq, int(entry) + 1)

    video_output_dir = os.path.join(video_root, str(next_seq))
    pic_output_dir = os.path.join(pic_root, str(next_seq))
    return pic_output_dir, video_output_dir


def resolve_tmp_recording_session_dirs(args):
    task_name = args.task if args.task else 'custom'
    session_id = f"{time.strftime('%Y%m%dT%H%M%S')}_{uuid.uuid4().hex[:8]}"
    session_dir = os.path.join('/tmp/openpi-lift2', task_name, session_id)
    frames_dir = os.path.join(session_dir, 'frames')
    videos_dir = os.path.join(session_dir, 'videos')
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(videos_dir, exist_ok=True)
    args.recording_tmp_session_dir = session_dir
    args.recording_tmp_frames_dir = frames_dir
    args.recording_tmp_videos_dir = videos_dir
    rospy.loginfo(f"Recording tmp session: {session_dir}")
    return frames_dir, videos_dir


def setup_recording_output_dirs(args):
    args.record_video_final_pic_dir = None
    args.record_video_final_video_dir = None
    if args.record_video:
        pic_dir, video_dir = resolve_recording_output_dirs(args)
        args.record_video_final_pic_dir = pic_dir
        args.record_video_final_video_dir = video_dir
        rospy.loginfo(
            f"Record video final output intent: videos={args.record_video_final_video_dir}"
        )

    if args.record_video or args.rtc_compare_record:
        frames_dir, videos_dir = resolve_tmp_recording_session_dirs(args)
        args.pic_output_dir = frames_dir
        args.video_output_dir = videos_dir
        args.camera_pic_dir_names = RosOperator.CAMERA_PIC_DIR_NAMES
        args.camera_file_names = RosOperator.CAMERA_FILE_NAMES
    else:
        args.pic_output_dir = None
        args.video_output_dir = None
        args.recording_tmp_session_dir = None
        args.recording_tmp_frames_dir = None
        args.recording_tmp_videos_dir = None


def prompt_keep_recorded_video(record_video_final_video_dir, recording_tmp_frames_dir):
    while True:
        answer = input(
            f"Keep --record_video outputs? Frames are in tmp session {recording_tmp_frames_dir}; "
            f"MP4s will be copied to {record_video_final_video_dir} after background transcode. "
            f"Enter y/yes or n/no: "
        ).strip().lower()
        if answer in ('n', 'no'):
            rospy.loginfo(
                "Record video output discarded. RTC compare MP4 copies (if enabled) are unaffected."
            )
            return False
        if answer in ('y', 'yes'):
            return True
        print("Please answer y/yes or n/no.")


def get_video_copy_outputs(args, include_record_video=True):
    tmp_videos_dir = getattr(args, 'recording_tmp_videos_dir', None)
    if not tmp_videos_dir:
        return []

    copy_pairs = []
    for camera_name, camera_dir_name in RTC_COMPARE_CAMERA_DIR_NAMES.items():
        source_path = str(Path(tmp_videos_dir) / RosOperator.CAMERA_FILE_NAMES[camera_name])
        if include_record_video and getattr(args, 'record_video', False):
            final_video_dir = getattr(args, 'record_video_final_video_dir', None)
            if final_video_dir:
                copy_pairs.append((
                    source_path,
                    str(Path(final_video_dir) / RosOperator.CAMERA_FILE_NAMES[camera_name]),
                ))
        if getattr(args, 'rtc_compare_record', False) and getattr(args, 'rtc_compare_task_dir', None):
            suffix = args.rtc_compare_mode_suffix
            task_dir = Path(args.rtc_compare_task_dir)
            copy_pairs.append((
                source_path,
                str(task_dir / f'{camera_dir_name}_{suffix}.mp4'),
            ))
    return copy_pairs


def finalize_recorded_video_session(args, interrupted=False):
    finalize_video_outputs(args, interrupted=interrupted)


def start_video_transcode_worker(pic_output_dir, video_output_dir, camera_file_names, camera_pic_dir_names, copy_outputs=None):
    worker_code = """
import json
import os
import shutil
import sys
from deploy.utils.rosoperator import save_recorded_videos_from_frames

pic_output_dir = sys.argv[1]
video_output_dir = sys.argv[2]
camera_file_names = json.loads(sys.argv[3])
camera_pic_dir_names = json.loads(sys.argv[4])
copy_outputs = json.loads(sys.argv[5]) if len(sys.argv) > 5 else {}
save_recorded_videos_from_frames(
    pic_output_dir,
    video_output_dir,
    camera_file_names,
    camera_pic_dir_names,
    fps=60.0,
)
copy_output_map = dict(copy_outputs) if isinstance(copy_outputs, list) else (copy_outputs or {})
for source_path, target_path in copy_output_map.items():
    if not os.path.exists(source_path):
        print(f"[Recording] Missing transcode output to copy: {source_path}", flush=True)
        continue
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    shutil.copy2(source_path, target_path)
    print(f"[Recording] Copied transcode output: {source_path} -> {target_path}", flush=True)
""".strip()

    process = subprocess.Popen(
        [
            sys.executable,
            '-c',
            worker_code,
            pic_output_dir,
            video_output_dir,
            json.dumps(camera_file_names),
            json.dumps(camera_pic_dir_names),
            json.dumps(copy_outputs or []),
        ],
        cwd=parent_dir,
        start_new_session=True,
    )
    return process


def setup_rtc_compare_recording_output(args):
    if not getattr(args, 'rtc_compare_record', False) or not getattr(args, 'rtc_compare_task_dir', None):
        args.rtc_compare_recording_enabled = False
        return

    args.rtc_compare_recording_enabled = True
    rospy.loginfo(
        f"RTC compare MP4 output intent: {args.rtc_compare_task_dir} "
        f"(suffix: _{args.rtc_compare_mode_suffix})"
    )


def should_start_video_recording(args):
    return bool(
        getattr(args, 'recording_tmp_frames_dir', None)
        and getattr(args, 'recording_tmp_videos_dir', None)
    )


def link_recorded_frames_to_rtc_compare(args):
    # Deprecated intentionally: rtc_compare must not copy or hard-link frames
    # from --record_video. The MP4s are copied only after the unified transcode
    # worker finishes converting tmp frames -> tmp videos.
    return


def get_rtc_compare_task_name(args):
    task_name = args.task if args.task else 'custom'
    if getattr(args, 'enable_inference_executor', False):
        return f'{task_name}_fast'
    return task_name


def get_rtc_compare_mode_suffix(client_mode):
    return 'withrtc' if client_mode == 'rtc' else 'nortc'


def get_rtc_compare_mode_name(client_mode):
    return 'rtc' if client_mode == 'rtc' else 'non_rtc'


def _resolve_rtc_compare_root(output_dir):
    output_path = Path(output_dir).expanduser()
    if not output_path.is_absolute():
        output_path = Path(parent_dir) / output_path
    return output_path


def prepare_rtc_compare_output(args):
    if not args.rtc_compare_record:
        args.rtc_compare_task_dir = None
        args.rtc_compare_mode_suffix = None
        args.rtc_compare_intervals = []
        return

    task_name = get_rtc_compare_task_name(args)
    mode_suffix = get_rtc_compare_mode_suffix(args.client_mode)
    task_dir = _resolve_rtc_compare_root(args.rtc_compare_output_dir) / task_name
    task_dir.mkdir(parents=True, exist_ok=True)

    for file_name in (
        f'action_timeline_{mode_suffix}.json',
        f'action_timeline_{mode_suffix}.png',
        f'inference_timeline_{mode_suffix}.json',
        f'inference_timeline_{mode_suffix}.png',
        f'camera_h_{mode_suffix}.mp4',
        f'camera_l_{mode_suffix}.mp4',
        f'camera_r_{mode_suffix}.mp4',
        'rtc_vs_nortc_comparison.json',
        'rtc_vs_nortc_comparison.png',
    ):
        path = task_dir / file_name
        if path.exists():
            path.unlink()

    args.rtc_compare_task_dir = str(task_dir)
    args.rtc_compare_mode_suffix = mode_suffix
    args.rtc_compare_mode_name = get_rtc_compare_mode_name(args.client_mode)
    args.rtc_compare_intervals = []
    args.rtc_compare_frame_index = 0
    args.rtc_compare_action_fetch_index = 0
    args.rtc_compare_execution_index = 0
    rospy.loginfo(f"RTC compare output: Enabled -> {task_dir} ({mode_suffix})")


def record_rtc_compare_interval(args, kind, label, start_s, end_s, metadata=None, start_wall_ns=None, end_wall_ns=None):
    if not getattr(args, 'rtc_compare_record', False):
        return
    if not getattr(args, 'rtc_compare_task_dir', None):
        return
    interval = {
        'mode': args.rtc_compare_mode_name,
        'kind': kind,
        'label': label,
        'start_s': float(start_s),
        'end_s': float(end_s),
        'metadata': metadata or {},
    }
    if start_wall_ns is not None and end_wall_ns is not None:
        interval['start_wall_ns'] = int(start_wall_ns)
        interval['end_wall_ns'] = int(end_wall_ns)
    args.rtc_compare_intervals.append(interval)


def record_rtc_compare_action_fetch(args, start_s, end_s, metadata=None, start_wall_ns=None, end_wall_ns=None):
    index = getattr(args, 'rtc_compare_action_fetch_index', 0)
    args.rtc_compare_action_fetch_index = index + 1
    label_prefix = 'rtc' if args.client_mode == 'rtc' else 'chunk'
    record_rtc_compare_interval(
        args,
        'action_fetch',
        f'{label_prefix}_{index}',
        start_s,
        end_s,
        metadata,
        start_wall_ns=start_wall_ns,
        end_wall_ns=end_wall_ns,
    )


def get_server_timing_metadata(result):
    if not isinstance(result, dict):
        return None
    timing = result.get('_timing') or result.get('server_timing')
    if isinstance(timing, dict):
        return timing
    return None


def record_rtc_compare_execution(args, start_s, end_s, step_index, start_wall_ns=None, end_wall_ns=None):
    index = getattr(args, 'rtc_compare_execution_index', 0)
    args.rtc_compare_execution_index = index + 1
    metadata = {
        'global_action_index': int(step_index),
        'control_tick_index': index,
        'source': 'new_rtc_chunk' if args.client_mode == 'rtc' else 'new_non_rtc_chunk',
    }
    if args.client_mode == 'rtc':
        metadata['estimated_delay_steps'] = getattr(args, 'rtc_compare_last_delay_steps', 0)
    record_rtc_compare_interval(
        args,
        'execution',
        f'action_{index}',
        start_s,
        end_s,
        metadata,
        start_wall_ns=start_wall_ns,
        end_wall_ns=end_wall_ns,
    )


def save_rtc_compare_frames(args, image_dict):
    # RTC compare image recording must use RosOperator.start_recording(), the
    # same 60Hz background recording thread as --record_video. Keeping a
    # synchronous fallback here makes the frame count match control steps and
    # hides recording setup bugs, so this path is intentionally disabled.
    return


def _interval_duration(interval):
    return float(interval['end_s']) - float(interval['start_s'])


def _interval_to_json(interval):
    data = {
        'mode': interval['mode'],
        'kind': interval['kind'],
        'label': interval['label'],
        'start_s': round(float(interval['start_s']), 6),
        'end_s': round(float(interval['end_s']), 6),
        'duration_s': round(_interval_duration(interval), 6),
    }
    if 'start_wall_ns' in interval and 'end_wall_ns' in interval:
        data['start_wall_ns'] = int(interval['start_wall_ns'])
        data['end_wall_ns'] = int(interval['end_wall_ns'])
    if interval.get('metadata'):
        data['metadata'] = interval['metadata']
    return data


def _relative_intervals(intervals):
    if not intervals:
        return []
    min_start_s = min(float(interval['start_s']) for interval in intervals)
    wall_starts = []
    for interval in intervals:
        if 'start_wall_ns' in interval:
            wall_starts.append(int(interval['start_wall_ns']))
        server_timing = (interval.get('metadata') or {}).get('server_timing')
        if isinstance(server_timing, dict) and server_timing.get('server_model_start_wall_ns') is not None:
            wall_starts.append(int(server_timing['server_model_start_wall_ns']))
    min_wall_ns = min(wall_starts) if wall_starts else None
    return [
        {
            **interval,
            'start_s': float(interval['start_s']) - min_start_s,
            'end_s': float(interval['end_s']) - min_start_s,
            **(
                {
                    'start_wall_rel_s': (int(interval['start_wall_ns']) - min_wall_ns) / 1e9,
                    'end_wall_rel_s': (int(interval['end_wall_ns']) - min_wall_ns) / 1e9,
                }
                if min_wall_ns is not None and 'start_wall_ns' in interval and 'end_wall_ns' in interval
                else {}
            ),
        }
        for interval in intervals
    ]


def _duration_summary(intervals, kind):
    durations = [_interval_duration(interval) for interval in intervals if interval['kind'] == kind]
    if not durations:
        return {'count': 0, 'mean_s': 0.0, 'p50_s': 0.0, 'max_s': 0.0}
    return {
        'count': len(durations),
        'mean_s': round(statistics.mean(durations), 6),
        'p50_s': round(statistics.median(durations), 6),
        'max_s': round(max(durations), 6),
    }


def _execution_gaps(intervals):
    executions = sorted(
        (interval for interval in intervals if interval['kind'] == 'execution'),
        key=lambda interval: interval['start_s'],
    )
    gaps = []
    for left, right in zip(executions, executions[1:]):
        gap = float(right['start_s']) - float(left['end_s'])
        if gap > 1e-4:
            gaps.append(gap)
    return gaps


def _rtc_compare_summary(intervals):
    gaps = _execution_gaps(intervals)
    return {
        'action_fetch': _duration_summary(intervals, 'action_fetch'),
        'execution': _duration_summary(intervals, 'execution'),
        'execution_gap_count': len(gaps),
        'max_execution_gap_s': round(max(gaps), 6) if gaps else 0.0,
        'mean_execution_gap_s': round(statistics.mean(gaps), 6) if gaps else 0.0,
    }


def _format_ms(seconds):
    return f'{seconds * 1000:.1f} ms'


def _rtc_compare_timeline_duration(args, intervals):
    if getattr(args, 'max_publish_step', 0) and args.max_publish_step > 0 and getattr(args, 'publish_rate', 0) > 0:
        return float(math.ceil(float(args.max_publish_step) / float(args.publish_rate)))
    return max((float(interval['end_s']) for interval in intervals), default=0.0)


def _wall_clock_timeline_intervals(intervals):
    plot_intervals = []
    for interval in intervals:
        metadata = interval.get('metadata') or {}
        server_timing = metadata.get('server_timing')
        if interval.get('kind') == 'action_fetch' and isinstance(server_timing, dict):
            start_ns = server_timing.get('server_model_start_wall_ns')
            end_ns = server_timing.get('server_model_end_wall_ns')
            if start_ns is not None and end_ns is not None:
                plot_intervals.append({
                    **interval,
                    'kind': 'inference',
                    'start_wall_ns': int(start_ns),
                    'end_wall_ns': int(end_ns),
                })
                continue
        if interval.get('kind') == 'action_fetch' and 'start_wall_ns' in interval and 'end_wall_ns' in interval:
            fallback_metadata = dict(metadata)
            fallback_metadata.setdefault('timing_source', 'client_action_fetch_fallback')
            plot_intervals.append({
                **interval,
                'kind': 'inference',
                'metadata': fallback_metadata,
                'start_wall_ns': int(interval['start_wall_ns']),
                'end_wall_ns': int(interval['end_wall_ns']),
            })
        elif interval.get('kind') == 'execution' and 'start_wall_ns' in interval and 'end_wall_ns' in interval:
            plot_intervals.append(interval)

    if not plot_intervals:
        return []
    min_wall_ns = min(int(interval['start_wall_ns']) for interval in plot_intervals)
    return [
        {
            **interval,
            'start_s': (int(interval['start_wall_ns']) - min_wall_ns) / 1e9,
            'end_s': (int(interval['end_wall_ns']) - min_wall_ns) / 1e9,
        }
        for interval in plot_intervals
    ]


def _server_timing_intervals(intervals):
    derived = []
    for interval in intervals:
        if interval.get('kind') != 'action_fetch':
            continue
        metadata = interval.get('metadata') or {}
        server_timing = metadata.get('server_timing')
        if not isinstance(server_timing, dict):
            continue

        server_total_s = float(server_timing.get('server_total_ms', 0.0)) / 1000.0
        server_preprocess_s = float(server_timing.get('server_preprocess_ms', 0.0)) / 1000.0
        server_model_s = float(server_timing.get('server_model_forward_ms', 0.0)) / 1000.0
        if server_total_s <= 0.0:
            continue

        # Server and client clocks are not shared. Align server timing to the
        # client-observed response end for visualization only.
        server_total_end = float(interval['end_s'])
        server_total_start = max(float(interval['start_s']), server_total_end - server_total_s)
        server_model_start = server_total_start + max(server_preprocess_s, 0.0)
        server_model_end = server_model_start + max(server_model_s, 0.0)
        server_model_end = min(server_model_end, server_total_end)

        derived.append({**interval, 'kind': 'server_total', 'start_s': server_total_start, 'end_s': server_total_end})
        if server_model_end > server_model_start:
            derived.append({**interval, 'kind': 'server_model_forward', 'start_s': server_model_start, 'end_s': server_model_end})
    return derived


def _plot_single_timeline(intervals, output_path, title, duration_limit_s=None):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    plot_intervals = _wall_clock_timeline_intervals(intervals) or intervals
    duration_s = duration_limit_s if duration_limit_s is not None else max(
        (interval['end_s'] for interval in plot_intervals),
        default=0.0,
    )
    fig, ax = plt.subplots(figsize=(13, 3.8))
    lanes = [
        ('inference', 11, 'tab:red'),
        ('execution', 2, 'tab:blue'),
    ]
    for kind, y_pos, color in lanes:
        bars = [
            (interval['start_s'], max(_interval_duration(interval), 1e-4))
            for interval in plot_intervals
            if interval['kind'] == kind
        ]
        ax.broken_barh(bars, (y_pos, 5.0), facecolors=color, edgecolors='none', alpha=0.85)
    ax.set_yticks([13.5, 4.5])
    ax.set_yticklabels(['Inference', 'Execution'])
    ax.set_xlabel('time (s)')
    ax.set_xlim(0, max(duration_s, 1e-3))
    ax.set_ylim(0, 18)
    ax.set_title(title)
    ax.grid(axis='x', linestyle='--', alpha=0.35)
    handles = [
        mpatches.Patch(color='tab:red', label='Inference'),
        mpatches.Patch(color='tab:blue', label='Execution'),
    ]
    ax.legend(handles=handles, loc='upper right', frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_comparison_timeline(non_rtc, rtc, output_path):
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    non_rtc_plot = list(non_rtc) + _server_timing_intervals(non_rtc)
    rtc_plot = list(rtc) + _server_timing_intervals(rtc)
    duration_s = max(
        max((interval['end_s'] for interval in non_rtc_plot), default=0.0),
        max((interval['end_s'] for interval in rtc_plot), default=0.0),
        1e-3,
    )
    fig, ax = plt.subplots(figsize=(14, 8.0))
    lanes = [
        ('non_rtc', non_rtc_plot, 'action_fetch', 67, 'tab:red'),
        ('non_rtc', non_rtc_plot, 'server_total', 58, 'tab:orange'),
        ('non_rtc', non_rtc_plot, 'server_model_forward', 49, 'tab:purple'),
        ('non_rtc', non_rtc_plot, 'execution', 40, 'tab:blue'),
        ('rtc', rtc_plot, 'action_fetch', 29, 'tab:red'),
        ('rtc', rtc_plot, 'server_total', 20, 'tab:orange'),
        ('rtc', rtc_plot, 'server_model_forward', 11, 'tab:purple'),
        ('rtc', rtc_plot, 'execution', 2, 'tab:green'),
    ]
    for _mode, intervals, kind, y_pos, color in lanes:
        bars = [
            (interval['start_s'], max(_interval_duration(interval), 1e-4))
            for interval in intervals
            if interval['kind'] == kind
        ]
        ax.broken_barh(bars, (y_pos, 5.0), facecolors=color, edgecolors='none', alpha=0.85)
    ax.set_yticks([69.5, 60.5, 51.5, 42.5, 31.5, 22.5, 13.5, 4.5])
    ax.set_yticklabels([
        'Non-RTC action fetch',
        'Non-RTC server total',
        'Non-RTC server model forward',
        'Non-RTC execution',
        'RTC action fetch',
        'RTC server total',
        'RTC server model forward',
        'RTC execution',
    ])
    ax.set_xlabel('time (s)')
    ax.set_xlim(0, duration_s)
    ax.set_ylim(0, 74)
    ax.set_title('LIFT2 real robot RTC vs non-RTC timing comparison')
    ax.grid(axis='x', linestyle='--', alpha=0.35)
    handles = [
        mpatches.Patch(color='tab:red', label='Action fetch'),
        mpatches.Patch(color='tab:orange', label='Server total'),
        mpatches.Patch(color='tab:purple', label='Server model forward'),
        mpatches.Patch(color='tab:blue', label='Non-RTC execution'),
        mpatches.Patch(color='tab:green', label='RTC execution'),
    ]
    ax.legend(handles=handles, loc='upper right', frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_rtc_compare_mode_outputs(args):
    if not getattr(args, 'rtc_compare_record', False) or not getattr(args, 'rtc_compare_task_dir', None):
        return None

    task_dir = Path(args.rtc_compare_task_dir)
    suffix = args.rtc_compare_mode_suffix
    intervals = _relative_intervals(args.rtc_compare_intervals)
    summary = _rtc_compare_summary(intervals)
    payload = {
        'config': {
            'task': get_rtc_compare_task_name(args),
            'client_mode': args.client_mode,
            'publish_rate': args.publish_rate,
            'execute_horizon': args.execute_horizon,
            'action_chunk_size': args.action_chunk_size,
            'enable_inference_executor': args.enable_inference_executor,
            'executor_rate_hz': args.executor_rate_hz,
            'executor_interpolation': args.executor_interpolation,
            'executor_gripper_mode': args.executor_gripper_mode,
            'rtc_action_horizon': args.rtc_action_horizon,
            'rtc_execution_horizon': args.rtc_execution_horizon,
            'rtc_inference_delay': args.rtc_inference_delay,
        },
        'summary': summary,
        'intervals': [_interval_to_json(interval) for interval in intervals],
    }
    json_path = task_dir / f'action_timeline_{suffix}.json'
    png_path = task_dir / f'action_timeline_{suffix}.png'
    json_path.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    try:
        _plot_single_timeline(
            intervals,
            png_path,
            f'LIFT2 client timeline ({suffix})',
            duration_limit_s=_rtc_compare_timeline_duration(args, intervals),
        )
    except Exception as exc:
        rospy.logwarn(f"[RTC Compare] Failed to plot {png_path}: {exc}")
    rospy.loginfo(f"RTC compare timeline saved: {json_path}")
    return json_path


def write_rtc_compare_combined_outputs(args):
    if not getattr(args, 'rtc_compare_record', False) or not getattr(args, 'rtc_compare_task_dir', None):
        return

    task_dir = Path(args.rtc_compare_task_dir)
    non_rtc_path = task_dir / 'action_timeline_nortc.json'
    rtc_path = task_dir / 'action_timeline_withrtc.json'
    if not non_rtc_path.exists() or not rtc_path.exists():
        return

    non_rtc_payload = json.loads(non_rtc_path.read_text(encoding='utf-8'))
    rtc_payload = json.loads(rtc_path.read_text(encoding='utf-8'))
    non_rtc = non_rtc_payload.get('intervals', [])
    rtc = rtc_payload.get('intervals', [])
    combined = {
        'summary': {
            'non_rtc': non_rtc_payload.get('summary', _rtc_compare_summary(non_rtc)),
            'rtc': rtc_payload.get('summary', _rtc_compare_summary(rtc)),
        },
        'non_rtc': non_rtc,
        'rtc': rtc,
    }
    json_path = task_dir / 'rtc_vs_nortc_comparison.json'
    png_path = task_dir / 'rtc_vs_nortc_comparison.png'
    json_path.write_text(json.dumps(combined, indent=2), encoding='utf-8')
    try:
        _plot_comparison_timeline(non_rtc, rtc, png_path)
    except Exception as exc:
        rospy.logwarn(f"[RTC Compare] Failed to plot {png_path}: {exc}")
    rospy.loginfo(f"RTC compare combined output saved: {json_path}")


def finalize_rtc_compare_session(args):
    if not getattr(args, 'rtc_compare_record', False):
        return
    write_rtc_compare_mode_outputs(args)
    write_rtc_compare_combined_outputs(args)


def finalize_video_outputs(args, interrupted=False):
    tmp_frames_dir = getattr(args, 'recording_tmp_frames_dir', None)
    tmp_videos_dir = getattr(args, 'recording_tmp_videos_dir', None)
    if not tmp_frames_dir or not tmp_videos_dir or not os.path.isdir(tmp_frames_dir):
        return

    include_record_video = False
    if getattr(args, 'record_video', False):
        if interrupted:
            rospy.loginfo("Run interrupted. Prompting whether to keep the recorded video session...")
        try:
            include_record_video = prompt_keep_recorded_video(
                getattr(args, 'record_video_final_video_dir', None),
                tmp_frames_dir,
            )
        except (EOFError, KeyboardInterrupt):
            rospy.loginfo("Record video output discarded after interrupted prompt.")
            include_record_video = False

    copy_outputs = get_video_copy_outputs(args, include_record_video=include_record_video)
    if getattr(args, 'record_video_final_video_dir', None) and include_record_video:
        os.makedirs(args.record_video_final_video_dir, exist_ok=True)

    start_video_transcode_worker(
        tmp_frames_dir,
        tmp_videos_dir,
        RosOperator.CAMERA_FILE_NAMES,
        RosOperator.CAMERA_PIC_DIR_NAMES,
        copy_outputs=copy_outputs,
    )
    rospy.loginfo(
        f"Video transcode started in background: frames={tmp_frames_dir} -> videos={tmp_videos_dir} "
        f"(copy_targets={len(copy_outputs)})"
    )


def wait_for_debug_step_confirmation(step_index):
    while True:
        try:
            answer = input(f"Step {step_index}: Press Enter to execute (Ctrl+C to abort, r to reprint action)...")
        except (KeyboardInterrupt, EOFError):
            raise

        if answer == '':
            return
        if answer.strip().lower() == 'r':
            return 'reprint'

        print("Invalid input. Press Enter to execute, or type r to reprint the action.")


class OpenPIClientModel:
    """OpenPI Inference Client for EEF Delta Control"""

    def __init__(self, host, port, execute_horizon=30,
                 action_chunk_size=30,
                 client_mode='standard', rtc_action_horizon=None, rtc_execution_horizon=None,
                 rtc_inference_delay=0, rtc_control_period_s=1.0 / 30.0,
                 rtc_prefix_attention_schedule='exp', rtc_max_guidance_weight=10.0):
        """
        Args:
            host: Policy server host
            port: Policy server port
            execute_horizon: Number of frames to execute per inference
            action_chunk_size: Number of frames used from each prediction chunk
            client_mode: standard uses local action queue; rtc returns one action per tick
        """
        self.client_mode = client_mode
        if self.client_mode == 'rtc':
            from openpi_client import rtc_client_policy

            self.client = rtc_client_policy.RTCClientPolicy(
                host=host,
                port=port,
                action_horizon=rtc_action_horizon,
                execution_horizon=rtc_execution_horizon,
                inference_delay=rtc_inference_delay,
                control_period_s=rtc_control_period_s,
                prefix_attention_schedule=rtc_prefix_attention_schedule,
                max_guidance_weight=rtc_max_guidance_weight,
            )
        else:
            self.client = websocket_client_policy.WebsocketClientPolicy(
                host=host,
                port=port
            )
        self.execute_horizon = execute_horizon
        self.executed_count = 0
        self.action_chunk_size = action_chunk_size
        self.reset()
        self.current_eef = None

    def reset(self):
        """Reset action queue at the start of each episode"""
        self.action_plan = collections.deque()
        self.executed_count = 0
        self.rtc_pred_eef = None
        self.latest_executor_action_chunk = None
        if self.client_mode == 'rtc':
            self.client.reset()
        return None

    def pop_latest_executor_action_chunk(self):
        action_chunk = self.latest_executor_action_chunk
        self.latest_executor_action_chunk = None
        return action_chunk

    def close(self):
        if hasattr(self.client, 'close'):
            self.client.close()

    def set_current_eef(self, eef):
        """
        Update current EEF state
        Args:
            eef: (14,) Current dual-arm EEF state [xyz, rpy, gripper] × 2
        """
        self.current_eef = eef

    def _build_openpi_observation(self, obs, args):
        head_img = obs['images']['head']
        left_wrist_img = obs['images']['left_wrist']
        right_wrist_img = obs['images']['right_wrist']
        current_eef = self.current_eef.astype(np.float32)
        if not np.all(np.isfinite(current_eef)):
            raise ValueError(f"Current EEF contains NaN/Inf: {current_eef}")

        observation = {
            "observation.images.head": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(head_img, 224, 224)
            ),
            "observation.images.left_wrist": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(left_wrist_img, 224, 224)
            ),
            "observation.images.right_wrist": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(right_wrist_img, 224, 224)
            ),
            "observation.state": current_eef,
            "prompt": args.language_instruction,
        }
        return observation, current_eef

    def _sanitize_delta_action(self, delta_action, args):
        delta_action = np.asarray(delta_action, dtype=np.float32).copy()
        if delta_action.shape != (14,):
            raise ValueError(f"Expected delta action shape (14,), got {delta_action.shape}")
        if not np.all(np.isfinite(delta_action)):
            raise ValueError(f"Policy delta action contains NaN/Inf: {delta_action}")

        max_delta_xyz = float(args.max_delta_xyz)
        max_delta_rpy = float(args.max_delta_rpy)
        clipped_delta = delta_action.copy()
        clipped_delta[0:3] = np.clip(clipped_delta[0:3], -max_delta_xyz, max_delta_xyz)
        clipped_delta[3:6] = np.clip(clipped_delta[3:6], -max_delta_rpy, max_delta_rpy)
        clipped_delta[7:10] = np.clip(clipped_delta[7:10], -max_delta_xyz, max_delta_xyz)
        clipped_delta[10:13] = np.clip(clipped_delta[10:13], -max_delta_rpy, max_delta_rpy)

        if not np.allclose(clipped_delta, delta_action):
            rospy.logwarn(
                "[Safety] Clipped delta action: "
                f"max_xyz={max_delta_xyz:.4f} m, max_rpy={max_delta_rpy:.4f} rad"
            )
        return clipped_delta

    def _smooth_delta_action(self, delta_action, args):
        alpha = float(args.smooth_alpha)
        if alpha >= 1.0:
            return delta_action

        smoothed_delta = delta_action.copy()
        smoothed_delta[0:3] *= alpha
        smoothed_delta[3:6] *= alpha
        smoothed_delta[7:10] *= alpha
        smoothed_delta[10:13] *= alpha
        # Gripper values are postprocessed separately and must not be scaled here.
        smoothed_delta[6] = delta_action[6]
        smoothed_delta[13] = delta_action[13]
        return smoothed_delta

    def _postprocess_gripper(self, action_predict, args):
        action_predict = np.array(action_predict).copy()
        left_gripper_norm = action_predict[6]
        right_gripper_norm = action_predict[13]

        if args.binarize_gripper:
            # Binarize and map to robot gripper range [0, 5].
            GRIPPER_THRESHOLD = 0.6
            GRIPPER_CLOSED = 0.5
            GRIPPER_OPEN = 4.9
            action_predict[6] = GRIPPER_OPEN if left_gripper_norm >= GRIPPER_THRESHOLD else GRIPPER_CLOSED
            action_predict[13] = GRIPPER_OPEN if right_gripper_norm >= GRIPPER_THRESHOLD else GRIPPER_CLOSED
        else:
            action_predict[6] = denormalize_gripper(left_gripper_norm)
            action_predict[13] = denormalize_gripper(right_gripper_norm)

        return action_predict

    def _step_rtc(self, obs, args):
        if obs is None:
            raise ValueError("RTC mode requires a fresh observation on every control tick")

        observation, current_eef = self._build_openpi_observation(obs, args)

        t0_wall_ns = time.time_ns()
        t0 = time.perf_counter()
        result = self.client.infer(observation)
        t1 = time.perf_counter()
        t1_wall_ns = time.time_ns()
        latency_ms = (t1 - t0) * 1000
        delay_steps = self.client.get_estimated_delay_steps()
        args.rtc_compare_last_delay_steps = delay_steps
        server_timing = get_server_timing_metadata(result)
        metadata = {
            'request_type': 'rtc_control_tick_action_fetch',
            'measurement_scope': 'client_observed_action_fetch_latency',
            'client_roundtrip_ms': latency_ms,
            'estimated_delay_steps': delay_steps,
            'execution_horizon': args.rtc_execution_horizon,
            'inference_delay': args.rtc_inference_delay,
        }
        if server_timing is not None:
            metadata['server_timing'] = server_timing
        record_rtc_compare_action_fetch(
            args,
            t0,
            t1,
            metadata,
            start_wall_ns=t0_wall_ns,
            end_wall_ns=t1_wall_ns,
        )

        delta_action = self._sanitize_delta_action(result["actions"], args)
        delta_action = self._smooth_delta_action(delta_action, args)
        # RTC returns one consecutive delta per tick. Accumulate on the last commanded
        # target, matching standard chunk execution and avoiding sensor-lag re-anchoring.
        base_eef = current_eef if self.rtc_pred_eef is None else self.rtc_pred_eef
        action_predict = apply_eef_delta(base_eef, delta_action)
        self.rtc_pred_eef = action_predict.copy()

        if args.log_latency or args.verbose:
            rospy.loginfo(f"[RTC Latency] control tick: {latency_ms:.1f} ms, estimated delay={delay_steps} steps")

        if args.verbose:
            rospy.loginfo(f"[RTC] Delta xyz: L={delta_action[:3]}, R={delta_action[7:10]}")
            rospy.loginfo(f"[RTC] Target xyz: L={action_predict[:3]}, R={action_predict[7:10]}")

        return self._postprocess_gripper(action_predict, args)

    def step(self, obs, args):
        """
        Execute one inference step

        Args:
            obs: Observation dict with images and eef
            args: Command line arguments

        Returns:
            action: (14,) Single-frame action [xyz, rpy, gripper] × 2 (absolute pose)
        """
        if self.client_mode == 'rtc':
            return self._step_rtc(obs, args)

        if not self.action_plan:
            observation, current_eef = self._build_openpi_observation(obs, args)

            # Call remote policy
            t0_wall_ns = time.time_ns()
            t0 = time.perf_counter()
            result = self.client.infer(observation)
            t1 = time.perf_counter()
            t1_wall_ns = time.time_ns()
            latency_ms = (t1 - t0) * 1000
            server_timing = get_server_timing_metadata(result)
            metadata = {
                'request_type': 'non_rtc_chunk_action_fetch',
                'measurement_scope': 'client_observed_action_fetch_latency',
                'client_roundtrip_ms': latency_ms,
                'execute_horizon': self.execute_horizon,
                'action_chunk_size': self.action_chunk_size,
            }
            if server_timing is not None:
                metadata['server_timing'] = server_timing
            record_rtc_compare_action_fetch(
                args,
                t0,
                t1,
                metadata,
                start_wall_ns=t0_wall_ns,
                end_wall_ns=t1_wall_ns,
            )

            if args.log_latency or args.verbose:
                rospy.loginfo(f"[Latency] single inference: {latency_ms:.1f} ms")

            action_chunk = result["actions"]  # Shape: (action_horizon, 14)
            # action_chunk contains delta actions

            # ✅ FIX: Accumulate deltas - each delta is relative to the previous predicted target
            # Training: action[i] = obs[i+1] - obs[i] (consecutive frame deltas)
            # Inference: Must accumulate deltas to get trajectory
            absolute_actions = []
            pred_eef = current_eef.copy()  # Start from current observation
            for delta_action in action_chunk:
                delta_action = self._sanitize_delta_action(delta_action, args)
                delta_action = self._smooth_delta_action(delta_action, args)
                next_eef = apply_eef_delta(pred_eef, delta_action)  # Relative to previous prediction
                absolute_actions.append(next_eef)
                pred_eef = next_eef  # Update for next delta (accumulate)

            # Cache actions to queue
            self.action_plan.extend(absolute_actions)
            executor_actions = [
                self._postprocess_gripper(action, args)
                for action in absolute_actions[:self.execute_horizon]
            ]
            self.latest_executor_action_chunk = np.asarray(
                executor_actions,
                dtype=np.float32,
            )
            self.executed_count = 0

            if args.verbose:
                rospy.loginfo(f"[Inference] Generated {len(absolute_actions)} frames, will execute {self.execute_horizon}")
                rospy.loginfo(f"[Debug] Current EEF xyz: L={current_eef[:3]}, R={current_eef[7:10]}")
                rospy.loginfo(f"[Debug] First delta xyz: L={action_chunk[0, :3]}, R={action_chunk[0, 7:10]}")
                rospy.loginfo(f"[Debug] First delta magnitude: L={np.linalg.norm(action_chunk[0, :3]):.6f}m, R={np.linalg.norm(action_chunk[0, 7:10]):.6f}m")
                rospy.loginfo(f"[Debug] First target xyz: L={absolute_actions[0][:3]}, R={absolute_actions[0][7:10]}")

                # Check if deltas are reasonable (should be mm-scale for right arm)
                right_delta_norm = np.linalg.norm(action_chunk[0, 7:10])
                if right_delta_norm < 0.0001:  # < 0.1mm
                    rospy.logwarn(f"[Warning] Right arm delta is very small ({right_delta_norm*1000:.3f}mm), may indicate denormalization issue")
                elif right_delta_norm > 0.05:  # > 50mm
                    rospy.logwarn(f"[Warning] Right arm delta is very large ({right_delta_norm*1000:.1f}mm), may cause sudden movements")

        # Pop one action frame
        action_predict = np.array(self.action_plan.popleft())
        self.executed_count += 1

        # Clear queue after execute_horizon frames to force re-inference
        if self.executed_count >= self.execute_horizon:
            if len(self.action_plan) > 0:
                if args.verbose:
                    rospy.loginfo(f"[Policy] Executed {self.execute_horizon} frames, discarding {len(self.action_plan)} remaining")
                self.action_plan.clear()
                self.executed_count = 0

        return self._postprocess_gripper(action_predict, args)


def get_action(args, config, ros_operator, policy):
    """
    Get action with intelligent sensor query strategy

    Args:
        args: Command line arguments
        config: Configuration dict
        ros_operator: ROS operator instance
        policy: ClientModel instance
    Returns:
        action: (14,) Action [xyz, rpy, gripper] × 2
    """
    print_flag = True
    rate = rospy.Rate(args.publish_rate)

    while True:
        if rospy.is_shutdown():
            raise KeyboardInterrupt("ROS shutdown while waiting for action")

        # Case 1: Action queue has remaining frames, use directly
        if len(policy.action_plan) > 0:
            action = policy.step(None, args)
            return action

        # Case 2: Queue empty, query sensors and inference
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                rospy.logwarn("Sensor sync failed, waiting...")
                print_flag = False
            rate.sleep()
            continue

        print_flag = True
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         arm_left_pose, arm_right_pose) = result

        # Construct observation dict
        obs = collections.OrderedDict()
        image_dict = {
            config['camera_names'][0]: img_front,
            config['camera_names'][1]: img_left,
            config['camera_names'][2]: img_right
        }
        save_rtc_compare_frames(args, image_dict)
        obs['images'] = image_dict

        if args.use_depth_image:
            image_depth_dict = {
                config['camera_names'][0]: img_front_depth,
                config['camera_names'][1]: img_left_depth,
                config['camera_names'][2]: img_right_depth
            }
            obs['images_depth'] = image_depth_dict

        # Current EEF state
        obs['eef'] = pose_to_eef(arm_left_pose, arm_right_pose)

        # Inference
        policy.set_current_eef(obs['eef'])
        action = policy.step(obs, args)

        return action

    raise KeyboardInterrupt("Interrupted while waiting for action")


def _sleep_control_period(rate_hz):
    """Sleep for one control period without relying on rospy.Rate."""
    time.sleep(1.0 / float(rate_hz))


def move_to_init_pose(ros_operator, left_init, right_init, duration=3.0, rate_hz=15):
    """
    Smoothly move to initial pose using end-effector control

    Args:
        ros_operator: ROS operator instance
        left_init: (7,) Left arm initial pose [x,y,z,roll,pitch,yaw,gripper(0-1)]
        right_init: (7,) Right arm initial pose
        duration: Movement duration (seconds)
        rate_hz: Control frequency
    """
    rospy.loginfo("Moving to initial pose...")

    # Wait for current pose
    wait_timeout_s = max(duration, 5.0)
    wait_deadline = time.time() + wait_timeout_s
    while len(ros_operator.arm_left_pose_deque) == 0 or len(ros_operator.arm_right_pose_deque) == 0:
        if time.time() >= wait_deadline:
            raise TimeoutError("Timed out waiting for arm state before homing")
        rospy.loginfo("Waiting for arm state data...")
        _sleep_control_period(rate_hz)

    # Get current pose
    left_current_msg = ros_operator.arm_left_pose_deque[-1]
    right_current_msg = ros_operator.arm_right_pose_deque[-1]

    current_eef = pose_to_eef(left_current_msg, right_current_msg)
    left_current = current_eef[:7]
    right_current = current_eef[7:14]

    left_init = np.array(left_init)
    right_init = np.array(right_init)

    rospy.loginfo(f"Left arm: {left_current[:3]} → {left_init[:3]}")
    rospy.loginfo(f"Right arm: {right_current[:3]} → {right_init[:3]}")

    total_steps = max(1, int(duration * rate_hz))

    # Linear interpolation
    for step in range(total_steps + 1):
        alpha = step / total_steps

        left_target = left_current * (1 - alpha) + left_init * alpha
        right_target = right_current * (1 - alpha) + right_init * alpha

        # Denormalize gripper for publishing
        left_target_pub = left_target.copy()
        right_target_pub = right_target.copy()
        left_target_pub[6] = denormalize_gripper(left_target[6])
        right_target_pub[6] = denormalize_gripper(right_target[6])

        ros_operator.eef_arm_publish(left_target_pub.tolist(), right_target_pub.tolist())

        if step % 15 == 0:
            progress = int(alpha * 100)
            rospy.loginfo(f"Progress: {progress}% ({step}/{total_steps})")

        if step < total_steps:
            _sleep_control_period(rate_hz)

    rospy.loginfo("Reached initial pose")


def model_inference(args, config, ros_operator):
    """
    Main inference loop

    Args:
        args: Command line arguments
        config: Configuration dict
        ros_operator: ROS operator instance
    """
    policy = OpenPIClientModel(
        args.host,
        args.port,
        execute_horizon=args.execute_horizon,
        action_chunk_size=args.action_chunk_size,
        client_mode=args.client_mode,
        rtc_action_horizon=args.rtc_action_horizon,
        rtc_execution_horizon=args.rtc_execution_horizon,
        rtc_inference_delay=args.rtc_inference_delay,
        rtc_control_period_s=args.rtc_control_period_s,
        rtc_prefix_attention_schedule=args.rtc_prefix_attention_schedule,
        rtc_max_guidance_weight=args.rtc_max_guidance_weight,
    )
    max_publish_step = config['episode_len']
    interrupted = False
    inference_executor = None
    if args.enable_inference_executor:
        executor_class = (
            EEFTrajectoryExecutor
            if args.executor_strategy == 'trajectory_buffer'
            else EEFInterpolatingExecutor
        )
        inference_executor = executor_class(
            ros_operator,
            policy_rate_hz=args.publish_rate,
            executor_rate_hz=args.executor_rate_hz,
            interpolation=args.executor_interpolation,
            gripper_mode=args.executor_gripper_mode,
            max_queue_size=args.executor_max_queue_size,
        )
        rospy.loginfo(
            f"Inference executor strategy: {args.executor_strategy} "
            f"(rtc_committed_prefix={args.rtc_committed_prefix})"
        )

    # Initial pose (normalized gripper [0, 1])
    left_init = args.left_init_pose if args.left_init_pose else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    right_init = args.right_init_pose if args.right_init_pose else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    try:
        # Auto move to initial pose
        if args.auto_init:
            move_to_init_pose(ros_operator, left_init, right_init,
                             duration=args.init_duration, rate_hz=args.publish_rate)
            if args.wait_after_init:
                input("Press Enter to start inference...")

        if inference_executor is not None:
            inference_executor.start()

        # Main inference loop
        start_time = time.time()
        count = 0

        policy.reset()
        t = 0
        rate = rospy.Rate(args.publish_rate)

        # Infinite loop if max_publish_step is 0 or negative
        run_infinite = max_publish_step <= 0

        # Start recording thread
        if should_start_video_recording(args):
            if not getattr(ros_operator, 'video_enabled', False):
                rospy.logerr(
                    "Video recording requested but RosOperator.video_enabled is False. "
                    "Check pic_output_dir/video_output_dir setup before RosOperator initialization."
                )
            ros_operator.start_recording()

        while run_infinite or t < max_publish_step:
            if rospy.is_shutdown():
                interrupted = True
                rospy.loginfo("ROS shutdown detected, stopping inference loop")
                break

            action = get_action(args, config, ros_operator, policy)

            duration = time.time() - start_time
            count += 1

            if args.verbose and t % 50 == 0:
                rospy.loginfo(f"Average Hz: {count/duration:.2f}")

            # Split dual-arm actions
            left_action = action[:7]
            right_action = action[7:14]

            # Gripper values are already in [0, 5] range from step() method
            # No additional processing needed

            # Debug mode: print action details and wait for keypress
            if args.debug:
                while True:
                    rospy.loginfo(f"[Debug Step {t:4d}] Left:  xyz={left_action[:3]}, rpy={left_action[3:6]}, gripper={left_action[6]:.2f}")
                    rospy.loginfo(f"[Debug Step {t:4d}] Right: xyz={right_action[:3]}, rpy={right_action[3:6]}, gripper={right_action[6]:.2f}")
                    try:
                        confirmation = wait_for_debug_step_confirmation(t)
                    except (KeyboardInterrupt, EOFError):
                        rospy.loginfo("Debug mode: user aborted")
                        interrupted = True
                        return interrupted

                    if confirmation == 'reprint':
                        continue
                    break

            # Publish to ROS
            execution_start_wall_ns = time.time_ns()
            execution_start_s = time.perf_counter()
            if inference_executor is not None:
                if args.executor_strategy == 'trajectory_buffer' and args.client_mode == 'rtc':
                    inference_executor.merge_rtc_action(
                        action,
                        committed_prefix=args.rtc_committed_prefix,
                    )
                elif args.executor_strategy == 'trajectory_buffer':
                    executor_action_chunk = policy.pop_latest_executor_action_chunk()
                    if executor_action_chunk is not None:
                        inference_executor.commit_actions(executor_action_chunk)
                else:
                    inference_executor.enqueue(action)
            else:
                ros_operator.eef_arm_publish(left_action, right_action)

            if t % 10 == 0:
                rospy.loginfo(f"[Step {t:4d}] L_gripper={left_action[6]:.2f}, R_gripper={right_action[6]:.2f}")

            step_index = t
            t += 1
            rate.sleep()
            record_rtc_compare_execution(
                args,
                execution_start_s,
                time.perf_counter(),
                step_index,
                start_wall_ns=execution_start_wall_ns,
                end_wall_ns=time.time_ns(),
            )

        if run_infinite:
            rospy.loginfo(f"Infinite mode interrupted, executed {t} steps")
        else:
            rospy.loginfo(f"Episode completed, executed {t} steps")

        return interrupted
    except (KeyboardInterrupt, EOFError):
        interrupted = True
        rospy.loginfo("Inference interrupted by user")
        return interrupted
    finally:
        executor_stopped = True
        if inference_executor is not None:
            normal_completion = not interrupted and not rospy.is_shutdown() and sys.exc_info()[0] is None
            executor_stopped = inference_executor.stop(drain=normal_completion)
        if args.auto_init:
            if not executor_stopped:
                rospy.logwarn(
                    "Skipping return to initial pose because inference executor did not stop cleanly. "
                    "This avoids competing command publishers."
                )
            else:
                rospy.loginfo("Returning to initial pose...")
                try:
                    move_to_init_pose(
                        ros_operator,
                        left_init,
                        right_init,
                        duration=args.init_duration,
                        rate_hz=args.publish_rate,
                    )
                except Exception as exc:
                    rospy.logwarn(f"Failed to return to initial pose: {exc}")
        policy.close()



def get_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='OpenPI LIFT2 Client - EEF Delta Control')

    # Launcher/profile options
    parser.add_argument('--profile', type=str, default=None,
                        help='Launch profile name from launch_profiles.yaml')
    parser.add_argument('--config', type=str, default=str(DEFAULT_LAUNCH_CONFIG),
                        help=f'YAML config file path (default: {DEFAULT_LAUNCH_CONFIG})')
    parser.add_argument('--check', action='store_true', default=False,
                        help='Only validate launch config and connectivity, then exit')
    parser.add_argument('--skip_connectivity_check', action='store_true', default=False,
                        help='Skip policy server connectivity check before startup')
    # Policy server
    parser.add_argument('--host', type=str, default=None,
                        help='Policy server host')
    parser.add_argument('--port', type=int, default=None,
                        help='Policy server port')

    # Task configuration
    parser.add_argument('--task', type=str, choices=sorted(PRESET_TASK_INSTRUCTIONS.keys()),
                        help='Preset task shortcut that auto-fills the language instruction')
    parser.add_argument('--language_instruction', type=str, default=None,
                        help='Language instruction (overrides --task)')
    parser.add_argument('--max_publish_step', type=int, default=None,
                        help='Maximum execution steps (0 or negative for infinite mode)')
    parser.add_argument('--infinite', action='store_true', default=False,
                        help='Run without max step limit')

    # Control parameters
    parser.add_argument('--client_mode', type=str, choices=('standard', 'rtc'), default=None,
                        help='Client execution mode: standard chunks actions locally; rtc requests asynchronous RTC chunks')
    parser.add_argument('--publish_rate', type=int, default=None,
                        help='Control frequency (Hz)')
    parser.add_argument('--execute_horizon', type=int, default=None,
                        help='Frames to execute per inference (default fallback: 30)')
    parser.add_argument('--max_delta_xyz', type=float, default=DEFAULT_MAX_DELTA_XYZ,
                        help=f'Max per-step EEF xyz delta in meters (default: {DEFAULT_MAX_DELTA_XYZ})')
    parser.add_argument('--max_delta_rpy', type=float, default=DEFAULT_MAX_DELTA_RPY,
                        help=f'Max per-step EEF rpy delta in radians (default: {DEFAULT_MAX_DELTA_RPY})')
    parser.add_argument('--smooth_alpha', type=float, default=1.0,
                        help='Scale EEF xyz/rpy deltas before applying them; 1.0 disables smoothing, e.g. 0.2 moves 20%% of each predicted delta')

    # RTC client parameters
    parser.add_argument('--rtc_action_horizon', type=int, default=None,
                        help='RTC policy chunk length; defaults to action_chunk_size')
    parser.add_argument('--rtc_execution_horizon', type=int, default=None,
                        help='RTC background request cadence in control steps; defaults to min(execute_horizon, rtc_action_horizon)')
    parser.add_argument('--rtc_inference_delay', type=int, default=None,
                        help='RTC fixed inference delay in control steps')
    parser.add_argument('--rtc_control_period_s', type=float, default=None,
                        help='RTC control period in seconds; defaults to 1 / publish_rate')
    parser.add_argument('--rtc_prefix_attention_schedule', type=str, default=None,
                        help='RTC prefix attention schedule, e.g. exp or linear')
    parser.add_argument('--rtc_max_guidance_weight', type=float, default=None,
                        help='RTC maximum prefix guidance weight')

    # Inference-only smooth executor
    parser.add_argument('--fast', action='store_true', default=False,
                        help='Enable smooth high-rate inference executor (default: 30Hz policy -> 90Hz publish)')
    parser.add_argument('--enable_inference_executor', action='store_true', default=False,
                        help='Enable background EEF interpolation executor')
    parser.add_argument('--executor_strategy', type=str, default='trajectory_buffer',
                        choices=('trajectory_buffer', 'legacy_interpolate'),
                        help='Executor strategy: trajectory_buffer preserves policy waypoint segments; legacy_interpolate uses last-command interpolation')
    parser.add_argument('--executor_rate_hz', type=float, default=90.0,
                        help='Background executor publish rate in Hz (default: 90)')
    parser.add_argument('--executor_interpolation', type=str, default='linear',
                        choices=('minimum_jerk', 'linear'),
                        help='Executor interpolation method')
    parser.add_argument('--executor_gripper_mode', type=str, default='passthrough',
                        choices=('passthrough', 'interp'),
                        help='Executor gripper handling mode')
    parser.add_argument('--executor_max_queue_size', type=int, default=60,
                        help='Maximum queued high-level EEF actions for the executor')
    parser.add_argument('--rtc_committed_prefix', type=int, default=None,
                        help='RTC executor committed prefix length; defaults to min(rtc_execution_horizon, rtc_inference_delay + 1)')
    parser.add_argument('--action_chunk_size', type=int, default=None,
                        help='Number of frames to use from each prediction chunk (default fallback: 30)')

    # Initialization
    parser.add_argument('--auto_init', action='store_true', default=True,
                        help='Auto move to initial pose on startup')
    parser.add_argument('--no_auto_init', action='store_false', dest='auto_init',
                        help='Disable auto initialization')
    parser.add_argument('--init_duration', type=float, default=3.0,
                        help='Duration to move to initial pose (seconds)')
    parser.add_argument('--wait_after_init', action='store_true', default=False,
                        help='Wait for user confirmation after reaching initial pose')
    parser.add_argument('--left_init_pose', type=float, nargs=7, default=None,
                        help='Left arm initial pose [x y z roll pitch yaw gripper(0-1)]')
    parser.add_argument('--right_init_pose', type=float, nargs=7, default=None,
                        help='Right arm initial pose [x y z roll pitch yaw gripper(0-1)]')

    # Camera topics
    parser.add_argument('--img_front_topic', type=str, default='/camera_h/color/image_raw',
                        help='Front camera topic (head camera)')
    parser.add_argument('--img_left_topic', type=str, default='/camera_l/color/image_raw',
                        help='Left wrist camera topic')
    parser.add_argument('--img_right_topic', type=str, default='/camera_r/color/image_raw',
                        help='Right wrist camera topic')

    parser.add_argument('--img_front_depth_topic', type=str, default='/camera_h/depth/image_rect_raw',
                        help='Front depth camera topic')
    parser.add_argument('--img_left_depth_topic', type=str, default='/camera_l/depth/image_rect_raw',
                        help='Left wrist depth camera topic')
    parser.add_argument('--img_right_depth_topic', type=str, default='/camera_r/depth/image_rect_raw',
                        help='Right wrist depth camera topic')

    parser.add_argument('--use_depth_image', action='store_true', help='Use depth images')

    # Arm topics (EEF pose)
    parser.add_argument('--arm_left_pose_topic', type=str, default='/arm_left/arm_status_ee',
                        help='Left arm end-effector pose topic')
    parser.add_argument('--arm_right_pose_topic', type=str, default='/arm_right/arm_status_ee',
                        help='Right arm end-effector pose topic')

    parser.add_argument('--arm_left_cmd_topic', type=str, default='/arm_left_cmd',
                        help='Left arm command topic')
    parser.add_argument('--arm_right_cmd_topic', type=str, default='/arm_right_cmd',
                        help='Right arm command topic')

    # Debug mode
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Debug mode: press Enter to execute each action step')
    parser.add_argument('--record_video', action='store_true', default=False,
                        help='Record three camera videos to ./video/{task}/{seq}')
    parser.add_argument('--rtc_compare_output_dir', type=str, default='rtc_real_compare',
                        help='Directory for real robot RTC/non-RTC comparison outputs')
    parser.add_argument('--compare', action='store_true', dest='rtc_compare_record', default=False,
                        help='Enable RTC/non-RTC comparison timeline and MP4 recording')
    parser.add_argument('--rtc_compare_record', action='store_true', dest='rtc_compare_record',
                        help='Alias for --compare')
    parser.add_argument('--no_rtc_compare_record', action='store_false', dest='rtc_compare_record',
                        help='Disable RTC/non-RTC comparison image and timeline recording')

    # Gripper control
    parser.add_argument('--binarize_gripper', action='store_true', default=True,
                        help='Binarize gripper to 0/1 (default: True)')
    parser.add_argument('--no_binarize_gripper', action='store_false', dest='binarize_gripper',
                        help='Keep continuous gripper values [0,1]')

    # Verbose / latency
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable verbose logging')
    parser.add_argument('--log_latency', action='store_true', default=False,
                        help='Log single-inference latency (ms) each time')

    args = parser.parse_args()
    if args.infinite:
        args.max_publish_step = 0
    apply_launch_profile(args)
    return finalize_runtime_args(args)


def main():
    """Main function"""
    args = get_arguments()

    if getattr(args, 'profile', None):
        print(format_launcher_summary(args))
        print()

        if not args.skip_connectivity_check:
            print('Checking policy server connectivity...')
            try:
                check_server_connectivity(args.host, args.port)
            except OSError:
                print(f'✗ Cannot reach policy server at {args.host}:{args.port}')
                print('  Please ensure the policy server is running:')
                print('  uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_lift2_lora --policy.dir=<checkpoint_dir>')
                raise SystemExit(1)
            else:
                print('✓ Policy server is reachable')

        if args.check:
            return

        print()
        print('Starting OpenPI client...')
        print()

    rospy.init_node('openpi_lift2_client', anonymous=True)

    rospy.loginfo("="*50)
    rospy.loginfo("OpenPI LIFT2 Client Starting (EEF Delta Control)")
    rospy.loginfo(f"Policy server: {args.host}:{args.port}")
    rospy.loginfo(f"Client mode: {args.client_mode}")
    rospy.loginfo(f"Control rate: {args.publish_rate} Hz")
    rospy.loginfo(f"Execute horizon: {args.execute_horizon} frames")
    if args.client_mode == 'rtc':
        rospy.loginfo(
            "RTC config: "
            f"action_horizon={args.rtc_action_horizon}, "
            f"execution_horizon={args.rtc_execution_horizon}, "
            f"inference_delay={args.rtc_inference_delay}, "
            f"control_period_s={args.rtc_control_period_s:.4f}, "
            f"prefix_attention={args.rtc_prefix_attention_schedule}, "
            f"max_guidance_weight={args.rtc_max_guidance_weight}"
        )
    rospy.loginfo(
        f"Inference executor: {'Enabled' if args.enable_inference_executor else 'Disabled'}"
    )
    if args.enable_inference_executor:
        rospy.loginfo(
            f"  policy {args.publish_rate}Hz -> executor {args.executor_rate_hz:.1f}Hz, "
            f"interpolation={args.executor_interpolation}, gripper_mode={args.executor_gripper_mode}"
        )
    rospy.loginfo(f"Auto initialization: {'Enabled' if args.auto_init else 'Disabled'}")
    if args.auto_init:
        rospy.loginfo(f"  Init duration: {args.init_duration}s")
        if args.left_init_pose:
            rospy.loginfo(f"  Left target: {args.left_init_pose}")
        if args.right_init_pose:
            rospy.loginfo(f"  Right target: {args.right_init_pose}")
    if args.task:
        rospy.loginfo(f"Task preset: {args.task}")
    rospy.loginfo(f"Language instruction: {args.language_instruction}")
    prepare_rtc_compare_output(args)
    setup_recording_output_dirs(args)
    setup_rtc_compare_recording_output(args)
    rospy.loginfo(
        "Video recording state: "
        f"record_video={args.record_video}, "
        f"rtc_compare_record={args.rtc_compare_record}, "
        f"rtc_compare_recording_enabled={getattr(args, 'rtc_compare_recording_enabled', False)}, "
        f"tmp_frames_dir={getattr(args, 'recording_tmp_frames_dir', None)}, "
        f"tmp_videos_dir={getattr(args, 'recording_tmp_videos_dir', None)}, "
        f"record_video_final_video_dir={getattr(args, 'record_video_final_video_dir', None)}"
    )
    if args.debug:
        rospy.loginfo("** DEBUG MODE: Press Enter to execute each step **")
    rospy.loginfo("="*50)

    # Initialize ROS operator
    ros_operator = RosOperator(args)
    if should_start_video_recording(args):
        rospy.loginfo(
            "Starting 60Hz background recording before inference loop: "
            f"tmp_frames={getattr(args, 'recording_tmp_frames_dir', None)}"
        )
        ros_operator.start_recording()

    # Configuration
    config = {
        'episode_len': args.max_publish_step,
        'camera_names': CAMERA_NAMES,
    }

    # Start inference
    interrupted = False
    try:
        interrupted = model_inference(args, config, ros_operator)
    finally:
        ros_operator.close_video_writers()
        finalize_rtc_compare_session(args)
        finalize_video_outputs(args, interrupted=interrupted)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Program interrupted")
    except KeyboardInterrupt:
        rospy.loginfo("User terminated program")
    except Exception as e:
        rospy.logerr(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

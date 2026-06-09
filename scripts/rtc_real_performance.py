"""Run a real-data RTC/non-RTC performance comparison against a live RTC server.

This script loads one observation from a LIFT2-style HDF5 dataset, sends it to a
live policy server, and records wall-clock inference/execution intervals for:

1. non-RTC synchronous chunk inference
2. RTC asynchronous background chunk inference

It writes a JSON summary and a PNG timeline.
"""

import argparse
import dataclasses
import itertools
import json
import pathlib
import statistics
import threading
import time
from typing import Any

import cv2
import h5py
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from openpi_client import image_tools
from openpi_client import msgpack_numpy
from openpi_client import rtc_client_policy
import websockets.sync.client

_GAP_EPS_S = 1e-4


@dataclasses.dataclass(frozen=True)
class _Interval:
    mode: str
    kind: str
    label: str
    start_s: float
    end_s: float
    metadata: dict[str, Any] | None = None

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict[str, Any]:
        data = {
            "mode": self.mode,
            "kind": self.kind,
            "label": self.label,
            "start_s": round(self.start_s, 6),
            "end_s": round(self.end_s, 6),
            "duration_s": round(self.duration_s, 6),
        }
        if self.metadata is not None:
            data["metadata"] = self.metadata
        return data


class _TimedWebsocketRequester:
    def __init__(self, host: str, port: int, mode: str, api_key: str | None = None) -> None:
        self._uri = host if host.startswith("ws") else f"ws://{host}"
        self._uri += f":{port}"
        self._mode = mode
        self._packer = msgpack_numpy.Packer()
        self._lock = threading.Lock()
        self._events = []
        self._request_count = 0

        headers = {"Authorization": f"Api-Key {api_key}"} if api_key else None
        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            additional_headers=headers,
        )
        self.metadata = msgpack_numpy.unpackb(self._ws.recv())

    @property
    def events(self) -> list[_Interval]:
        return list(self._events)

    def request(self, request: dict) -> dict:
        with self._lock:
            request_idx = self._request_count
            self._request_count += 1
            rtc_context = request.get("rtc_context")
            is_rtc_guided = rtc_context is not None
            label_prefix = "rtc" if is_rtc_guided else "chunk"
            start_s = time.perf_counter()
            self._ws.send(self._packer.pack(request))
            response = self._ws.recv()
            end_s = time.perf_counter()

        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")

        self._events.append(
            _Interval(
                mode=self._mode,
                kind="inference",
                label=f"{label_prefix}_{request_idx}",
                start_s=start_s,
                end_s=end_s,
                metadata={
                    "request_id": request_idx,
                    "request_type": "rtc_guided" if is_rtc_guided else "chunk",
                    "inference_delay": rtc_context.get("inference_delay") if is_rtc_guided else None,
                    "execution_horizon": rtc_context.get("execution_horizon") if is_rtc_guided else None,
                    "prefix_attention_horizon": rtc_context.get("prefix_attention_horizon")
                    if is_rtc_guided
                    else None,
                },
            )
        )
        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        self._ws.close()


def _decode_image(encoded_image) -> np.ndarray:
    if isinstance(encoded_image, bytes):
        encoded = np.frombuffer(encoded_image, dtype=np.uint8)
    else:
        encoded = np.asarray(encoded_image, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode JPEG image from dataset frame")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _normalize_gripper(raw_value: float) -> float:
    return float(np.clip(raw_value, 0.0, 5.0) / 5.0)


def _build_eef_state(eef_raw: np.ndarray) -> np.ndarray:
    state = eef_raw.copy().astype(np.float32)
    state[6] = _normalize_gripper(float(eef_raw[6]))
    state[13] = _normalize_gripper(float(eef_raw[13]))
    return state


def _load_dataset_observation(data_dir: pathlib.Path, episode: int, frame: int, prompt: str) -> dict:
    episode_path = data_dir / f"episode_{episode}.hdf5"
    if not episode_path.exists():
        raise FileNotFoundError(f"Dataset episode not found: {episode_path}")

    with h5py.File(episode_path, "r") as f:
        eef_raw = f["observations/eef"][:]
        if frame >= eef_raw.shape[0]:
            raise ValueError(f"Frame {frame} is out of range for {episode_path}; episode has {eef_raw.shape[0]} frames")

        head_img = _decode_image(f["observations/images/head"][frame])
        left_img = _decode_image(f["observations/images/left_wrist"][frame])
        right_img = _decode_image(f["observations/images/right_wrist"][frame])

        return {
            "observation.images.head": image_tools.convert_to_uint8(image_tools.resize_with_pad(head_img, 224, 224)),
            "observation.images.left_wrist": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(left_img, 224, 224)
            ),
            "observation.images.right_wrist": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(right_img, 224, 224)
            ),
            "observation.state": _build_eef_state(eef_raw[frame]),
            "prompt": prompt,
        }


def _execute_until(end_s: float) -> None:
    remaining_s = end_s - time.perf_counter()
    if remaining_s > 0:
        time.sleep(remaining_s)


def _run_non_rtc(
    requester: _TimedWebsocketRequester,
    obs: dict,
    *,
    duration_s: float,
    control_period_s: float,
    non_rtc_execution_horizon: int | None,
) -> list[_Interval]:
    intervals = []
    benchmark_start_s = time.perf_counter()
    action_idx = 0
    chunk_idx = 0

    while time.perf_counter() - benchmark_start_s < duration_s:
        result = requester.request(obs)
        action_chunk = result["actions"]
        chunk_execution_horizon = action_chunk.shape[0]
        if non_rtc_execution_horizon is not None:
            chunk_execution_horizon = min(non_rtc_execution_horizon, chunk_execution_horizon)

        for action_step in range(chunk_execution_horizon):
            if time.perf_counter() - benchmark_start_s >= duration_s:
                break
            start_s = time.perf_counter()
            end_s = min(start_s + control_period_s, benchmark_start_s + duration_s)
            _execute_until(end_s)
            intervals.append(
                _Interval(
                    "non_rtc",
                    "execution",
                    f"chunk{chunk_idx}_action{action_step}",
                    start_s,
                    end_s,
                    metadata={
                        "chunk_id": chunk_idx,
                        "chunk_step": action_step,
                        "global_action_index": action_idx,
                        "source": "new_non_rtc_chunk",
                    },
                )
            )
            action_idx += 1
        chunk_idx += 1

    return requester.events + intervals


def _run_rtc(
    requester: _TimedWebsocketRequester,
    obs: dict,
    *,
    duration_s: float,
    control_period_s: float,
    action_horizon: int | None,
    execution_horizon: int,
    inference_delay: int,
) -> list[_Interval]:
    policy = rtc_client_policy.RTCClientPolicy(
        request_fn=requester.request,
        action_horizon=action_horizon,
        execution_horizon=execution_horizon,
        inference_delay=inference_delay,
        control_period_s=control_period_s,
    )
    intervals = []
    benchmark_start_s = time.perf_counter()
    action_idx = 0
    chunk_ids: dict[int, int] = {}
    pending_reuse_intervals: list[_Interval] = []
    conditioned_future_ids: set[int] = set()

    try:
        while time.perf_counter() - benchmark_start_s < duration_s:
            result = policy.infer(obs)
            if "actions" not in result:
                raise ValueError("RTC policy result does not contain 'actions'")
            active_result = getattr(policy, "_active_result", None)
            active_result_id = id(active_result)
            is_new_active_chunk = active_result_id not in chunk_ids
            if active_result_id not in chunk_ids:
                chunk_ids[active_result_id] = len(chunk_ids)
            chunk_idx = chunk_ids[active_result_id]
            action_chunk = active_result.get("actions") if isinstance(active_result, dict) else None
            chunk_horizon = int(action_chunk.shape[0]) if hasattr(action_chunk, "shape") else action_horizon
            raw_chunk_step = max(0, int(getattr(policy, "_active_step", 1)) - 1)
            chunk_step = _clamp_action_step(raw_chunk_step, chunk_horizon)
            if is_new_active_chunk and chunk_idx > 0:
                intervals.extend(
                    _discarded_prefix_intervals(chunk_idx, chunk_step, pending_reuse_intervals, time.perf_counter())
                )
                pending_reuse_intervals = []
            pending_future = getattr(policy, "_pending_future", None)
            pending_request_step = int(getattr(policy, "_pending_request_step", 0))
            inference_pending = pending_future is not None
            bridge_to_pending_chunk = inference_pending and raw_chunk_step >= pending_request_step
            repeated_last_action = raw_chunk_step != chunk_step
            source = "initial_chunk" if chunk_idx == 0 else "new_rtc_chunk"
            start_s = time.perf_counter()
            if pending_future is not None and id(pending_future) not in conditioned_future_ids:
                conditioned_future_ids.add(id(pending_future))
                intervals.append(
                    _conditioning_interval(
                        chunk_idx,
                        pending_request_step,
                        execution_horizon,
                        chunk_horizon,
                        start_s,
                    )
                )
            end_s = min(start_s + control_period_s, benchmark_start_s + duration_s)
            _execute_until(end_s)
            intervals.append(
                _Interval(
                    "rtc",
                    "execution",
                    f"chunk{chunk_idx}_action{chunk_step}",
                    start_s,
                    end_s,
                    metadata={
                        "chunk_id": chunk_idx,
                        "chunk_step": chunk_step,
                        "raw_chunk_step": raw_chunk_step,
                        "chunk_horizon": chunk_horizon,
                        "global_action_index": action_idx,
                        "pending_request_step": pending_request_step if pending_future is not None else None,
                        "source": source,
                        "inference_pending": inference_pending,
                        "executed_while_inference_pending": inference_pending,
                        "bridge_to_pending_chunk": bridge_to_pending_chunk,
                        "repeated_last_action": repeated_last_action,
                        "reused_action": bridge_to_pending_chunk,
                        "reuse_while_inferring": bridge_to_pending_chunk,
                        "discarded_prefix_steps": chunk_step if is_new_active_chunk and chunk_idx > 0 else 0,
                    },
                )
            )
            if bridge_to_pending_chunk:
                pending_reuse_intervals.append(intervals[-1])
            elif not is_new_active_chunk:
                pending_reuse_intervals = []
            action_idx += 1
    finally:
        policy.close()

    return requester.events + intervals


def _clamp_action_step(raw_step: int, chunk_horizon: int | None) -> int:
    if chunk_horizon is None or chunk_horizon <= 0:
        return max(0, raw_step)
    return max(0, min(raw_step, chunk_horizon - 1))


def _conditioning_interval(
    chunk_id: int,
    start_step: int,
    prefix_horizon: int,
    chunk_horizon: int | None,
    event_s: float,
) -> _Interval:
    end_step = start_step + prefix_horizon
    if chunk_horizon is not None:
        end_step = min(end_step, chunk_horizon)
    return _Interval(
        "rtc",
        "conditioning",
        f"chunk{chunk_id}_condition_actions{start_step}_{max(start_step, end_step - 1)}",
        event_s,
        event_s,
        metadata={
            "chunk_id": chunk_id,
            "start_step": start_step,
            "end_step": end_step,
            "chunk_horizon": chunk_horizon,
            "source": "used_to_generate_next_chunk",
            "prefix_attention_horizon": prefix_horizon,
        },
    )


def _discarded_prefix_intervals(
    incoming_chunk_id: int,
    discarded_steps: int,
    reused_intervals: list[_Interval],
    event_s: float,
) -> list[_Interval]:
    if discarded_steps <= 0:
        return []

    discarded = []
    for discarded_step in range(discarded_steps):
        reused_interval = reused_intervals[discarded_step] if discarded_step < len(reused_intervals) else None
        reused_metadata = reused_interval.metadata if reused_interval is not None and reused_interval.metadata else {}
        discarded.append(
            _Interval(
                "rtc",
                "discarded",
                f"chunk{incoming_chunk_id}_discarded_action{discarded_step}",
                reused_interval.start_s if reused_interval is not None else event_s,
                reused_interval.end_s if reused_interval is not None else event_s,
                metadata={
                    "chunk_id": incoming_chunk_id,
                    "chunk_step": discarded_step,
                    "source": "discarded_rtc_prefix",
                    "reused_chunk_id": reused_metadata.get("chunk_id"),
                    "reused_chunk_step": reused_metadata.get("chunk_step"),
                    "reused_global_action_index": reused_metadata.get("global_action_index"),
                },
            )
        )
    return discarded


def _relative_intervals(intervals: list[_Interval]) -> list[_Interval]:
    min_start_s = min(interval.start_s for interval in intervals)
    return [
        _Interval(
            interval.mode,
            interval.kind,
            interval.label,
            interval.start_s - min_start_s,
            interval.end_s - min_start_s,
            interval.metadata,
        )
        for interval in intervals
    ]


def _execution_gaps(intervals: list[_Interval]) -> list[float]:
    executions = sorted((interval for interval in intervals if interval.kind == "execution"), key=lambda i: i.start_s)
    return [
        right.start_s - left.end_s
        for left, right in itertools.pairwise(executions)
        if right.start_s - left.end_s > _GAP_EPS_S
    ]


def _duration_summary(intervals: list[_Interval], kind: str) -> dict[str, float | int]:
    durations = [interval.duration_s for interval in intervals if interval.kind == kind]
    if not durations:
        return {"count": 0, "mean_s": 0.0, "p50_s": 0.0, "max_s": 0.0}
    return {
        "count": len(durations),
        "mean_s": round(statistics.mean(durations), 6),
        "p50_s": round(statistics.median(durations), 6),
        "max_s": round(max(durations), 6),
    }


def _summarize(intervals: list[_Interval]) -> dict[str, Any]:
    gaps = _execution_gaps(intervals)
    action_sources: dict[str, int] = {}
    reused_action_count = 0
    repeated_last_action_count = 0
    bridge_to_pending_chunk_count = 0
    executed_while_inference_pending_count = 0
    for interval in intervals:
        if interval.kind != "execution":
            continue
        metadata = interval.metadata or {}
        source = metadata.get("source", "unknown")
        action_sources[source] = action_sources.get(source, 0) + 1
        if metadata.get("executed_while_inference_pending"):
            executed_while_inference_pending_count += 1
        if metadata.get("bridge_to_pending_chunk"):
            bridge_to_pending_chunk_count += 1
        if metadata.get("repeated_last_action"):
            repeated_last_action_count += 1
        if metadata.get("reused_action") or source == "reuse_old_chunk_while_inferring":
            reused_action_count += 1
    return {
        "inference": _duration_summary(intervals, "inference"),
        "execution": _duration_summary(intervals, "execution"),
        "action_sources": action_sources,
        "reused_action_count": reused_action_count,
        "repeated_last_action_count": repeated_last_action_count,
        "bridge_to_pending_chunk_count": bridge_to_pending_chunk_count,
        "executed_while_inference_pending_count": executed_while_inference_pending_count,
        "conditioned_action_count": _conditioned_action_count(intervals),
        "discarded_action_count": sum(1 for interval in intervals if interval.kind == "discarded"),
        "execution_gap_count": len(gaps),
        "max_execution_gap_s": round(max(gaps), 6) if gaps else 0.0,
        "mean_execution_gap_s": round(statistics.mean(gaps), 6) if gaps else 0.0,
    }


def _conditioned_action_count(intervals: list[_Interval]) -> int:
    return sum(
        int((interval.metadata or {}).get("end_step", 0)) - int((interval.metadata or {}).get("start_step", 0))
        for interval in intervals
        if interval.kind == "conditioning"
    )


def _write_outputs(
    output_dir: pathlib.Path,
    non_rtc: list[_Interval],
    rtc: list[_Interval],
    config: dict[str, Any],
) -> pathlib.Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    non_rtc = _relative_intervals(non_rtc)
    rtc = _relative_intervals(rtc)

    summary = {
        "non_rtc": _summarize(non_rtc),
        "rtc": _summarize(rtc),
    }

    json_path = output_dir / "rtc_real_performance_timeline.json"
    png_path = output_dir / "rtc_real_performance_timeline.png"
    chunk_fate_path = output_dir / "rtc_real_performance_chunk_fate_timeline.png"
    json_path.write_text(
        json.dumps(
            {
                "config": config,
                "summary": summary,
                "non_rtc": [interval.to_dict() for interval in non_rtc],
                "rtc": [interval.to_dict() for interval in rtc],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_timelines(non_rtc, rtc, png_path, config["duration_s"])
    _plot_rtc_chunk_fate_timeline(rtc, chunk_fate_path, config["duration_s"])
    return png_path


def _plot_timelines(non_rtc: list[_Interval], rtc: list[_Interval], output_path: pathlib.Path, duration_s: float) -> None:
    _plot_readable_timelines(non_rtc, rtc, output_path, duration_s)


def _plot_readable_timelines(
    non_rtc: list[_Interval],
    rtc: list[_Interval],
    output_path: pathlib.Path,
    duration_s: float,
) -> None:
    summary = {"non_rtc": _summarize(non_rtc), "rtc": _summarize(rtc)}
    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(15, 7.5),
        gridspec_kw={"height_ratios": [1.0, 2.0]},
    )
    _plot_summary_table(axes[0], summary)
    _plot_overview_timeline(axes[1], non_rtc, rtc, duration_s)

    handles = [
        mpatches.Patch(color="tab:red", label="Inference"),
        mpatches.Patch(color="tab:blue", label="Non-RTC execution"),
        mpatches.Patch(color="tab:green", label="RTC execution"),
        mpatches.Patch(color="tab:orange", alpha=0.25, label="Execution gap"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=3,
        frameon=False,
        fontsize=9,
        columnspacing=1.4,
        handlelength=1.8,
    )
    fig.suptitle("RTC real-time performance overview", fontsize=14, y=0.985)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96), h_pad=1.2)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_summary_table(ax, summary: dict[str, Any]) -> None:
    ax.axis("off")
    non_rtc = summary["non_rtc"]
    rtc = summary["rtc"]
    rows = [
        (
            "Inference p50",
            _format_ms(non_rtc["inference"]["p50_s"]),
            _format_ms(rtc["inference"]["p50_s"]),
            "single request latency",
        ),
        (
            "Max execution gap",
            _format_ms(non_rtc["max_execution_gap_s"]),
            _format_ms(rtc["max_execution_gap_s"]),
            "smaller is better",
        ),
        (
            "Actions executed",
            str(non_rtc["execution"]["count"]),
            str(rtc["execution"]["count"]),
            "closer to ideal control rate is better",
        ),
        (
            "Actions during pending inference",
            str(non_rtc["executed_while_inference_pending_count"]),
            str(rtc["executed_while_inference_pending_count"]),
            "RTC should hide inference here",
        ),
        (
            "Repeated last action",
            str(non_rtc["repeated_last_action_count"]),
            str(rtc["repeated_last_action_count"]),
            "0 means no queue underrun",
        ),
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Non-RTC", "RTC", "How to read it"],
        loc="center",
        cellLoc="left",
        colLoc="left",
        colWidths=[0.24, 0.15, 0.15, 0.46],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.35)
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("0.92")
        elif col == 2:
            cell.set_facecolor("#edf7ed")


def _plot_overview_timeline(
    ax,
    non_rtc: list[_Interval],
    rtc: list[_Interval],
    duration_s: float,
) -> None:
    lanes = [
        ("non_rtc", "inference", 30, "tab:red", "Non-RTC inference"),
        ("non_rtc", "execution", 21, "tab:blue", "Non-RTC execution"),
        ("rtc", "inference", 10, "tab:red", "RTC inference"),
        ("rtc", "execution", 1, "tab:green", "RTC execution"),
    ]
    by_mode = {"non_rtc": non_rtc, "rtc": rtc}
    for mode, kind, y_pos, color, _label in lanes:
        bars = [
            (interval.start_s, interval.duration_s)
            for interval in by_mode[mode]
            if interval.kind == kind and interval.duration_s > 0
        ]
        ax.broken_barh(bars, (y_pos, 5.5), facecolors=color, edgecolors="none", alpha=0.85)

    _plot_execution_gaps(ax, non_rtc, y_pos=21)
    _plot_execution_gaps(ax, rtc, y_pos=1)
    _annotate_timeline_counts(ax, non_rtc, y_pos=21, label="Non-RTC")
    _annotate_timeline_counts(ax, rtc, y_pos=1, label="RTC")

    ax.set_title("Timeline overview: inference bars should overlap execution, not create gaps")
    ax.set_yticks([32.75, 23.75, 12.75, 3.75])
    ax.set_yticklabels([lane[-1] for lane in lanes])
    ax.set_xlim(0, duration_s)
    ax.set_ylim(0, 38)
    ax.set_xlabel("time (s)")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.text(
        0.01,
        0.98,
        "Detailed conditioning/discarded action events are still written to the JSON; "
        "this figure only shows the latency story.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        color="0.35",
    )


def _plot_rtc_chunk_fate_timeline(rtc: list[_Interval], output_path: pathlib.Path, duration_s: float) -> None:
    fig, (ax_inference, ax) = plt.subplots(
        nrows=2,
        ncols=1,
        sharex=True,
        figsize=(16, 9),
        gridspec_kw={"height_ratios": [0.8, 5.2], "hspace": 0.08},
    )
    chunk_specs = _rtc_chunk_specs(rtc)
    if not chunk_specs:
        ax.text(0.5, 0.5, "No RTC chunks recorded", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        ax_inference.set_axis_off()
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return

    inference_bars = [
        (interval.start_s, interval.duration_s)
        for interval in rtc
        if interval.kind == "inference" and interval.duration_s > 0
    ]
    ax_inference.broken_barh(inference_bars, (0.25, 0.5), facecolors="tab:red", edgecolors="none", alpha=0.35)
    ax_inference.set_yticks([])
    ax_inference.set_ylim(0, 1)
    ax_inference.set_ylabel("inference", fontsize=8)
    ax_inference.grid(axis="x", linestyle="--", alpha=0.25)
    ax_inference.spines[["left", "right", "top"]].set_visible(False)

    row_height = 0.72
    min_width_s = max(_GAP_EPS_S, duration_s * 0.002)
    y_ticks = []
    y_labels = []

    for row_idx, spec in enumerate(reversed(chunk_specs)):
        y_center = row_idx
        chunk_id = spec["chunk_id"]
        horizon = max(1, spec["horizon"])
        ax.broken_barh(
            [(0, duration_s)],
            (y_center - row_height / 2, row_height),
            facecolors="white",
            edgecolors="0.82",
            linewidth=0.7,
            alpha=0.65,
        )

        for interval in sorted(spec["execution_intervals"], key=lambda item: item.start_s):
            metadata = interval.metadata or {}
            width = max(interval.duration_s, min_width_s)
            ax.broken_barh(
                [(interval.start_s, width)],
                (y_center - 0.30, 0.26),
                facecolors="tab:green",
                edgecolors="none",
                alpha=0.9,
            )
            if metadata.get("reuse_while_inferring") or metadata.get("source") == "reuse_old_chunk_while_inferring":
                ax.broken_barh(
                    [(interval.start_s, width)],
                    (y_center - 0.30, 0.26),
                    facecolors="none",
                    edgecolors="tab:orange",
                    linewidth=1.1,
                    hatch="\\\\",
                )

        for interval in sorted(spec["discarded_intervals"], key=lambda item: item.start_s):
            width = max(interval.duration_s, min_width_s)
            ax.broken_barh(
                [(interval.start_s, width)],
                (y_center + 0.02, 0.20),
                facecolors="0.75",
                edgecolors="black",
                linewidth=0.4,
                hatch="///",
                alpha=0.95,
            )

        for interval in sorted(spec["conditioning_intervals"], key=lambda item: item.start_s):
            metadata = interval.metadata or {}
            start_step = _int_or_none(metadata.get("start_step"))
            end_step = _int_or_none(metadata.get("end_step"))
            if start_step is None or end_step is None or end_step <= start_step:
                continue
            ax.vlines(interval.start_s, y_center + 0.20, y_center + 0.36, colors="tab:purple", linewidth=1.2)
            ax.scatter([interval.start_s], [y_center + 0.31], color="tab:purple", marker="D", s=18, zorder=3)

        y_ticks.append(y_center)
        y_labels.append(f"chunk {chunk_id}\nh={horizon}")

    handles = [
        mpatches.Patch(color="tab:red", alpha=0.35, label="RTC inference in flight"),
        mpatches.Patch(color="tab:green", label="executed action"),
        mpatches.Patch(facecolor="none", edgecolor="tab:orange", hatch="\\\\", label="reused while next chunk infers"),
        mpatches.Patch(facecolor="0.75", edgecolor="black", hatch="///", label="discarded incoming action"),
        mpatches.Patch(facecolor="none", edgecolor="tab:purple", label="RTC prefix marker"),
    ]
    fig.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=5,
        fontsize=8,
        frameon=False,
        columnspacing=1.3,
        handlelength=1.8,
    )
    fig.suptitle("RTC returned chunk fate timeline", fontsize=14, y=0.965)
    fig.text(
        0.5,
        0.925,
        "16:9 view: top lane shows background inference; each lower row is one returned action chunk. "
        "Bars are not individually labeled to avoid overlap; use row label + legend to read the fate.",
        ha="center",
        va="top",
        fontsize=8.5,
        color="0.35",
    )
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=7 if len(chunk_specs) > 16 else 8)
    ax.set_xlim(0, duration_s)
    ax.set_ylim(-0.6, len(chunk_specs) - 0.4)
    ax.set_xlabel("time (s), aligned with the overview timeline")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    ax.spines[["right", "top"]].set_visible(False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.12, top=0.88, hspace=0.08)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_rtc_chunk_fate(ax, rtc: list[_Interval]) -> None:
    chunk_specs = _rtc_chunk_specs(rtc)
    if not chunk_specs:
        ax.text(0.5, 0.5, "No RTC chunks recorded", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return

    row_height = 1.0
    row_gap = 0.35
    y_step = row_height + row_gap
    max_horizon = max(max(1, spec["horizon"]) for spec in chunk_specs)
    y_ticks = []
    y_labels = []

    for row_idx, spec in enumerate(reversed(chunk_specs)):
        y_base = row_idx * y_step
        chunk_id = spec["chunk_id"]
        horizon = max(1, spec["horizon"])
        ax.broken_barh([(0, horizon)], (y_base, row_height), facecolors="white", edgecolors="0.7", linewidth=0.7)

        for step, metadata in sorted(spec["executed"].items()):
            ax.broken_barh([(step, 1)], (y_base + 0.08, 0.32), facecolors="tab:green", edgecolors="none", alpha=0.9)
            if metadata.get("reuse_while_inferring") or metadata.get("source") == "reuse_old_chunk_while_inferring":
                ax.broken_barh(
                    [(step, 1)],
                    (y_base + 0.08, 0.32),
                    facecolors="none",
                    edgecolors="tab:orange",
                    linewidth=0.9,
                    hatch="\\\\",
                )

        for step in sorted(spec["discarded"]):
            ax.broken_barh(
                [(step, 1)],
                (y_base + 0.40, 0.25),
                facecolors="0.75",
                edgecolors="black",
                linewidth=0.4,
                hatch="///",
                alpha=0.95,
            )

        for start_step, end_step in spec["conditioned_ranges"]:
            width = max(0, end_step - start_step)
            if width <= 0:
                continue
            ax.broken_barh(
                [(start_step, width)],
                (y_base + 0.68, 0.24),
                facecolors="none",
                edgecolors="tab:purple",
                linewidth=0.9,
                hatch="...",
            )

        y_ticks.append(y_base + row_height / 2)
        y_labels.append(f"chunk {chunk_id} (a0-a{horizon - 1})")

    handles = [
        mpatches.Patch(color="tab:green", label="executed action"),
        mpatches.Patch(facecolor="none", edgecolor="tab:orange", hatch="\\\\", label="reused while next chunk infers"),
        mpatches.Patch(facecolor="none", edgecolor="tab:purple", hatch="...", label="reused as RTC prefix"),
        mpatches.Patch(facecolor="0.75", edgecolor="black", hatch="///", label="discarded incoming action"),
    ]
    ax.legend(handles=handles, loc="upper right", ncol=2, fontsize=8, frameon=True)
    ax.set_title("RTC returned chunk fate: which action slots are executed, reused, or discarded")
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels, fontsize=8)
    ax.set_xlim(0, max_horizon)
    ax.set_ylim(-0.1, len(chunk_specs) * y_step)
    ax.set_xlabel("action index inside returned chunk")
    ax.grid(axis="x", linestyle="--", alpha=0.35)


def _rtc_chunk_specs(intervals: list[_Interval]) -> list[dict[str, Any]]:
    specs: dict[int, dict[str, Any]] = {}

    def spec_for(chunk_id: int) -> dict[str, Any]:
        if chunk_id not in specs:
            specs[chunk_id] = {
                "chunk_id": chunk_id,
                "horizon": 0,
                "executed": {},
                "discarded": set(),
                "conditioned_ranges": [],
                "execution_intervals": [],
                "discarded_intervals": [],
                "conditioning_intervals": [],
            }
        return specs[chunk_id]

    for interval in intervals:
        metadata = interval.metadata or {}
        chunk_id = _int_or_none(metadata.get("chunk_id"))
        if chunk_id is None:
            continue
        spec = spec_for(chunk_id)
        horizon = _int_or_none(metadata.get("chunk_horizon"))
        if horizon is not None:
            spec["horizon"] = max(spec["horizon"], horizon)

        if interval.kind == "execution":
            spec["execution_intervals"].append(interval)
            step = _int_or_none(metadata.get("chunk_step"))
            if step is None:
                continue
            spec["executed"][step] = metadata
            spec["horizon"] = max(spec["horizon"], step + 1)
        elif interval.kind == "discarded":
            spec["discarded_intervals"].append(interval)
            step = _int_or_none(metadata.get("chunk_step"))
            if step is None:
                continue
            spec["discarded"].add(step)
            spec["horizon"] = max(spec["horizon"], step + 1)
        elif interval.kind == "conditioning":
            spec["conditioning_intervals"].append(interval)
            start_step = _int_or_none(metadata.get("start_step"))
            end_step = _int_or_none(metadata.get("end_step"))
            if start_step is None or end_step is None:
                continue
            spec["conditioned_ranges"].append((start_step, end_step))
            spec["horizon"] = max(spec["horizon"], end_step)

    return [specs[chunk_id] for chunk_id in sorted(specs)]


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _plot_execution_gaps(ax, intervals: list[_Interval], y_pos: float) -> None:
    executions = sorted((interval for interval in intervals if interval.kind == "execution"), key=lambda i: i.start_s)
    for left, right in itertools.pairwise(executions):
        gap_s = right.start_s - left.end_s
        if gap_s <= _GAP_EPS_S:
            continue
        ax.broken_barh(
            [(left.end_s, gap_s)],
            (y_pos, 5.5),
            facecolors="tab:orange",
            edgecolors="none",
            alpha=0.25,
        )


def _annotate_timeline_counts(ax, intervals: list[_Interval], y_pos: float, label: str) -> None:
    summary = _summarize(intervals)
    text = f"{label}: {summary['execution']['count']} actions, max gap {_format_ms(summary['max_execution_gap_s'])}"
    ax.text(
        0.995,
        y_pos + 2.75,
        text,
        ha="right",
        va="center",
        fontsize=8,
        color="black",
        bbox={"facecolor": "white", "edgecolor": "0.85", "alpha": 0.85, "pad": 2},
        transform=ax.get_yaxis_transform(),
    )


def _format_ms(seconds: float) -> str:
    return f"{seconds * 1000:.1f} ms"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a real-data RTC/non-RTC performance benchmark")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--api-key")
    parser.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("./dataset_0403_cube"))
    parser.add_argument("--prompt", default="perform task")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--duration-s", type=float, default=10.0)
    parser.add_argument("--control-period-s", type=float, default=0.05)
    parser.add_argument("--action-horizon", type=int)
    parser.add_argument("--execution-horizon", type=int, default=10)
    parser.add_argument(
        "--non-rtc-execution-horizon",
        type=int,
        help="Number of actions to execute per non-RTC chunk. Defaults to the full returned chunk.",
    )
    parser.add_argument("--inference-delay", type=int, default=2)
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("./rtc_real_performance_outputs"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    obs = _load_dataset_observation(args.data_dir, args.episode, args.frame, args.prompt)
    config = {
        "host": args.host,
        "port": args.port,
        "data_dir": str(args.data_dir),
        "episode": args.episode,
        "frame": args.frame,
        "duration_s": args.duration_s,
        "control_period_s": args.control_period_s,
        "action_horizon": args.action_horizon,
        "execution_horizon": args.execution_horizon,
        "non_rtc_execution_horizon": args.non_rtc_execution_horizon,
        "inference_delay": args.inference_delay,
    }

    print(f"Loading observation from {args.data_dir}/episode_{args.episode}.hdf5 frame {args.frame}")
    print(f"Connecting to RTC server at {args.host}:{args.port}")

    non_rtc_requester = _TimedWebsocketRequester(args.host, args.port, "non_rtc", api_key=args.api_key)
    try:
        non_rtc = _run_non_rtc(
            non_rtc_requester,
            obs,
            duration_s=args.duration_s,
            control_period_s=args.control_period_s,
            non_rtc_execution_horizon=args.non_rtc_execution_horizon,
        )
    finally:
        non_rtc_requester.close()

    rtc_requester = _TimedWebsocketRequester(args.host, args.port, "rtc", api_key=args.api_key)
    try:
        rtc = _run_rtc(
            rtc_requester,
            obs,
            duration_s=args.duration_s,
            control_period_s=args.control_period_s,
            action_horizon=args.action_horizon,
            execution_horizon=args.execution_horizon,
            inference_delay=args.inference_delay,
        )
    finally:
        rtc_requester.close()

    output_path = _write_outputs(args.output_dir, non_rtc, rtc, config)
    non_rtc_summary = _summarize(_relative_intervals(non_rtc))
    rtc_summary = _summarize(_relative_intervals(rtc))

    print("\nSummary:")
    print(json.dumps({"non_rtc": non_rtc_summary, "rtc": rtc_summary}, indent=2))
    print(f"\nWrote timeline: {output_path}")
    print(f"Wrote JSON: {args.output_dir / 'rtc_real_performance_timeline.json'}")


if __name__ == "__main__":
    main()

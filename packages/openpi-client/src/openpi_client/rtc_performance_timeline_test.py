import dataclasses
import json
import os
import pathlib

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


_TEST_DURATION_S = 10.0
_CONTROL_PERIOD_S = 0.05
_ACTION_HORIZON = 30
_EXECUTION_HORIZON = 10
_INFERENCE_DELAY = 2
_INFERENCE_TIME_S = 0.14
_EPS = 1e-9


@dataclasses.dataclass(frozen=True)
class _Interval:
    kind: str
    label: str
    start_s: float
    end_s: float

    @property
    def duration_s(self) -> float:
        return self.end_s - self.start_s

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "label": self.label,
            "start_s": round(self.start_s, 6),
            "end_s": round(self.end_s, 6),
            "duration_s": round(self.duration_s, 6),
        }


def _simulate_non_rtc_timeline() -> list[_Interval]:
    """Synchronous chunking: infer the next chunk only after the old chunk is exhausted."""
    intervals = []
    t_s = 0.0
    chunk_idx = 0
    action_idx = 0

    while t_s < _TEST_DURATION_S - _EPS:
        inference_end_s = min(t_s + _INFERENCE_TIME_S, _TEST_DURATION_S)
        intervals.append(_Interval("inference", f"chunk_{chunk_idx}", t_s, inference_end_s))
        t_s = inference_end_s

        for _ in range(_EXECUTION_HORIZON):
            if t_s >= _TEST_DURATION_S - _EPS:
                break
            execution_end_s = min(t_s + _CONTROL_PERIOD_S, _TEST_DURATION_S)
            intervals.append(_Interval("execution", f"action_{action_idx}", t_s, execution_end_s))
            t_s = execution_end_s
            action_idx += 1

        chunk_idx += 1

    return intervals


def _simulate_rtc_timeline() -> list[_Interval]:
    """RTC: after warmup, background inference overlaps continuous action execution."""
    intervals = [_Interval("inference", "initial_chunk", 0.0, _INFERENCE_TIME_S)]

    action_idx = 0
    t_s = _INFERENCE_TIME_S
    while t_s < _TEST_DURATION_S - _EPS:
        execution_end_s = min(t_s + _CONTROL_PERIOD_S, _TEST_DURATION_S)
        intervals.append(_Interval("execution", f"action_{action_idx}", t_s, execution_end_s))
        t_s = execution_end_s
        action_idx += 1

    request_idx = 0
    request_start_s = _INFERENCE_TIME_S
    request_period_s = _EXECUTION_HORIZON * _CONTROL_PERIOD_S
    while request_start_s < _TEST_DURATION_S - _EPS:
        request_end_s = min(request_start_s + _INFERENCE_TIME_S, _TEST_DURATION_S)
        intervals.append(_Interval("inference", f"rtc_next_chunk_{request_idx}", request_start_s, request_end_s))
        request_start_s += request_period_s
        request_idx += 1

    return sorted(intervals, key=lambda interval: (interval.start_s, interval.kind))


def _execution_gaps(intervals: list[_Interval]) -> list[tuple[float, float]]:
    executions = sorted((interval for interval in intervals if interval.kind == "execution"), key=lambda i: i.start_s)
    return [
        (left.end_s, right.start_s)
        for left, right in zip(executions, executions[1:])
        if right.start_s - left.end_s > _EPS
    ]


def _overlaps(left: _Interval, right: _Interval) -> bool:
    return left.start_s < right.end_s and right.start_s < left.end_s


def _write_timeline_outputs(tmp_path: pathlib.Path, non_rtc: list[_Interval], rtc: list[_Interval]) -> pathlib.Path:
    output_dir = pathlib.Path(os.environ.get("RTC_TIMELINE_OUTPUT_DIR", tmp_path))
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "rtc_timeline_comparison.json"
    png_path = output_dir / "rtc_timeline_comparison.png"

    json_path.write_text(
        json.dumps(
            {
                "config": {
                    "test_duration_s": _TEST_DURATION_S,
                    "control_period_s": _CONTROL_PERIOD_S,
                    "action_horizon": _ACTION_HORIZON,
                    "execution_horizon": _EXECUTION_HORIZON,
                    "inference_delay": _INFERENCE_DELAY,
                    "inference_time_s": _INFERENCE_TIME_S,
                },
                "non_rtc": [interval.to_dict() for interval in non_rtc],
                "rtc": [interval.to_dict() for interval in rtc],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _plot_timeline_comparison(non_rtc, rtc, png_path)
    return png_path


def _plot_timeline_comparison(non_rtc: list[_Interval], rtc: list[_Interval], output_path: pathlib.Path) -> None:
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(14, 5.5), sharex=True)
    _plot_timeline(axes[0], non_rtc, "Non-RTC: synchronous inference creates execution gaps")
    _plot_timeline(axes[1], rtc, "RTC: background inference overlaps execution, no post-warmup gaps")

    handles = [
        mpatches.Patch(color="tab:red", label="Inference"),
        mpatches.Patch(color="tab:blue", label="Execution"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2)
    axes[1].set_xlabel("time (s)")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_timeline(ax, intervals: list[_Interval], title: str) -> None:
    lanes = {
        "inference": (12, "tab:red"),
        "execution": (3, "tab:blue"),
    }
    for kind, (y_pos, color) in lanes.items():
        bars = [(interval.start_s, interval.duration_s) for interval in intervals if interval.kind == kind]
        ax.broken_barh(bars, (y_pos, 6), facecolors=color, alpha=0.85)

    ax.set_title(title)
    ax.set_yticks([6, 15])
    ax.set_yticklabels(["execution", "inference"])
    ax.set_xlim(0, _TEST_DURATION_S)
    ax.set_ylim(0, 21)
    ax.grid(axis="x", linestyle="--", alpha=0.4)


def test_rtc_timeline_shows_no_execution_gap_during_inference(tmp_path):
    non_rtc = _simulate_non_rtc_timeline()
    rtc = _simulate_rtc_timeline()

    non_rtc_gaps = _execution_gaps(non_rtc)
    rtc_gaps = _execution_gaps(rtc)
    assert non_rtc_gaps
    assert not rtc_gaps

    rtc_executions = [interval for interval in rtc if interval.kind == "execution"]
    rtc_background_inferences = [
        interval for interval in rtc if interval.kind == "inference" and interval.label != "initial_chunk"
    ]
    assert rtc_background_inferences
    assert all(any(_overlaps(inference, execution) for execution in rtc_executions) for inference in rtc_background_inferences)

    output_path = _write_timeline_outputs(tmp_path, non_rtc, rtc)
    assert output_path.exists()

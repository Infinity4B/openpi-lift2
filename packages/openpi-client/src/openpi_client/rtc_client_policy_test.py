import threading
import time

import numpy as np
import pytest

from openpi_client import rtc_client_policy


def test_rtc_client_returns_actions_and_sends_context():
    requests = []
    release_background = threading.Event()

    def request_fn(request):
        requests.append(request)
        chunk_id = len(requests) - 1
        if "rtc_context" in request:
            release_background.wait(timeout=1.0)
        actions = np.arange(4, dtype=np.float32)[:, None] + chunk_id * 10
        return {
            "actions": actions,
            "_rtc_model_actions": actions + 100,
            "server_timing": {"infer_ms": 1.0},
        }

    policy = rtc_client_policy.RTCClientPolicy(
        request_fn=request_fn,
        action_horizon=4,
        execution_horizon=2,
        inference_delay=1,
        control_period_s=0.01,
    )

    try:
        first = policy.infer({"state": np.array([0], dtype=np.float32)})
        second = policy.infer({"state": np.array([1], dtype=np.float32)})

        assert np.array_equal(first["actions"], np.array([0], dtype=np.float32))
        assert np.array_equal(second["actions"], np.array([1], dtype=np.float32))

        # Let the background request complete, then install it on the next tick.
        release_background.set()
        for _ in range(20):
            if len(requests) >= 2:
                time.sleep(0.01)
                break
            time.sleep(0.01)
        third = policy.infer({"state": np.array([2], dtype=np.float32)})

        assert len(requests) >= 2
        assert "rtc_context" in requests[1]
        rtc_context = requests[1]["rtc_context"]
        assert rtc_context["inference_delay"] == 1
        assert rtc_context["execution_horizon"] == 2
        assert rtc_context["prefix_attention_horizon"] == 2
        assert rtc_context["prefix_attention_schedule"] == "exp"
        assert rtc_context["max_guidance_weight"] == 10.0
        assert rtc_context["prev_action_chunk"].shape == (4, 1)
        assert np.array_equal(rtc_context["prev_action_chunk"][:, 0], np.array([100, 101, 102, 103], dtype=np.float32))
        assert np.array_equal(third["actions"], np.array([12], dtype=np.float32))
    finally:
        policy.close()


def test_rtc_client_reset_clears_active_chunk():
    requests = []

    def request_fn(request):
        requests.append(request)
        value = float(len(requests))
        actions = np.full((3, 2), value, dtype=np.float32)
        return {"actions": actions, "_rtc_model_actions": actions}

    policy = rtc_client_policy.RTCClientPolicy(request_fn=request_fn, action_horizon=3, execution_horizon=2)

    try:
        assert np.array_equal(policy.infer({})["actions"], np.array([1.0, 1.0], dtype=np.float32))
        policy.reset()
        action_after_reset = policy.infer({})["actions"]
        assert np.array_equal(action_after_reset, np.full((2,), float(len(requests)), dtype=np.float32))
    finally:
        policy.close()


def test_rtc_client_blocks_instead_of_repeating_last_action_after_chunk_exhaustion():
    requests = []
    release_background = threading.Event()

    def request_fn(request):
        requests.append(request)
        chunk_id = len(requests) - 1
        if "rtc_context" in request:
            release_background.wait(timeout=1.0)
        actions = np.arange(4, dtype=np.float32)[:, None] + chunk_id * 10
        return {"actions": actions, "_rtc_model_actions": actions + 100}

    policy = rtc_client_policy.RTCClientPolicy(
        request_fn=request_fn,
        action_horizon=4,
        execution_horizon=2,
        inference_delay=1,
        control_period_s=0.01,
    )

    try:
        observed_actions = [policy.infer({})["actions"] for _ in range(4)]
        assert [float(action[0]) for action in observed_actions] == [0.0, 1.0, 2.0, 3.0]

        result_holder = {}
        infer_thread = threading.Thread(target=lambda: result_holder.update(action=policy.infer({})["actions"]))
        infer_thread.start()

        time.sleep(0.05)
        assert infer_thread.is_alive()

        release_background.set()
        infer_thread.join(timeout=1.0)

        assert not infer_thread.is_alive()
        assert len(requests) >= 3
        assert np.array_equal(result_holder["action"], np.array([20], dtype=np.float32))
    finally:
        release_background.set()
        policy.close()


def test_rtc_client_validates_original_horizon_constraints():
    def request_fn(request):
        del request
        actions = np.zeros((4, 1), dtype=np.float32)
        return {"actions": actions, "_rtc_model_actions": actions}

    with pytest.raises(ValueError, match="execution_horizon must be at least inference_delay"):
        rtc_client_policy.RTCClientPolicy(
            request_fn=request_fn,
            action_horizon=4,
            execution_horizon=1,
            inference_delay=2,
        )

    with pytest.raises(ValueError, match="execution_horizon cannot exceed action_horizon"):
        rtc_client_policy.RTCClientPolicy(
            request_fn=request_fn,
            action_horizon=4,
            execution_horizon=5,
            inference_delay=2,
        )


def test_rtc_client_requires_model_space_actions():
    def request_fn(request):
        del request
        return {"actions": np.zeros((4, 1), dtype=np.float32)}

    policy = rtc_client_policy.RTCClientPolicy(
        request_fn=request_fn,
        action_horizon=4,
        execution_horizon=2,
        inference_delay=1,
    )

    try:
        with pytest.raises(ValueError, match="_rtc_model_actions"):
            policy.infer({})
    finally:
        policy.close()

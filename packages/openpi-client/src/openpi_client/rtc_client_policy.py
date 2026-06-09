import concurrent.futures
import copy
import logging
import math
import threading
import time
from typing import Any
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np
from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy

logger = logging.getLogger(__name__)


class RTCClientPolicy(_base_policy.BasePolicy):
    """Asynchronous RTC client that returns one action per control tick.

    The server still returns full chunks. This client keeps executing the active
    chunk while a background request generates the next chunk with RTC prefix
    conditioning. The private `_rtc_model_actions` field returned by the RTC
    server is used as the model-space prefix for the next request.
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        *,
        action_horizon: Optional[int] = None,
        execution_horizon: Optional[int] = 10,
        inference_delay: Optional[int] = None,
        control_period_s: float = 0.02,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 10.0,
        request_fn: Optional[Callable[[Dict], Dict]] = None,
    ) -> None:
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._request_fn = request_fn
        self._ws = None
        if request_fn is None:
            if host.startswith("ws"):
                self._uri = host
            else:
                self._uri = "ws://" + host
            if port is not None:
                self._uri += ":" + str(port)
            self._ws, self._server_metadata = self._wait_for_server()
        else:
            self._uri = "local://rtc-request-fn"
            self._server_metadata = {}

        rtc_metadata = self._server_metadata.get("rtc", {}) if isinstance(self._server_metadata, dict) else {}
        metadata_horizon = (
            self._server_metadata.get("action_horizon") if isinstance(self._server_metadata, dict) else None
        )
        self._action_horizon = action_horizon or metadata_horizon
        self._execution_horizon = execution_horizon if execution_horizon is not None else rtc_metadata.get(
            "execution_horizon", 10
        )
        self._inference_delay = int(
            inference_delay if inference_delay is not None else rtc_metadata.get("inference_delay", 0)
        )
        self._control_period_s = control_period_s
        self._prefix_attention_schedule = prefix_attention_schedule or rtc_metadata.get(
            "prefix_attention_schedule", "exp"
        )
        self._max_guidance_weight = float(
            max_guidance_weight if max_guidance_weight is not None else rtc_metadata.get("max_guidance_weight", 10.0)
        )
        self._validate_rtc_config()

        self._active_result = None  # type: Optional[Dict]
        self._active_model_chunk = None  # type: Optional[np.ndarray]
        self._active_step = 0
        self._next_request_step = 0
        self._estimated_delay_steps = 0

        self._pending_future = None  # type: Optional[concurrent.futures.Future]
        self._pending_request_step = 0
        self._pending_request_time = 0.0

        self._lock = threading.RLock()
        self._ws_lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def get_estimated_delay_steps(self) -> int:
        return self._estimated_delay_steps

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info("Waiting for RTC server at %s...", self._uri)
        while True:
            try:
                headers = {"Authorization": "Api-Key " + self._api_key} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for RTC server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict) -> Dict:  # noqa: UP006
        with self._lock:
            self._maybe_install_pending_locked()

            if self._active_result is None:
                result = self._request_chunk({"obs": obs})
                self._install_chunk_locked(result, active_step=0)
                self._next_request_step = 0

            self._maybe_schedule_request_locked(obs)

            if self._is_active_chunk_exhausted_locked():
                self._wait_for_next_chunk_locked(obs)

            action = self._slice_active_action_locked()
            self._active_step += 1
            return action

    @override
    def reset(self) -> None:
        with self._lock:
            if self._pending_future is not None:
                self._pending_future.cancel()
            self._active_result = None
            self._active_model_chunk = None
            self._active_step = 0
            self._next_request_step = 0
            self._pending_future = None
            self._pending_request_step = 0
            self._pending_request_time = 0.0
            self._estimated_delay_steps = 0

    def close(self) -> None:
        self._executor.shutdown(wait=False)
        if self._ws is not None:
            self._ws.close()

    def _request_chunk(self, request: Dict) -> Dict:
        if self._request_fn is not None:
            return self._request_fn(request)

        data = self._packer.pack(request)
        with self._ws_lock:
            assert self._ws is not None
            self._ws.send(data)
            response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError("Error in RTC inference server:\n" + response)
        return msgpack_numpy.unpackb(response)

    def _maybe_schedule_request_locked(self, obs: Dict) -> None:
        if self._pending_future is not None or self._active_model_chunk is None:
            return
        if self._action_horizon is None:
            return
        self._validate_rtc_config()
        if self._active_step < self._next_request_step:
            return

        shifted_prefix = self._shift_chunk_left(self._active_model_chunk, self._active_step)
        rtc_context = {
            "prev_action_chunk": shifted_prefix,
            "inference_delay": self._inference_delay,
            "execution_horizon": self._execution_horizon,
            "prefix_attention_horizon": self._execution_horizon,
            "prefix_attention_schedule": self._prefix_attention_schedule,
            "max_guidance_weight": self._max_guidance_weight,
        }
        request = {"obs": copy.deepcopy(obs), "rtc_context": rtc_context}
        self._pending_request_step = self._active_step
        self._pending_request_time = time.monotonic()
        self._pending_future = self._executor.submit(self._request_chunk, request)
        self._next_request_step = self._action_horizon + 1

    def _maybe_install_pending_locked(self) -> None:
        if self._pending_future is None or not self._pending_future.done():
            return

        future = self._pending_future
        self._pending_future = None
        try:
            result = future.result()
        except Exception:
            logger.exception("RTC background inference failed; retrying on the next tick.")
            self._next_request_step = self._active_step
            return

        elapsed_steps = max(0, self._active_step - self._pending_request_step)
        elapsed_s = max(0.0, time.monotonic() - self._pending_request_time)
        if self._control_period_s > 0:
            self._estimated_delay_steps = max(0, int(math.ceil(elapsed_s / self._control_period_s)))
        if self._action_horizon is not None and elapsed_steps >= self._action_horizon:
            logger.warning(
                "RTC background chunk arrived after all %s actions were already exhausted; requesting a fresh chunk.",
                self._action_horizon,
            )
            return
        self._install_chunk_locked(result, active_step=elapsed_steps)

    def _wait_for_next_chunk_locked(self, obs: Dict) -> None:
        """Block instead of repeating the final action when the current chunk is exhausted."""
        assert self._action_horizon is not None

        if self._pending_future is None:
            result = self._request_chunk({"obs": obs})
            self._install_chunk_locked(result, active_step=0)
            return

        future = self._pending_future
        self._pending_future = None
        try:
            result = future.result()
        except Exception:
            logger.exception(
                "RTC background inference failed after active chunk was exhausted; requesting a fresh chunk."
            )
            result = self._request_chunk({"obs": obs})
            self._install_chunk_locked(result, active_step=0)
            return

        elapsed_steps = max(0, self._active_step - self._pending_request_step)
        elapsed_s = max(0.0, time.monotonic() - self._pending_request_time)
        if self._control_period_s > 0:
            self._estimated_delay_steps = max(0, int(math.ceil(elapsed_s / self._control_period_s)))

        if elapsed_steps >= self._action_horizon:
            logger.warning(
                "RTC background chunk arrived after all %s actions were already exhausted; requesting a fresh chunk.",
                self._action_horizon,
            )
            result = self._request_chunk({"obs": obs})
            self._install_chunk_locked(result, active_step=0)
            return

        self._install_chunk_locked(result, active_step=elapsed_steps)

    def _install_chunk_locked(self, result: Dict, active_step: int) -> None:
        if "actions" not in result or not isinstance(result["actions"], np.ndarray):
            raise ValueError("RTC server response must include an ndarray 'actions' chunk")

        action_horizon = result["actions"].shape[0]
        if self._action_horizon is None:
            self._action_horizon = action_horizon
        elif action_horizon != self._action_horizon:
            raise ValueError("Expected action horizon %s, got %s" % (self._action_horizon, action_horizon))

        model_chunk = result.get("_rtc_model_actions")
        if not isinstance(model_chunk, np.ndarray):
            raise ValueError("RTC server response must include an ndarray '_rtc_model_actions' chunk")
        if model_chunk.shape[0] != self._action_horizon:
            raise ValueError("RTC model action chunk has incompatible horizon %s" % (model_chunk.shape[0],))

        self._active_result = result
        self._active_model_chunk = model_chunk
        self._active_step = max(0, min(active_step, self._action_horizon - 1))
        self._validate_rtc_config()
        self._next_request_step = self._execution_horizon

    def _slice_active_action_locked(self) -> Dict:
        assert self._active_result is not None
        assert self._action_horizon is not None
        if self._active_step >= self._action_horizon:
            raise RuntimeError("Active RTC chunk is exhausted; wait for or request the next chunk before slicing.")
        step = max(0, self._active_step)
        return {
            key: self._slice_value(value, step)
            for key, value in self._active_result.items()
            if not key.startswith("_rtc_")
        }

    def _is_active_chunk_exhausted_locked(self) -> bool:
        return self._action_horizon is not None and self._active_step >= self._action_horizon

    def _slice_value(self, value: Any, step: int) -> Any:
        if isinstance(value, np.ndarray) and self._action_horizon is not None and value.shape[:1] == (
            self._action_horizon,
        ):
            return value[step, ...]
        if isinstance(value, dict):
            return {k: self._slice_value(v, step) for k, v in value.items()}
        return value

    def _shift_chunk_left(self, chunk: np.ndarray, start_step: int) -> np.ndarray:
        assert self._action_horizon is not None
        start_step = max(0, min(start_step, self._action_horizon))
        shifted = np.zeros_like(chunk)
        if start_step < self._action_horizon:
            shifted[: self._action_horizon - start_step] = chunk[start_step:]
        return shifted

    def _validate_rtc_config(self) -> None:
        if self._inference_delay < 0:
            raise ValueError(f"inference_delay must be non-negative, got {self._inference_delay}")
        if self._action_horizon is None or self._execution_horizon is None:
            return
        if self._execution_horizon <= 0:
            raise ValueError(f"execution_horizon must be positive, got {self._execution_horizon}")
        if self._execution_horizon > self._action_horizon:
            raise ValueError(
                f"execution_horizon cannot exceed action_horizon: {self._execution_horizon} > {self._action_horizon}"
            )
        if self._execution_horizon < self._inference_delay:
            raise ValueError(
                "execution_horizon must be at least inference_delay: "
                f"{self._execution_horizon} < {self._inference_delay}"
            )

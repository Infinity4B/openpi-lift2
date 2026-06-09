import asyncio
import http
import logging
import time
import traceback

import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class RTCPolicyServer:
    """Serves action chunks with optional inference-time RTC context over websocket.

    Requests are msgpack dictionaries with either a raw observation payload or:
        {"obs": observation, "rtc_context": {...}}

    The response is the policy output chunk. For RTC requests, the server also
    returns `_rtc_model_actions`, the normalized/model-space chunk used as the
    prefix target for the next inpainting request.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
        *,
        debug_print: bool = False,
        debug_interval: int = 1,
        fix_state_mask: np.ndarray | None = None,
        fix_state_values: np.ndarray | None = None,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 10.0,
        execution_horizon: int | None = 10,
        inference_delay: int = 0,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = {
            **(metadata or {}),
            "rtc": {
                "prefix_attention_schedule": prefix_attention_schedule,
                "max_guidance_weight": max_guidance_weight,
                "execution_horizon": execution_horizon,
                "inference_delay": inference_delay,
            },
        }
        self._debug_print = debug_print
        self._debug_interval = debug_interval
        self._fix_state_mask = fix_state_mask
        self._fix_state_values = fix_state_values
        self._prefix_attention_schedule = prefix_attention_schedule
        self._max_guidance_weight = max_guidance_weight
        self._execution_horizon = execution_horizon
        self._inference_delay = inference_delay
        self._step_count = 0
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info("RTC connection from %s opened", websocket.remote_address)
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        while True:
            try:
                start_time = time.monotonic()
                request = msgpack_numpy.unpackb(await websocket.recv())
                obs, rtc_context = self._parse_request(request)

                if self._fix_state_mask is not None and "observation.state" in obs:
                    state = obs["observation.state"]
                    if isinstance(state, np.ndarray):
                        state = state.copy()
                        state[self._fix_state_mask] = self._fix_state_values[self._fix_state_mask]
                        obs["observation.state"] = state

                if self._debug_print and self._step_count % self._debug_interval == 0:
                    self._print_debug_input(obs, rtc_context)

                infer_time = time.monotonic()
                action = self._policy.infer(obs, rtc_context=rtc_context, return_model_actions=True)
                infer_time = time.monotonic() - infer_time

                if self._debug_print and self._step_count % self._debug_interval == 0:
                    self._print_debug_output(action, infer_time)

                self._step_count += 1
                action["server_timing"] = {"infer_ms": infer_time * 1000}
                if prev_total_time is not None:
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000

                await websocket.send(packer.pack(action))
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info("RTC connection from %s closed", websocket.remote_address)
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise

    def _parse_request(self, request: dict) -> tuple[dict, dict | None]:
        if "obs" not in request:
            return request, None

        rtc_context = request.get("rtc_context")
        if rtc_context is not None:
            rtc_context = dict(rtc_context)
            rtc_context.setdefault("inference_delay", self._inference_delay)
            rtc_context.setdefault("prefix_attention_schedule", self._prefix_attention_schedule)
            rtc_context.setdefault("max_guidance_weight", self._max_guidance_weight)
            if self._execution_horizon is not None:
                rtc_context.setdefault("execution_horizon", self._execution_horizon)
                # Legacy clients may still read this field; it now has the same LeRobot-style meaning.
                rtc_context.setdefault("prefix_attention_horizon", self._execution_horizon)
        return request["obs"], rtc_context

    def _print_debug_input(self, obs: dict, rtc_context: dict | None) -> None:
        print(f"\n{'='*80}")
        print(f"[RTC DEBUG] Step {self._step_count} - MODEL INPUT")
        print(f"{'='*80}")
        if rtc_context is not None:
            print(
                "rtc_context: "
                f"inference_delay={rtc_context.get('inference_delay')}, "
                f"execution_horizon={rtc_context.get('execution_horizon')}, "
                f"prefix_attention_horizon={rtc_context.get('prefix_attention_horizon')}"
            )
        for key, value in obs.items():
            if "image" in key.lower():
                continue
            if isinstance(value, np.ndarray):
                print(
                    f"obs[{key!r}]: shape={value.shape}, dtype={value.dtype}, "
                    f"range=[{value.min():.4f}, {value.max():.4f}]"
                )
            else:
                print(f"obs[{key!r}]: {value}")

    def _print_debug_output(self, action: dict, infer_time: float) -> None:
        print(f"\n{'='*80}")
        print(f"[RTC DEBUG] Step {self._step_count} - MODEL OUTPUT")
        print(f"{'='*80}")
        for key, value in action.items():
            if isinstance(value, np.ndarray):
                print(
                    f"action[{key!r}]: shape={value.shape}, dtype={value.dtype}, "
                    f"range=[{value.min():.4f}, {value.max():.4f}]"
                )
            else:
                print(f"action[{key!r}]: {value}")
        print(f"Inference time: {infer_time * 1000:.2f} ms")


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    return None

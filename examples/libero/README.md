# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.8 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e third_party/libero
export PYTHONPATH=$PYTHONPATH:$PWD/third_party/libero

# Run the simulation
python examples/libero/main.py

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

## RTC Evaluation

RTC (Real-Time Chunking) evaluation is intended to compare the standard synchronous chunk rollout with an asynchronous rollout that keeps executing the current chunk while the next chunk is inferred in the background with RTC prefix guidance.

Important: the current `examples/libero/main.py` uses `WebsocketClientPolicy` and a local `replan_steps` action queue. That is the standard non-RTC baseline. Starting the server with `--server-mode rtc` alone does not make the rollout use RTC; use `examples/libero/main_rtc.py` for the RTC rollout client.

### Start an RTC policy server

With Docker, pass the RTC server arguments through `SERVER_ARGS` and switch the runtime container to the RTC rollout script with `CLIENT_SCRIPT`:

```bash
CLIENT_SCRIPT=examples/libero/main_rtc.py \
SERVER_ARGS="--env LIBERO --server-mode rtc --port 8000 --rtc-execution-horizon 5 --rtc-inference-delay 2 --rtc-prefix-attention-schedule exp --rtc-max-guidance-weight 10.0" \
CLIENT_ARGS="--args.action-horizon 10 --args.execution-horizon 5 --args.inference-delay 2 --args.control-period-s 0.05 --args.prefix-attention-schedule exp --args.max-guidance-weight 10.0" \
  docker compose -f examples/libero/compose.yml up --build
```

Without Docker, run the RTC server directly:

```bash
uv run scripts/serve_policy.py \
  --env LIBERO \
  --server-mode rtc \
  --port 8000 \
  --rtc-execution-horizon 5 \
  --rtc-inference-delay 2 \
  --rtc-prefix-attention-schedule exp \
  --rtc-max-guidance-weight 10.0
```

To evaluate a custom checkpoint, keep the same RTC flags and append the checkpoint loader arguments:

```bash
uv run scripts/serve_policy.py \
  --env LIBERO \
  --server-mode rtc \
  --port 8000 \
  --rtc-execution-horizon 5 \
  --rtc-inference-delay 2 \
  --rtc-prefix-attention-schedule exp \
  --rtc-max-guidance-weight 10.0 \
  policy:checkpoint \
  --policy.config pi05_libero \
  --policy.dir ./my_custom_checkpoint
```

Recommended starting parameters for the default LIBERO checkpoint are:

- `action_horizon=10`: the policy chunk length used by `pi05_libero` in `src/openpi/training/config.py`.
- `execution_horizon=5`: match the standard LIBERO evaluator's `replan_steps=5`, so RTC starts the next background request after 5 executed control steps.
- `inference_delay=2`: fixed delay in control timesteps. Adjust this by measuring server latency and converting it to control timesteps.
- `control_period_s=0.05`: LIBERO's default `control_freq=20`, so each environment control step is 1 / 20 seconds. The RTC client uses this for delay-step estimation.
- `prefix_attention_schedule=exp` and `max_guidance_weight=10.0`: default RTC guidance settings.

### Run RTC LIBERO rollouts

The RTC rollout entrypoint is `examples/libero/main_rtc.py`. It builds the same LIBERO observation dict as `examples/libero/main.py`, but uses `openpi_client.rtc_client_policy.RTCClientPolicy`, calls `policy.infer(element)` every environment step, and immediately executes the returned single action. It does not fill a local `action_plan` with `replan_steps` actions, because RTC scheduling is handled by the client policy.

After starting the RTC server without Docker, run the RTC client in the LIBERO environment:

```bash
python examples/libero/main_rtc.py \
  --args.action-horizon 10 \
  --args.execution-horizon 5 \
  --args.inference-delay 2 \
  --args.control-period-s 0.05 \
  --args.prefix-attention-schedule exp \
  --args.max-guidance-weight 10.0
```

To run a specific task suite, pass the same task-suite flag used by the standard evaluator:

```bash
python examples/libero/main_rtc.py \
  --args.task-suite-name libero_10 \
  --args.action-horizon 10 \
  --args.execution-horizon 5 \
  --args.inference-delay 2
```

The RTC client should be configured consistently with the server, for example:

```python
from openpi_client import rtc_client_policy

policy = rtc_client_policy.RTCClientPolicy(
    host="localhost",
    port=8000,
    action_horizon=10,
    execution_horizon=5,
    inference_delay=2,
    control_period_s=0.05,
    prefix_attention_schedule="exp",
    max_guidance_weight=10.0,
)
```

Call `policy.reset()` at the start of each LIBERO episode so the first chunk of every rollout is requested synchronously and later chunks are scheduled with a fresh RTC state.

### Suggested evaluation matrix

Run each selected task suite with the same checkpoint, seed, number of trials, and video settings:

| Condition | Server mode | Client behavior | Notes |
|-----------|-------------|-----------------|-------|
| Baseline chunk rollout | `websocket` | Current `WebsocketClientPolicy` with `replan_steps` | Reproduces the standard LIBERO evaluation path. |
| RTC exp | `rtc` | `RTCClientPolicy`, `action_horizon=10`, `execution_horizon=5`, `inference_delay=2`, `prefix_attention_schedule=exp` | Main RTC comparison. |
| RTC linear (optional) | `rtc` | Same as above, but `prefix_attention_schedule=linear` | Ablates the prefix weighting schedule. |
| RTC delay sweep (optional) | `rtc` | Same as RTC exp, but vary `inference_delay` | Use values derived from measured inference latency. |

Report at least the following metrics:

- success rate for each LIBERO suite (`libero_spatial`, `libero_object`, `libero_goal`, and `libero_10`);
- average success rate across suites;
- average episode wall-clock time;
- server inference latency or the measured latency used to choose `inference_delay`;
- whether replay videos were saved and the `video_out_path` used for each condition.

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85

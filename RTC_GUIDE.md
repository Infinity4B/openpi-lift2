# OpenPI RTC 使用指南

本文档简要说明当前分支 `wx_dev_rtc` 中实现的 RTC（Real-Time Chunking）功能，以及如何启动、使用和验证。

## 1. 做了什么

本分支实现的是 **inference-time RTC**：

- 不需要重新训练或微调模型。
- 模型仍然输出 action chunk。
- 客户端执行当前 chunk 的同时，在后台异步请求下一段 chunk。
- 下一段 chunk 会使用上一段 chunk 的 model-space action 作为 prefix 约束，使 chunk 之间更连续。
- 目标是在推理耗时较长时，仍然保持动作执行不中断。

整体新增了一套 RTC client/server：

- RTC server：支持带 `rtc_context` 的推理请求。
- RTC client：按控制周期逐步返回 action，同时后台请求下一段 chunk。
- 性能时间轴测试：用于对比 non-RTC 和 RTC 的推理/执行时间线，展示 RTC 后台推理与执行重叠。

当前实现遵循 LeRobot 的 inference-time RTC 语义：使用固定 `inference_delay`、`execution_horizon` 和 prefix attention schedule。`execution_horizon` 直接作为 prefix attention 的 end；兼容字段 `prefix_attention_horizon` 会被视为同一含义。

## 2. 启动 RTC server

普通 websocket server 仍然可以照常启动：

```bash
uv run scripts/serve_policy.py \
  --server-mode websocket \
  --port 8000
```

启动 RTC server：

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
  --server-mode rtc \
  --port 8000 \
  --rtc-execution-horizon 10 \
  --rtc-inference-delay 2 \
  --rtc-prefix-attention-schedule exp \
  --rtc-max-guidance-weight 10.0
```

参数含义：

- `rtc-execution-horizon`：与 LeRobot 对齐的 RTC horizon，既是 client 开始请求下一段 chunk 的执行步数，也是模型中 `get_prefix_weights(inference_delay, execution_horizon, action_horizon)` 的 prefix attention end。
- `rtc-inference-delay`：固定推理延迟，单位是控制 timestep。
- `rtc-prefix-attention-schedule`：prefix 权重方式，支持 `exp`、`linear`、`ones`、`zeros`。
- `rtc-max-guidance-weight`：RTC guidance 的最大强度，默认 `10.0`。

约束条件：

```text
0 <= inference_delay <= execution_horizon <= action_horizon
```

例如 `action_horizon=30`、`execution_horizon=10`、`inference_delay=2` 是合法配置。

## 3. 使用 RTC client

示例：

```python
from openpi_client import rtc_client_policy

policy = rtc_client_policy.RTCClientPolicy(
    host="localhost",
    port=8000,
    action_horizon=30,
    execution_horizon=10,
    inference_delay=2,
    control_period_s=0.02,
    prefix_attention_schedule="exp",
    max_guidance_weight=10.0,
)

try:
    while True:
        obs = get_observation()
        result = policy.infer(obs)
        execute_action(result["actions"])
finally:
    policy.close()
```

使用方式和普通 policy 类似，调用 `infer(obs)` 会返回当前 timestep 要执行的 action。不同之处是：RTC client 会在后台维护 action chunk，并异步请求下一段 chunk。

如果 server metadata 已经提供了 `action_horizon`、`execution_horizon`、`inference_delay`，client 侧可以省略这些参数。

## 4. 手动发送 RTC request

如果不使用 `RTCClientPolicy`，也可以手动向 RTC server 发送：

```python
request = {
    "obs": obs,
    "rtc_context": {
        "prev_action_chunk": prev_model_space_chunk,
        "inference_delay": 2,
        "execution_horizon": 10,
        "prefix_attention_horizon": 10,  # legacy alias, optional
        "prefix_attention_schedule": "exp",
        "max_guidance_weight": 10.0,
    },
}
```

注意：`prev_action_chunk` 默认必须使用 server 上一次返回的 `_rtc_model_actions`，不要使用 output transform 之后的 `actions`。如果你手动传入的是 absolute/action-space prefix，并且 policy 使用 delta/relative actions，可以同时传 `prev_action_chunk_space="absolute"` 或 `reanchor_relative_prefix=True`，必要时传 `relative_action_mask`，server 会参考 `DeltaActions`/`AbsoluteActions` 的 mask 语义把 prefix 重新锚定到当前 state。

## 5. 使用真实数据做 RTC 性能测试

如果要用真实数据测试 RTC，不要跑 `pytest` 的模拟时间轴测试，直接运行真实数据性能脚本：

1. 先启动 RTC server，例如你的 checkpoint 可以这样启动：

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py --server-mode rtc --debug-print --debug-interval 1 --port 7777 --rtc-execution-horizon 10 --rtc-inference-delay 3 --rtc-prefix-attention-schedule exp --rtc-max-guidance-weight 10.0 policy:checkpoint --policy.config=pi05_0403_cube_chunk30_30hz_lora_wx --policy.dir=./checkpoints/pi05_0403_cube_chunk30_30hz_lora_wx/pi05_0403_cube_chunk30_30hz_lora/29999
```

如果要换 GPU，把 `CUDA_VISIBLE_DEVICES=0` 改成对应卡号，例如 `CUDA_VISIBLE_DEVICES=1`。

2. 再用 `dataset_0403_cube` 的一帧 observation 跑真实 wall-clock 对比：

```bash
uv run scripts/rtc_real_performance.py --host localhost --port 7777 --data-dir ./dataset_0403_cube --episode 0 --frame 0 --prompt "perform task" --duration-s 10 --control-period-s 0.033333 --action-horizon 30 --execution-horizon 10 --inference-delay 3 --output-dir ./rtc_real_performance_outputs
```

这个脚本会：

- 从 `dataset_0403_cube/episode_0.hdf5` 读取真实 observation。
- 按真实 LIFT2 输入格式解码三路图像、构造 EEF state 和 prompt。
- 连接同一个 RTC server。
- 分别运行 non-RTC 同步推理和 RTC 后台推理；non-RTC 默认执行完整返回 chunk，例如 `chunk30` 会执行 30 个 action 后再请求下一段。
- 记录真实 wall-clock inference 时间、execution 时间和 execution gap。
- 生成：

```text
./rtc_real_performance_outputs/rtc_real_performance_timeline.json
./rtc_real_performance_outputs/rtc_real_performance_timeline.png
```

PNG 是多行时间轴，不再把所有 execution 混在同一行：

- 浅红色：普通 chunk inference。
- 红色：RTC-guided inference，也就是带 `rtc_context` 的下一段 chunk 推理。
- 蓝色：non-RTC 或 RTC initial chunk 的执行。
- 橙色：RTC 正在后台推理下一段时，前台继续复用/执行旧 chunk。
- 绿色：RTC 后台推理完成后，开始执行新安装的 RTC chunk。

这样可以直接看到：non-RTC 是“推理一段、执行一段、再推理”；RTC 是“执行当前 chunk 的同时后台推理下一段”，并且能区分当前执行的是旧 chunk 复用部分还是新 RTC chunk。

JSON 中每个 interval 也会带 metadata，例如：

- `chunk_id`：执行来自第几个 chunk。
- `chunk_step`：当前 action 在该 chunk 内的下标。
- `source`：`initial_chunk`、`reuse_old_chunk_while_inferring`、`new_rtc_chunk` 或 `new_non_rtc_chunk`。
- `request_type`：`chunk` 或 `rtc_guided`。

## 6. 可选单元测试

下面这些是开发验证用的单元测试，不是真实数据性能测试。

RTC client 和 prefix weight 测试：

```bash
uv run pytest \
  "packages/openpi-client/src/openpi_client/rtc_client_policy_test.py" \
  "src/openpi/models/model_test.py::test_pi0_prefix_weights"
```

模拟性能时间轴测试：

```bash
uv run pytest \
  "packages/openpi-client/src/openpi_client/rtc_performance_timeline_test.py"
```

该测试只模拟 10 秒执行过程，并生成 non-RTC 和 RTC 的时间轴：

- non-RTC：同步推理，推理期间动作执行会出现间隔。
- RTC：后台推理与动作执行重叠，warmup 后执行无间隔。

如需指定输出目录：

```bash
RTC_TIMELINE_OUTPUT_DIR=/tmp/rtc_timeline uv run pytest \
  "packages/openpi-client/src/openpi_client/rtc_performance_timeline_test.py"
```

输出文件：

```text
/tmp/rtc_timeline/rtc_timeline_comparison.json
/tmp/rtc_timeline/rtc_timeline_comparison.png
```

其中 PNG 图中：

- 红色表示 inference。
- 蓝色表示 action execution。

## 7. 当前限制

- 当前只实现 inference-time RTC，不需要微调。
- 当前主要支持 JAX `Pi0` policy。
- RTC client 依赖 server 返回 `_rtc_model_actions`，用于下一次 prefix 约束。
- `inference_delay` 使用固定配置，不用动态耗时估计作为算法输入。
- `scripts/rtc_real_performance.py` 是真实 wall-clock 测试，但仍然是在离线 dataset observation 上重复请求同一帧，不是真实机器人闭环控制。

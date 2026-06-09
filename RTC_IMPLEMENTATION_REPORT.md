# OpenPI RTC 实现与性能图表说明

本文用于汇报当前 OpenPI 中 RTC（Real-Time Chunking）的实现方式，以及性能图表中每个元素的含义。

## 1. RTC 要解决的问题

普通 chunk policy 一次推理返回一个完整动作块，例如长度为 `action_horizon` 的 action chunk。同步执行时，流程通常是：

1. 等模型推理完成；
2. 顺序执行返回的动作块；
3. 动作块用完后再次等待模型推理。

如果模型推理时间不可忽略，控制循环中会出现明显空档。RTC 的目标是：**在执行当前动作块的同时，后台异步推理下一个动作块**，从而把推理耗时隐藏在控制执行过程中。

核心思想：

- 当前 chunk 继续用于实时控制；
- 当执行到指定步数后，客户端启动后台请求；
- 后台请求会把当前 chunk 中尚未执行的动作作为 RTC prefix 发给模型；
- 模型生成下一个 chunk 时，被引导尽量和这些 prefix 动作保持连续；
- 新 chunk 返回后，根据后台推理期间已经执行了多少步，跳过新 chunk 前面已经“过期”的动作，只执行后面的动作。

## 2. 当前 RTC 的整体数据流

相关文件：

- 客户端异步调度：`packages/openpi-client/src/openpi_client/rtc_client_policy.py`
- RTC WebSocket 服务端：`src/openpi/serving/rtc_policy_server.py`
- 通用 policy 封装和 RTC 参数传递：`src/openpi/policies/policy.py`
- Pi0 模型内 RTC guidance：`src/openpi/models/pi0.py`
- 性能测试和画图：`scripts/rtc_real_performance.py`

整体流程如下：

```text
robot control loop
    |
    v
RTCClientPolicy.infer(obs)  每个控制 tick 调一次
    |
    |-- 没有 active chunk 时：同步请求第一个 chunk
    |
    |-- 有 active chunk 时：
    |      1. 如果后台结果已完成，安装新 chunk
    |      2. 如果到达 execution_horizon，启动后台 RTC 请求
    |      3. 从 active chunk 中切出当前 step 的单步动作返回
    |
    v
RTCPolicyServer
    |
    |-- 普通请求：policy.infer(obs)
    |-- RTC 请求：policy.infer(obs, rtc_context=..., return_model_actions=True)
    |
    v
Policy.infer
    |
    |-- 处理输入 transform
    |-- 解析 rtc_context
    |-- 调 Pi0.sample_actions(... RTC 参数 ...)
    |-- 输出 robot-space actions
    |-- 额外返回 _rtc_model_actions 作为下一次 RTC prefix
```

## 3. 客户端 RTC 调度逻辑

`RTCClientPolicy` 维护以下关键状态：

- `_active_result`：当前正在执行的模型返回结果。
- `_active_model_chunk`：当前 chunk 在模型空间中的动作，用于下一次 RTC prefix。
- `_active_step`：当前 chunk 已执行到第几步。
- `_pending_future`：后台推理任务。
- `_pending_request_step`：后台请求发起时 active chunk 的 step。
- `_next_request_step`：下一次允许发起后台请求的 step。

### 3.1 第一个 chunk

第一次调用 `infer(obs)` 时没有 active chunk，因此客户端同步请求一个普通 chunk：

```text
request = {obs}
response = {actions, _rtc_model_actions, ...}
```

安装后：

- `_active_result = response`
- `_active_model_chunk = response["_rtc_model_actions"]`
- `_active_step = 0`
- `_next_request_step = execution_horizon`

### 3.2 何时发起后台 RTC 请求

当满足以下条件时会发起后台请求：

- 当前没有 pending 请求；
- 已经有 active model chunk；
- `_active_step >= _next_request_step`。

默认 `execution_horizon = 10`，也就是说默认执行当前 chunk 的前 10 步后开始请求下一个 chunk。

后台请求发送的 `rtc_context` 主要包含：

```python
{
    "prev_action_chunk": shifted_prefix,
    "inference_delay": inference_delay,
    "execution_horizon": execution_horizon,
    "prefix_attention_horizon": execution_horizon,  # legacy alias
    "prefix_attention_schedule": "exp",
    "max_guidance_weight": 10.0,
}
```

其中 `shifted_prefix` 是把当前 `_active_model_chunk` 从 `_active_step` 开始左移后的结果。也就是：

```text
当前还没执行的动作 -> 放到 prefix 开头
剩余位置 -> 补 0
```

这样模型生成新 chunk 时，会用这些“当前 chunk 的未来动作”作为连续性约束。

### 3.3 后台推理期间如何控制

后台请求发出后，客户端不会等待它完成，而是继续从当前 active chunk 中取动作执行。

如果后台推理耗时为 3 个控制 tick，那么这 3 个 tick 仍然由旧 chunk 提供动作。这就是 RTC 的异步收益来源。

### 3.4 新 chunk 返回后如何安装

当 pending future 完成后：

```python
elapsed_steps = active_step - pending_request_step
install_chunk(result, active_step=elapsed_steps)
```

含义是：后台请求发出后，旧 chunk 已经额外执行了 `elapsed_steps` 步。因此新 chunk 的前 `elapsed_steps` 个动作已经过期，需要跳过，从新 chunk 的第 `elapsed_steps` 步开始执行。

这也是图 3 中 “discarded incoming action” 的来源。

## 4. 服务端 RTC 请求处理

`RTCPolicyServer` 支持两种请求：

1. 普通请求：直接传 observation；
2. RTC 请求：传 `{ "obs": observation, "rtc_context": {...} }`。

服务端会：

1. 解析请求；
2. 给缺失的 RTC 参数填默认值；
3. 调用：

```python
policy.infer(obs, rtc_context=rtc_context, return_model_actions=True)
```

返回中包含：

- `actions`：经过 output transform 后的 robot-space action chunk，用于实际控制；
- `_rtc_model_actions`：模型空间 action chunk，用于下一次 RTC prefix；
- `policy_timing` / `server_timing`：推理时间统计。

`_rtc_model_actions` 很重要：它避免把已经转换到 robot-space 的动作再反向猜回模型空间。客户端下一次 RTC prefix 直接使用模型空间 chunk，更稳定。

## 5. Policy 层如何传递 RTC 参数

`Policy.infer(..., rtc_context=...)` 做几件事：

1. 对 observation 做输入 transform；
2. 读取 `prev_action_chunk`；
3. 校验：

```text
0 <= inference_delay <= execution_horizon <= action_horizon
```

4. 把 RTC 参数传给 Pi0：

```python
sample_kwargs["rtc_prev_action_chunk"] = prev_action_chunk
sample_kwargs["rtc_inference_delay"] = inference_delay
sample_kwargs["rtc_execution_horizon"] = execution_horizon
sample_kwargs["rtc_prefix_attention_horizon"] = execution_horizon
sample_kwargs["rtc_prefix_attention_schedule"] = schedule
sample_kwargs["rtc_max_guidance_weight"] = max_guidance_weight
```

### 5.1 relative action re-anchor

如果某些 action 维度使用 delta / relative action 表示，RTC prefix 也必须进入同样的动作空间。当前实现提供：

```python
reanchor_relative_rtc_prefix(prev_actions_absolute, current_state, mask)
```

它会把被 mask 选中的 action 维度转换为：

```text
relative_action = absolute_action - current_state
```

触发条件：

- `rtc_context["prev_action_chunk_space"] == "absolute"`；或
- `rtc_context["reanchor_relative_prefix"] == True`；或
- 显式传入 `relative_action_mask`。

当前标准 RTCClientPolicy 通常使用服务端返回的 `_rtc_model_actions`，它已经在模型空间中，因此一般不需要 re-anchor；该机制主要用于兼容 absolute prefix 输入。

## 6. Pi0 模型内 RTC guidance

Pi0 的普通采样是 diffusion / flow matching 形式，从噪声开始迭代去噪得到 action chunk。

### 6.1 普通推理

普通推理流程：

1. 预处理 observation；
2. 生成初始 noise；
3. 对图像和语言 prefix 做一次 forward，得到 LLM KV cache；
4. 每个 denoising step 只计算 suffix，也就是 state/action/time token；
5. 得到 velocity `v_t`；
6. 更新：

```python
x_t = x_t + dt * v_t
```

这里 `dt < 0`，时间从 `t=1` 走到 `t=0`。

### 6.2 RTC prefix weights

RTC 会根据：

- `inference_delay`
- `execution_horizon`
- `action_horizon`
- `prefix_attention_schedule`

生成每个 action step 的权重。

语义：

- `inference_delay`：硬 prefix 区域的结束位置，可以理解为已经确定会被旧 chunk 覆盖的 delay 步数；
- `execution_horizon`：RTC prefix 影响范围的终点；
- `execution_horizon` 之后权重为 0；
- `prefix_attention_horizon` 是旧参数名，现在和 `execution_horizon` 语义对齐。

schedule：

- `ones`：prefix 范围内权重全 1；
- `zeros`：只保留硬 prefix；
- `linear`：从硬 prefix 后开始线性衰减；
- `exp`：指数式衰减，当前默认。

### 6.3 RTC correction

当 `rtc_prev_action_chunk` 存在时，Pi0 在每个 denoising step 里做 RTC guidance：

```python
v_t = velocity(x_t, time)
x_0 = x_t - time * stop_gradient(v_t)
correction = (rtc_prev_action_chunk - x_0) * prefix_weights
v_t = v_t - guidance_weight * correction
```

解释：

- `x_0` 是当前 denoising step 对最终 action chunk 的估计；
- `rtc_prev_action_chunk` 是希望新 chunk 在 prefix 区域贴近的目标；
- `correction` 会推动生成结果靠近旧 chunk 的剩余动作；
- `prefix_weights` 控制哪些 action step 受影响、影响多强；
- `guidance_weight` 随时间变化，并被 `max_guidance_weight` 截断。

当前默认：

```text
max_guidance_weight = 10.0
execution_horizon = 10
prefix_attention_schedule = "exp"
```

重要优化：当前实现复用普通采样中的 cached `velocity()`，并对 `v_t` 使用 `stop_gradient`。因此 RTC correction 是轻量的 action-space residual，不再对整个大模型做 VJP / 反向传播。这一点是当前 RTC 推理速度恢复正常的关键。

## 7. 性能测试输出文件

运行 `scripts/rtc_real_performance.py` 后会输出：

```text
rtc_real_performance_timeline.json
rtc_real_performance_timeline.png
rtc_real_performance_chunk_fate_timeline.png
```

### 7.1 JSON interval 类型

JSON 中的每个事件是一个 interval：

```json
{
  "mode": "rtc",
  "kind": "execution",
  "label": "chunk0_action3",
  "start_s": 0.150,
  "end_s": 0.200,
  "duration_s": 0.050,
  "metadata": {...}
}
```

`kind` 的含义：

| kind | 含义 |
|---|---|
| `inference` | 一次模型请求的耗时区间 |
| `execution` | 一个控制 tick 中实际执行动作的时间区间 |
| `conditioning` | 某个 chunk 的一段动作被用作下一次 RTC prefix |
| `discarded` | 新返回 chunk 的前若干动作因为后台推理期间已由旧 chunk 执行，所以被丢弃 |

常用 metadata：

| 字段 | 含义 |
|---|---|
| `chunk_id` | 第几个返回的 action chunk |
| `chunk_step` | 当前 action 在该 chunk 内的 step |
| `chunk_horizon` | chunk 长度 |
| `global_action_index` | 全局控制步编号 |
| `source` | 动作来源，例如 initial chunk、新 RTC chunk、复用旧 chunk |
| `inference_pending` | 当前动作执行时是否有后台推理正在运行 |
| `reuse_while_inferring` | 是否是在等待新 chunk 时复用旧 chunk 的动作 |
| `discarded_prefix_steps` | 新 chunk 安装时被跳过的前缀步数 |
| `start_step` / `end_step` | conditioning prefix 的 step 范围 |

## 8. 主图 `rtc_real_performance_timeline.png`

主图用于回答：**RTC 是否把推理隐藏到了执行过程中？是否减少了控制空档？**

它包含两个部分。

### 8.1 Summary table

表格指标：

| 指标 | 含义 | 如何解读 |
|---|---|---|
| `Inference p50` | 推理耗时中位数 | 单次请求延迟；RTC 不应显著更慢 |
| `Max execution gap` | 相邻两个 execution interval 之间最大空档 | 越小越好，是控制连续性的核心指标 |
| `Actions executed` | benchmark 内实际执行动作数量 | 越接近理想控制频率越好 |
| `Actions during pending inference` | 后台推理进行时仍然执行的动作数 | RTC 的核心收益，越多说明越能隐藏推理 |
| `Repeated last action` | chunk 用尽后重复最后一个动作的次数 | 理想情况下为 0，表示没有 queue underrun |

### 8.2 Timeline overview

颜色和 lane：

| 图中元素 | 含义 |
|---|---|
| 红色条 | inference，请求模型生成 chunk 的时间 |
| 蓝色条 | non-RTC execution |
| 绿色条 | RTC execution |
| 橙色半透明区域 | execution gap，即控制空档 |

四条 lane：

1. `Non-RTC inference`
2. `Non-RTC execution`
3. `RTC inference`
4. `RTC execution`

理想结果：

- non-RTC 中，红色 inference 往往会阻塞 execution，导致橙色 gap；
- RTC 中，红色 inference 应该和绿色 execution 重叠；
- RTC 的绿色 execution 应该更连续，橙色 gap 更少或更短。

## 9. 图 3 `rtc_real_performance_chunk_fate_timeline.png`

图 3 用于回答：**每个 RTC 返回的 action chunk 中，哪些动作被执行、哪些被复用、哪些被丢弃、哪些参与了 RTC prefix？**

当前图是 16:9 布局，和主图使用同一个时间轴。

### 9.1 上方 inference lane

上方单独一条 lane：

| 元素 | 含义 |
|---|---|
| 红色条 | RTC 后台 inference 的时间区间 |

这条 lane 用来直观看出后台推理什么时候开始、什么时候结束。

### 9.2 下方 chunk fate rows

下方每一行表示一个返回的 action chunk：

```text
chunk 0
h=50
```

其中：

- `chunk 0`：第 0 个返回的 chunk；
- `h=50`：该 chunk 的 action horizon 为 50。

图中没有给每个小 bar 都标 `a0/a1/...`，这是为了避免文字重叠。具体 action index 可从 JSON 的 `chunk_step` 查看。

### 9.3 图 3 颜色和符号

| 元素 | 含义 |
|---|---|
| 绿色条 | 实际发送给控制器执行的 action |
| 绿色条上的橙色斜线 | 等待下一次后台推理时，复用了旧 chunk 的 action |
| 灰色斜线条 | 新返回 chunk 中被丢弃的 incoming action |
| 紫色菱形 / 短竖线 | 该 chunk 的动作被用作下一次 RTC prefix conditioning |
| 白色背景行 | 一个完整 returned chunk 的时间行 |

### 9.4 如何读图 3

典型 RTC 交接过程如下：

1. 当前 chunk 执行到 `execution_horizon`；
2. 客户端发起后台 RTC 请求；
3. 后台推理期间，旧 chunk 继续执行，这些动作显示为绿色条，且带橙色斜线；
4. 新 chunk 返回；
5. 新 chunk 前面与旧 chunk 已执行时间重叠的动作被跳过，显示为灰色斜线；
6. 新 chunk 从未过期的位置开始执行，显示为绿色条。

因此：

- 橙色斜线越多，说明后台推理期间旧 chunk 被有效复用；
- 灰色斜线表示异步推理带来的正常丢弃，不一定是坏事；
- 关键是绿色 execution 是否连续，以及是否避免 repeated last action；
- 紫色 marker 表示这段旧动作参与了下一次 chunk 的连续性约束。

## 10. 当前实现的关键默认值

| 参数 | 当前默认值 | 含义 |
|---|---:|---|
| `execution_horizon` | `10` | 执行到第几步后发起下一次 RTC 请求；也是 prefix guidance 的影响终点 |
| `inference_delay` | 服务端默认 `0`，性能脚本常用配置值 | 预计/配置的硬 delay 步数，必须不超过 `execution_horizon` |
| `prefix_attention_schedule` | `exp` | prefix 权重衰减方式 |
| `max_guidance_weight` | `10.0` | RTC guidance 最大强度 |
| `control_period_s` | 性能脚本默认 `0.05s` | 控制 tick 周期 |

## 11. 汇报时可以强调的结论

1. 当前 RTC 是完整的端到端实现：客户端异步调度、服务端 RTC request、Policy 参数传递、Pi0 内部 guidance、性能可视化都已打通。
2. 语义已和 LeRobot 风格对齐：`execution_horizon` 是 RTC prefix guidance 的主要 horizon，`prefix_attention_horizon` 只是兼容旧字段。
3. Pi0 内部 RTC correction 已从昂贵的 full-model VJP 改为轻量 action-space residual，并复用 KV cache，所以推理速度不会因为 RTC 明显变慢。
4. 主图看整体实时性：推理是否被 execution 覆盖、execution gap 是否减少。
5. 图 3 看每个 RTC chunk 的命运：执行、等待时复用、返回后丢弃、作为 prefix conditioning。
6. 判断 RTC 效果时不要只看单次 inference latency，更重要的是 `Max execution gap`、`Actions during pending inference` 和图 3 中 chunk 交接是否连续。

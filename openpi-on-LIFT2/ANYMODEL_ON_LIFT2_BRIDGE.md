# AnyModel-on-LIFT2：LIFT2 真机对接手册（openpi 仓库 + 本文档）

## 文档目标与前提

**目标**：在 **仅拥有本 openpi 单仓库**（含 `src/openpi`、`scripts/serve_policy.py`、`openpi-on-LIFT2/`）的前提下，完成 **π₀.₅ LIFT2 checkpoint → GPU 策略服务 → LIFT2 真机 ARX R5 双臂** 的闭环部署。

**两台机器分工**：

| 机器 | 仓库内容 | 运行什么 |
|------|----------|----------|
| **训练机 / GPU 服务器** | 本仓库 `openpi` 根目录 | `uv run scripts/serve_policy.py`（加载你训练好的 LIFT2 checkpoint） |
| **LIFT2 机载电脑** | 拷贝 **`openpi-on-LIFT2/`** 文件夹（或按本文 §1–§7 自写客户端） | `python deploy/client_lift2.py` + ROS |

二者通过局域网 **WebSocket + msgpack_numpy** 通信（默认端口 **7777**，见 `launch_profiles.yaml`）。

**本文档中的 Python 代码块**：实现机器人侧 **策略客户端 + 14D EEF 桥接**；**不替代** openpi 内的模型推理（须在 GPU 机用 `serve_policy.py`）。

**训练与数据**：LIFT2 HDF5 → LeRobot、`norm_stats`、config 名称与 `action_horizon` 等，见 [LIFT2_GUIDE.md](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/LIFT2_GUIDE.md)。

---

## 0. 全链路数据契约（必读）

### 0.1 维度与语义（LIFT2 EEF 14D）

与 [LIFT2_GUIDE.md](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/LIFT2_GUIDE.md) / `convert_hdf5_to_lerobot_eef.py` 一致：

```text
index:  0-2      3-5        6        7-9      10-12      13
        L_xyz    L_rpy   L_grip    R_xyz    R_rpy   R_grip
```

| 场 | 单位 / 范围 | 说明 |
|----|-------------|------|
| xyz | 米，绝对 | 末端位置 |
| rpy | 弧度，绝对 | roll/pitch/yaw |
| grip（进策略前） | **[0, 1]** | 由真机 **[0, 5]** 线性归一化（0 闭合，5 张开） |
| grip（下发真机） | **[0, 5]** | 客户端 `postprocess_gripper` 二值化或反归一化 |

**策略 `actions`（服务端 → 客户端）**：对 LIFT2 EEF 训练数据，LeRobot 里 **action 为 delta**（连续帧差分）；推理时服务端 `Policy.infer` 经 `Unnormalize` 后，客户端收到的 `actions` 应为：

| 下标 | 语义 |
|------|------|
| 0–2, 7–9 | **Δxyz**（米），相对**预测链**上一步，不是相对真机延迟观测 |
| 3–5, 10–12 | **Δrpy**（弧度） |
| 6, 13 | **绝对**夹爪命令，**归一化 [0, 1]**（不加到当前夹爪） |

客户端必须用 **`apply_eef_delta` 沿 chunk 累积**（standard）或 **`rtc_pred_eef` 链**（RTC），再转真机绝对 `PosCmd`。

**禁止**：每步用真机当前 state 重锚定 xyz/rpy delta（会与训练 `action[t]=state[t+1]-state[t]` 不一致）。

### 0.2 模块 I/O 总表

| 模块 | 输入（期待） | 输出（产生） | 运行位置 |
|------|----------------|--------------|----------|
| **ROS `RosOperator`** | 话题：三路 `sensor_msgs/Image`、双臂 `arm_control/PosCmd` 状态 | BGR `uint8` 图 + `PosCmd`；发布 `PosCmd` 命令 | 机器人 |
| **`build_openpi_observation`** | 三路图（任意分辨率）、`state` float32**(14,) 物理量纲**、`prompt` str | dict（见 §0.3） | 机器人 |
| **`WebsocketClientPolicy`** | 上表 dict（**扁平**，无 `obs` 包装） | dict，**必含** `actions` **(H, 14)**；可选 `policy_timing`、`_timing` | 机器人 |
| **`RTCClientPolicy.infer`** | 同左，每 tick 一次 | dict，**`actions` 为 (14,) 单步**；后台仍收 **(H, 14)** chunk | 机器人 |
| **`RTCPolicyServer` / `Policy.infer`** | `{"obs":...,"rtc_context":...}` 或扁平 obs | **`actions` (H,14)** + **`_rtc_model_actions` (H,14)** model space | GPU |
| **`scripts/serve_policy.py`** | checkpoint 路径 + **与训练一致的** `--policy.config` | 监听 WS，`metadata` 首包 | GPU |
| **`Lift2PolicyBridge.control_step`** | `frame` 或 `None`（standard 队列非空） | **绝对 14D**，grip 已为 **[0,5]** | 机器人 |

默认 ROS 话题（`client_lift2.py`）：`/camera_h|l|r/color/image_raw`，`/arm_left|right/arm_status_ee`，发布 `/arm_left_cmd`、`/arm_right_cmd`。

### 0.3 WebSocket 载荷（与 `lift2_policy.py` 一致）

**连接**：`ws://<server_ip>:<port>`，`compression=None`。Server **第一条消息** = `metadata`（msgpack）。

**非 RTC — Client → Server（整条 msgpack dict）**：

客户端发送的是 **物理量纲、未做 norm_stats 的 state**（与 `make_lift2_example()` / 真机 `pose_to_eef` 一致）。`Normalize` 在 **GPU 端 `Policy.infer` 输入链**内完成（`create_trained_policy` → `transforms.Normalize(norm_stats)`），机器人 **不要**自行减均值除方差。

```python
{
  "observation.images.head": uint8[224, 224, 3],
  "observation.images.left_wrist": uint8[224, 224, 3],
  "observation.images.right_wrist": uint8[224, 224, 3],
  "observation.state": float32[14],   # 物理单位；grip 已 [0,1]
  "prompt": str,                      # 与训练 default_prompt / 任务一致
}
```

Server 侧：`Lift2Inputs` → `Normalize(norm_stats)` → 模型 → `Unnormalize` → `Lift2Outputs` 截取 **14 维**。

**非 RTC — Server → Client**：

```python
{
  "actions": float32[action_horizon, 14],  # 见 §0.1 delta 语义；H 来自训练 config
  "state": float32[14],                    # 可选，调试
  "policy_timing": {"infer_ms": float},
  "_timing" / "server_timing": dict,       # 可选
}
```

**RTC — Client → Server**：

- 首包：`{"obs": <与上相同的 dict>}`
- 后续：`{"obs": ..., "rtc_context": {"prev_action_chunk": float32[H,14], "inference_delay": int, "execution_horizon": int, ...}}`
  - `prev_action_chunk` **必须**来自上一响应的 **`_rtc_model_actions`**（不是 `actions`）

**RTC — Server → Client**：在非 RTC 字段基础上 **必须**含 `_rtc_model_actions`: `float32[H, 14]`（output transform **之前**的 model chunk）。

**约束**：`0 <= inference_delay <= execution_horizon <= action_horizon`。

**RTC 模型限制**（当前仓库）：仅 **JAX π₀（Pi0）**；`serve_policy.py --server-mode rtc` 与 checkpoint 的 `action_horizon` 须与客户端 `rtc_action_horizon` / `launch_profiles.yaml` 一致。

### 0.4 `metadata` 首包（Server → Client）

由 `policy.metadata` + RTC server 注入，客户端可读：

```python
{
  "action_horizon": 30,   # 建议与训练 Pi0Config.action_horizon 一致
  "rtc": {               # 仅 --server-mode rtc
    "execution_horizon": 10,
    "inference_delay": 4,
    "prefix_attention_schedule": "exp",
    "max_guidance_weight": 10.0,
  },
}
```

### 0.5 训练 config 与部署参数对齐（示例）

| 训练 `TrainConfig.name` | `action_horizon` | 客户端 `execute_horizon` / `action_chunk_size` | 控制频率 |
|-------------------------|------------------|-----------------------------------------------|----------|
| `pi05_*_chunk30_*` | 30 | 30 / 30 | 30 Hz |
| `pi05_*_chunk10_*` | 10 | 10 / 10 | 30 Hz |

以你 checkpoint 对应的 **`config.py` 里 `Pi0Config(action_horizon=...)`** 为准；`launch_profiles.yaml` 的 `default` / `rtc` 仅为本仓库预设。

---

## 1. msgpack + NumPy 线协议

### 输入 / 输出

- **输入**：任意可嵌套 `dict`，值含 `np.ndarray` / 标量。
- **输出**：`bytes`（msgpack）；反序列化后恢复 ndarray。
- **不负责**：业务字段名；只保证与 openpi `openpi_client.msgpack_numpy` 兼容。

### 具体实现

```python
def _pack_array(obj):
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")
    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(
            buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"]
        )
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array)
packb = functools.partial(msgpack.packb, default=_pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)
```

### 抽象实现

**ModelCodec**：`pack(dict)->bytes`、`unpack(bytes)->dict`；嵌套 `ndarray`；禁止 pickle。

### 本仓库源码位置

- [openpi-on-LIFT2/openpi_client/msgpack_numpy.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/openpi-on-LIFT2/openpi_client/msgpack_numpy.py)

---

## 2. 图像预处理（224 pad-resize）

### 输入 / 输出

- **输入**：`uint8` 或 `float` 的 `H×W×3`（ROS 常为 BGR `uint8`）。
- **输出**：`uint8[224,224,3]`，等比缩放 + 黑边 pad（与 `ModelTransformFactory` 中 `ResizeImages(224,224)` 一致）。
- **注意**：Server 端仍会再做 transform；客户端提前 resize 可降低带宽，**必须与训练一致 224**。

### 具体实现

```python
def convert_to_uint8(img: np.ndarray) -> np.ndarray:
    if np.issubdtype(img.dtype, np.floating):
        img = (255 * img).astype(np.uint8)
    return img


def _resize_with_pad_pil(image: Image.Image, height: int, width: int, method=Image.BILINEAR) -> Image.Image:
    cur_width, cur_height = image.size
    if cur_width == width and cur_height == height:
        return image
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_image = image.resize((resized_width, resized_height), resample=method)
    zero_image = Image.new(resized_image.mode, (width, height), 0)
    pad_height = max(0, int((height - resized_height) / 2))
    pad_width = max(0, int((width - resized_width) / 2))
    zero_image.paste(resized_image, (pad_width, pad_height))
    return zero_image


def resize_with_pad(images: np.ndarray, height: int, width: int, method=Image.BILINEAR) -> np.ndarray:
    if images.shape[-3:-1] == (height, width):
        return images
    original_shape = images.shape
    images = images.reshape(-1, *original_shape[-3:])
    resized = np.stack(
        [_resize_with_pad_pil(Image.fromarray(im), height, width, method=method) for im in images]
    )
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])
```

### 抽象实现

**ImagePreprocessor**：原图 → `uint8[224,224,3]`，与训练一致。

### 本仓库源码位置

- [openpi-on-LIFT2/openpi_client/image_tools.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/openpi-on-LIFT2/openpi_client/image_tools.py)

---

## 3. LIFT2 十四维 EEF 几何

### 输入 / 输出

- **`pose_arrays_to_eef`**：输入左右 `PosCmd` 七元组（grip 原始 0–5）→ 输出 **策略 state** 14D（grip 0–1）。
- **`apply_eef_delta`**：输入当前链上 14D + **单步 delta** 14D → 输出下一绝对 14D（grip 维替换，不叠加）。
- **真机 `PosCmd`**：输出 14D 中 grip 已为 0–5（经 `postprocess_gripper`）。

### 具体实现

```python
GRIPPER_MIN = 0.0
GRIPPER_MAX = 5.0
ACTION_DIM = 14


def normalize_gripper(gripper_raw: float) -> float:
    return float(np.clip(gripper_raw, GRIPPER_MIN, GRIPPER_MAX) / GRIPPER_MAX)


def denormalize_gripper(gripper_norm: float) -> float:
    return float(np.clip(gripper_norm, 0.0, 1.0) * GRIPPER_MAX)


def pose_arrays_to_eef(left_pose7: np.ndarray, right_pose7: np.ndarray) -> np.ndarray:
    """Each pose7: [x,y,z,roll,pitch,yaw,gripper_raw_0_5]. Returns 14D with norm grippers."""
    left_pose7 = np.asarray(left_pose7, dtype=np.float32)
    right_pose7 = np.asarray(right_pose7, dtype=np.float32)
    out = np.zeros(ACTION_DIM, dtype=np.float32)
    out[0:3] = left_pose7[0:3]
    out[3:6] = left_pose7[3:6]
    out[6] = normalize_gripper(left_pose7[6])
    out[7:10] = right_pose7[0:3]
    out[10:13] = right_pose7[3:6]
    out[13] = normalize_gripper(right_pose7[6])
    return out


def apply_eef_delta(current_eef: np.ndarray, delta_eef: np.ndarray) -> np.ndarray:
    """xyz/rpy additive on current; gripper indices absolute normalized."""
    current_eef = np.asarray(current_eef, dtype=np.float32)
    delta_eef = np.asarray(delta_eef, dtype=np.float32)
    next_eef = np.zeros(ACTION_DIM, dtype=np.float32)
    next_eef[0:3] = current_eef[0:3] + delta_eef[0:3]
    next_eef[3:6] = current_eef[3:6] + delta_eef[3:6]
    next_eef[6] = delta_eef[6]
    next_eef[7:10] = current_eef[7:10] + delta_eef[7:10]
    next_eef[10:13] = current_eef[10:13] + delta_eef[10:13]
    next_eef[13] = delta_eef[13]
    return next_eef


def split_abs_eef_to_arms(abs_14d: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    abs_14d = np.asarray(abs_14d, dtype=np.float32)
    return abs_14d[:7].copy(), abs_14d[7:14].copy()
```

### 抽象实现

14D state/command；delta 时 xyz/rpy 沿**预测链**累积，grip 为绝对归一化；真机 grip `[0,5]`。

### 本仓库源码位置

- [openpi-on-LIFT2/deploy/utils/rotation.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/openpi-on-LIFT2/deploy/utils/rotation.py)

---

## 4. Observation 组包与 Delta 后处理

### 输入 / 输出

- **`build_openpi_observation`**  
  - 入：三路 RGB、`state_14d`（物理）、`prompt`  
  - 出：§0.3 WebSocket **扁平** observation（键名固定为 `observation.images.*`）。
- **`sanitize_*` / `deltas_to_absolute_trajectory`**  
  - 入：服务端 `actions` 一行或整 chunk（delta）  
  - 出：绝对 14D 轨迹（grip 仍归一化，下发前再 `postprocess_gripper`）。

### 具体实现

```python
DEFAULT_IMAGE_KEYS = (
    "observation.images.head",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)
DEFAULT_STATE_KEY = "observation.state"
DEFAULT_PROMPT_KEY = "prompt"


def build_openpi_observation(
    *,
    head_rgb: np.ndarray,
    left_wrist_rgb: np.ndarray,
    right_wrist_rgb: np.ndarray,
    state_14d: np.ndarray,
    prompt: str,
    image_size: int = 224,
) -> Dict[str, Any]:
    state_14d = np.asarray(state_14d, dtype=np.float32)
    if not np.all(np.isfinite(state_14d)):
        raise ValueError(f"state contains NaN/Inf: {state_14d}")
    images = [head_rgb, left_wrist_rgb, right_wrist_rgb]
    keys = DEFAULT_IMAGE_KEYS
    observation: Dict[str, Any] = {}
    for key, img in zip(keys, images):
        observation[key] = convert_to_uint8(resize_with_pad(np.asarray(img), image_size, image_size))
    observation[DEFAULT_STATE_KEY] = state_14d
    observation[DEFAULT_PROMPT_KEY] = prompt
    return observation


def sanitize_delta_action(
    delta_action: np.ndarray,
    *,
    max_delta_xyz: float = 0.05,
    max_delta_rpy: float = 0.2,
) -> np.ndarray:
    delta_action = np.asarray(delta_action, dtype=np.float32).copy()
    if delta_action.shape != (ACTION_DIM,):
        raise ValueError(f"expected delta shape ({ACTION_DIM},), got {delta_action.shape}")
    if not np.all(np.isfinite(delta_action)):
        raise ValueError(f"delta contains NaN/Inf: {delta_action}")
    delta_action[0:3] = np.clip(delta_action[0:3], -max_delta_xyz, max_delta_xyz)
    delta_action[3:6] = np.clip(delta_action[3:6], -max_delta_rpy, max_delta_rpy)
    delta_action[7:10] = np.clip(delta_action[7:10], -max_delta_xyz, max_delta_xyz)
    delta_action[10:13] = np.clip(delta_action[10:13], -max_delta_rpy, max_delta_rpy)
    return delta_action


def smooth_delta_action(delta_action: np.ndarray, *, smooth_alpha: float = 1.0) -> np.ndarray:
    alpha = float(smooth_alpha)
    if alpha >= 1.0:
        return delta_action
    smoothed = delta_action.copy()
    smoothed[0:3] *= alpha
    smoothed[3:6] *= alpha
    smoothed[7:10] *= alpha
    smoothed[10:13] *= alpha
    smoothed[6] = delta_action[6]
    smoothed[13] = delta_action[13]
    return smoothed


def postprocess_gripper_for_robot(
    abs_eef_14d: np.ndarray,
    *,
    binarize: bool = True,
    threshold: float = 0.6,
    closed_val: float = 0.5,
    open_val: float = 4.9,
) -> np.ndarray:
    out = np.asarray(abs_eef_14d, dtype=np.float32).copy()
    if binarize:
        out[6] = open_val if out[6] >= threshold else closed_val
        out[13] = open_val if out[13] >= threshold else closed_val
    else:
        out[6] = denormalize_gripper(out[6])
        out[13] = denormalize_gripper(out[13])
    return out


def deltas_to_absolute_trajectory(base_eef: np.ndarray, delta_chunk: np.ndarray, cfg: "Lift2BridgeConfig") -> List[np.ndarray]:
    pred = np.asarray(base_eef, dtype=np.float32).copy()
    trajectory: List[np.ndarray] = []
    for delta in delta_chunk:
        d = smooth_delta_action(
            sanitize_delta_action(delta, max_delta_xyz=cfg.max_delta_xyz, max_delta_rpy=cfg.max_delta_rpy),
            smooth_alpha=cfg.smooth_alpha,
        )
        pred = apply_eef_delta(pred, d)
        trajectory.append(pred.copy())
    return trajectory
```

### 抽象实现

**ObservationBuilder** / **ActionPostprocessor**：键名对齐训练 config；sanitize→smooth→累积→gripper。

### 本仓库源码位置

- [openpi-on-LIFT2/deploy/client_lift2.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/openpi-on-LIFT2/deploy/client_lift2.py)

---

## 4B. 拼装 §5–§7 所需的公共依赖

从本文 **复制 §1 的 `Packer`/`unpackb`** 后，再复制下列代码；**机器人上也可直接使用** `openpi-on-LIFT2/openpi_client/`（与 `packages/openpi-client` 同构），无需手拼。

### 输入 / 输出

- 无运行时 I/O；提供 `BasePolicy`、WebSocket URI 与连接 helper，供 `WebsocketClientPolicy` / `RTCClientPolicy` 继承。

### 具体实现

```python
import abc
import logging
import time
from typing import Dict, Optional

import websockets.sync.client as ws_sync

# 需已定义：Packer, unpackb（§1）


class BasePolicy(abc.ABC):
    @abc.abstractmethod
    def infer(self, obs: Dict) -> Dict:
        raise NotImplementedError

    def reset(self) -> None:
        pass

    def close(self) -> None:
        pass


def _ws_connect(uri: str, headers: Optional[dict]):
    try:
        return ws_sync.connect(uri, compression=None, max_size=None, additional_headers=headers)
    except TypeError:
        return ws_sync.connect(uri, compression=None, max_size=None, extra_headers=headers)


def _make_uri(host: str, port: Optional[int]) -> str:
    if host.startswith("ws"):
        uri = host
    else:
        uri = f"ws://{host}"
    if port is not None:
        uri += f":{port}"
    return uri
```

### 抽象实现

**Transport**：同步 WS，一发一收；RTC 在单连接上用锁 + 后台线程发第二路请求。

### 本仓库源码位置

- [openpi-on-LIFT2/openpi_client/base_policy.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/openpi-on-LIFT2/openpi_client/base_policy.py)
- [openpi-on-LIFT2/openpi_client/websocket_client_policy.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/openpi-on-LIFT2/openpi_client/websocket_client_policy.py)（`_connect` 同 `_ws_connect`）

---

## 5. 非 RTC：`WebsocketClientPolicy`

### 输入 / 输出

- **构造**：`host`/`port` → 连接后缓存 `metadata`。
- **`infer(obs)`**  
  - 入：§0.3 扁平 dict  
  - 出：完整响应 dict；**`result["actions"].shape == (H, 14)`**。
- **不负责**：拆 chunk；由 `Lift2PolicyBridge` 队列 + `execute_horizon` 消费。

### 具体实现

```python
class WebsocketClientPolicy(BasePolicy):
    """Non-RTC: one infer() = one synchronous round-trip; returns full chunk in result['actions']."""

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        self._uri = _make_uri(host, port)
        self._packer = Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self):
        logging.info("Waiting for policy server at %s...", self._uri)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = _ws_connect(self._uri, headers)
                metadata = unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting...")
                time.sleep(5)

    def infer(self, obs: Dict) -> Dict:
        self._ws.send(self._packer.pack(obs))
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Server error:\n{response}")
        return unpackb(response)

    def close(self) -> None:
        self._ws.close()


```

### 抽象实现

**SyncChunkClient**：单次 RTT 整 chunk；`action_plan` + `execute_horizon` 在 Bridge。  
**接入其他 VLA**：仅当该模型在本仓库内也有 **相同 §0.3 WS 键名与 (H,14) delta 语义** 的 `serve_policy` 实现；否则应改 `build_openpi_observation` 键名或单独写 Server transform，**不能**只改端口。

### 本仓库源码位置

- [openpi-on-LIFT2/openpi_client/websocket_client_policy.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/openpi-on-LIFT2/openpi_client/websocket_client_policy.py)
- [src/openpi/serving/websocket_policy_server.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/src/openpi/serving/websocket_policy_server.py)

---

## 6. RTC：`RTCClientPolicy`

### 输入 / 输出

- **`infer(obs)` 每控制 tick**  
  - 入：§0.3 的 `obs`（包在内部 `{{"obs": obs}}` 发送）  
  - 出：与完整响应相同 key，但 **`actions` 为 (14,) 单步**（类内 slice）。
- **内部请求 Server**  
  - 首次：`{{"obs": obs}}`  
  - 重叠：`{{"obs", "rtc_context"}}`，`prev_action_chunk` shape **(H, 14)**。
- **要求 Server**：每次 chunk 响应含 **`_rtc_model_actions`**，与 `actions` 同 horizon。

### 具体实现

```python
class RTCClientPolicy(BasePolicy):
    """RTC: infer(obs) returns one control-step slice; schedules background chunk requests."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        api_key: Optional[str] = None,
        *,
        action_horizon: Optional[int] = None,
        execution_horizon: Optional[int] = 10,
        inference_delay: Optional[int] = None,
        control_period_s: float = 1.0 / 30.0,
        prefix_attention_schedule: str = "exp",
        max_guidance_weight: float = 10.0,
        request_fn: Optional[Callable[[Dict], Dict]] = None,
    ) -> None:
        self._packer = Packer()
        self._api_key = api_key
        self._request_fn = request_fn
        self._ws = None
        if request_fn is None:
            self._uri = _make_uri(host, port)
            self._ws, self._server_metadata = self._wait_for_server()
        else:
            self._uri = "local://rtc-request-fn"
            self._server_metadata = {}

        rtc_md = self._server_metadata.get("rtc", {}) if isinstance(self._server_metadata, dict) else {}
        md_h = self._server_metadata.get("action_horizon") if isinstance(self._server_metadata, dict) else None
        self._action_horizon = action_horizon or md_h
        self._execution_horizon = (
            execution_horizon if execution_horizon is not None else rtc_md.get("execution_horizon", 10)
        )
        self._inference_delay = int(
            inference_delay if inference_delay is not None else rtc_md.get("inference_delay", 0)
        )
        self._control_period_s = control_period_s
        self._prefix_attention_schedule = prefix_attention_schedule or rtc_md.get("prefix_attention_schedule", "exp")
        self._max_guidance_weight = float(
            max_guidance_weight if max_guidance_weight is not None else rtc_md.get("max_guidance_weight", 10.0)
        )
        self._validate_rtc_config()

        self._active_result: Optional[Dict] = None
        self._active_model_chunk: Optional[np.ndarray] = None
        self._active_step = 0
        self._next_request_step = 0
        self._estimated_delay_steps = 0
        self._pending_future = None
        self._pending_request_step = 0
        self._pending_request_time = 0.0
        self._lock = threading.RLock()
        self._ws_lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def get_estimated_delay_steps(self) -> int:
        return self._estimated_delay_steps

    def _wait_for_server(self):
        logging.info("Waiting for RTC server at %s...", self._uri)
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = _ws_connect(self._uri, headers)
                return conn, unpackb(conn.recv())
            except ConnectionRefusedError:
                time.sleep(5)

    def infer(self, obs: Dict) -> Dict:
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
            raise RuntimeError(f"RTC server error:\n{response}")
        return unpackb(response)

    def _maybe_schedule_request_locked(self, obs: Dict) -> None:
        if self._pending_future is not None or self._active_model_chunk is None:
            return
        if self._action_horizon is None:
            return
        self._validate_rtc_config()
        if self._active_step < self._next_request_step:
            return
        shifted = self._shift_chunk_left(self._active_model_chunk, self._active_step)
        rtc_context = {
            "prev_action_chunk": shifted,
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
            logger.exception("RTC background inference failed")
            self._next_request_step = self._active_step
            return
        elapsed_steps = max(0, self._active_step - self._pending_request_step)
        elapsed_s = max(0.0, time.monotonic() - self._pending_request_time)
        if self._control_period_s > 0:
            self._estimated_delay_steps = max(0, int(math.ceil(elapsed_s / self._control_period_s)))
        if self._action_horizon is not None and elapsed_steps >= self._action_horizon:
            return
        self._install_chunk_locked(result, active_step=elapsed_steps)

    def _wait_for_next_chunk_locked(self, obs: Dict) -> None:
        assert self._action_horizon is not None
        if self._pending_future is None:
            self._install_chunk_locked(self._request_chunk({"obs": obs}), active_step=0)
            return
        future = self._pending_future
        self._pending_future = None
        try:
            result = future.result()
        except Exception:
            logger.exception("RTC wait failed; fresh chunk")
            self._install_chunk_locked(self._request_chunk({"obs": obs}), active_step=0)
            return
        elapsed_steps = max(0, self._active_step - self._pending_request_step)
        if elapsed_steps >= self._action_horizon:
            self._install_chunk_locked(self._request_chunk({"obs": obs}), active_step=0)
            return
        self._install_chunk_locked(result, active_step=elapsed_steps)

    def _install_chunk_locked(self, result: Dict, active_step: int) -> None:
        if "actions" not in result or not isinstance(result["actions"], np.ndarray):
            raise ValueError("RTC response must include ndarray 'actions'")
        action_horizon = result["actions"].shape[0]
        if self._action_horizon is None:
            self._action_horizon = action_horizon
        elif action_horizon != self._action_horizon:
            raise ValueError(f"action horizon mismatch {self._action_horizon} vs {action_horizon}")
        model_chunk = result.get("_rtc_model_actions")
        if not isinstance(model_chunk, np.ndarray):
            raise ValueError("RTC response must include '_rtc_model_actions'")
        self._active_result = result
        self._active_model_chunk = model_chunk
        self._active_step = max(0, min(active_step, self._action_horizon - 1))
        self._validate_rtc_config()
        self._next_request_step = self._execution_horizon

    def _slice_active_action_locked(self) -> Dict:
        assert self._active_result is not None and self._action_horizon is not None
        if self._active_step >= self._action_horizon:
            raise RuntimeError("RTC chunk exhausted")
        step = max(0, self._active_step)
        return {
            k: self._slice_value(v, step)
            for k, v in self._active_result.items()
            if not k.startswith("_rtc_")
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
            raise ValueError("inference_delay must be non-negative")
        if self._action_horizon is None or self._execution_horizon is None:
            return
        if self._execution_horizon <= 0:
            raise ValueError("execution_horizon must be positive")
        if self._execution_horizon > self._action_horizon:
            raise ValueError("execution_horizon cannot exceed action_horizon")
        if self._execution_horizon < self._inference_delay:
            raise ValueError("execution_horizon must be >= inference_delay")
```

### 抽象实现

**RTCChunkClient**：每 tick 单步 slice；`execution_horizon` 触发后台请求；`prev_action_chunk=left_shift(_rtc_model_actions)`。

### 本仓库源码位置

- [openpi-on-LIFT2/openpi_client/rtc_client_policy.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/openpi-on-LIFT2/openpi_client/rtc_client_policy.py)
- [src/openpi/serving/rtc_policy_server.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/src/openpi/serving/rtc_policy_server.py)
- [src/openpi/policies/policy.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/src/openpi/policies/policy.py)
- [RTC_GUIDE.md](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/RTC_GUIDE.md)

---

## 7. 机器人桥：`Lift2PolicyBridge`

### 输入 / 输出

- **`control_step(frame, prompt)`**  
  - `frame`：`{{"images": {{"head","left_wrist","right_wrist"}}, "eef": float32[14]}}`；standard 且队列非空时 **`frame=None`**。  
  - 出：**`float32[14]`** 绝对 EEF，**grip 已为 [0,5]**，可直接 `eef_arm_publish`。
- **`pop_executor_chunk()`**（可选）：`(execute_horizon, 14)` 供高频 executor。

### 具体实现

```python
@dataclass
class Lift2BridgeConfig:
    client_mode: str = "standard"  # "standard" | "rtc"
    execute_horizon: int = 30
    action_chunk_size: int = 30
    publish_rate_hz: float = 30.0
    max_delta_xyz: float = 0.05
    max_delta_rpy: float = 0.2
    smooth_alpha: float = 1.0
    binarize_gripper: bool = True
    rtc_action_horizon: Optional[int] = 30
    rtc_execution_horizon: int = 10
    rtc_inference_delay: int = 4
    rtc_prefix_attention_schedule: str = "exp"
    rtc_max_guidance_weight: float = 10.0


class Lift2PolicyBridge:
    """
    Connects any OpenPI-compatible WS/RTC server to LIFT2 14D absolute EEF commands.
    Integrate with ROS by passing frames from your RosOperator.get_frame() equivalent.
    """

    def __init__(
        self,
        host: str,
        port: int,
        config: Optional[Lift2BridgeConfig] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.cfg = config or Lift2BridgeConfig()
        self.cfg.client_mode = self.cfg.client_mode.lower()
        self.current_eef: Optional[np.ndarray] = None
        self.action_plan: collections.deque = collections.deque()
        self.executed_count = 0
        self.rtc_pred_eef: Optional[np.ndarray] = None
        self.latest_executor_chunk: Optional[np.ndarray] = None

        if self.cfg.client_mode == "rtc":
            self.client: BasePolicy = RTCClientPolicy(
                host=host,
                port=port,
                api_key=api_key,
                action_horizon=self.cfg.rtc_action_horizon,
                execution_horizon=self.cfg.rtc_execution_horizon,
                inference_delay=self.cfg.rtc_inference_delay,
                control_period_s=1.0 / self.cfg.publish_rate_hz,
                prefix_attention_schedule=self.cfg.rtc_prefix_attention_schedule,
                max_guidance_weight=self.cfg.rtc_max_guidance_weight,
            )
        else:
            self.client = WebsocketClientPolicy(host=host, port=port, api_key=api_key)

    def reset(self) -> None:
        self.action_plan.clear()
        self.executed_count = 0
        self.rtc_pred_eef = None
        self.latest_executor_chunk = None
        self.current_eef = None
        self.client.reset()

    def close(self) -> None:
        self.client.close()

    def set_current_eef(self, eef_14d: np.ndarray) -> None:
        self.current_eef = np.asarray(eef_14d, dtype=np.float32)

    def frame_to_observation(self, frame: RosFrame, prompt: str) -> Dict[str, Any]:
        if self.current_eef is None:
            raise RuntimeError("call set_current_eef() before inference")
        images = frame["images"]
        return build_openpi_observation(
            head_rgb=images["head"],
            left_wrist_rgb=images["left_wrist"],
            right_wrist_rgb=images["right_wrist"],
            state_14d=self.current_eef,
            prompt=prompt,
        )

    def pop_executor_chunk(self) -> Optional[np.ndarray]:
        chunk = self.latest_executor_chunk
        self.latest_executor_chunk = None
        return chunk

    def control_step(self, frame: Optional[RosFrame], prompt: str) -> np.ndarray:
        """
        One policy tick @ publish_rate_hz.
        - standard: pass frame=None when action_plan non-empty (no new images).
        - rtc: always pass fresh frame.
        Returns absolute 14D for robot (gripper in [0,5] if binarize).
        """
        if self.cfg.client_mode == "rtc":
            if frame is None:
                raise ValueError("RTC requires fresh frame every tick")
            self.set_current_eef(frame["eef"])
            return self._step_rtc(frame, prompt)

        if len(self.action_plan) > 0:
            return self._pop_standard_queue()
        if frame is None:
            raise ValueError("standard mode needs frame when action_plan is empty")
        self.set_current_eef(frame["eef"])
        return self._infer_standard_chunk(frame, prompt)

    def _step_rtc(self, frame: RosFrame, prompt: str) -> np.ndarray:
        observation = self.frame_to_observation(frame, prompt)
        result = self.client.infer(observation)
        delta = smooth_delta_action(
            sanitize_delta_action(
                result["actions"],
                max_delta_xyz=self.cfg.max_delta_xyz,
                max_delta_rpy=self.cfg.max_delta_rpy,
            ),
            smooth_alpha=self.cfg.smooth_alpha,
        )
        base = self.current_eef if self.rtc_pred_eef is None else self.rtc_pred_eef
        assert base is not None
        abs_pred = apply_eef_delta(base, delta)
        self.rtc_pred_eef = abs_pred.copy()
        self.latest_executor_chunk = np.asarray([abs_pred], dtype=np.float32)
        return postprocess_gripper_for_robot(abs_pred, binarize=self.cfg.binarize_gripper)

    def _infer_standard_chunk(self, frame: RosFrame, prompt: str) -> np.ndarray:
        observation = self.frame_to_observation(frame, prompt)
        result = self.client.infer(observation)
        chunk = np.asarray(result["actions"], dtype=np.float32)
        assert self.current_eef is not None
        trajectory = deltas_to_absolute_trajectory(self.current_eef, chunk, self.cfg)
        self.action_plan.extend(trajectory)
        exec_slice = trajectory[: self.cfg.execute_horizon]
        self.latest_executor_chunk = np.asarray(
            [postprocess_gripper_for_robot(p, binarize=self.cfg.binarize_gripper) for p in exec_slice],
            dtype=np.float32,
        )
        return self._pop_standard_queue()

    def _pop_standard_queue(self) -> np.ndarray:
        abs_pred = np.asarray(self.action_plan.popleft(), dtype=np.float32)
        self.executed_count += 1
        if self.executed_count >= self.cfg.execute_horizon:
            self.action_plan.clear()
            self.executed_count = 0
        return postprocess_gripper_for_robot(abs_pred, binarize=self.cfg.binarize_gripper)
```

### 抽象实现

**control_step**：standard 用 `action_plan` 摊销推理；rtc 每 tick 采图并 `RTCClientPolicy.infer`。换模型只改 observation 键名与 client 类。

### 本仓库源码位置

- 真机完整版: [openpi-on-LIFT2/deploy/client_lift2.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/openpi-on-LIFT2/deploy/client_lift2.py)

---

## 8. standard / RTC 逐步逻辑对照

### 具体实现

```python
# --- 非 RTC：OpenPIClientModel.step（standard）等价逻辑 ---
def step_standard(client, observation, current_eef, action_plan, executed_count, execute_horizon, cfg):
    if not action_plan:
        result = client.infer(observation)
        pred = current_eef.copy()
        for delta in result["actions"]:
            d = smooth_delta_action(
                sanitize_delta_action(delta, max_delta_xyz=cfg.max_delta_xyz, max_delta_rpy=cfg.max_delta_rpy),
                smooth_alpha=cfg.smooth_alpha,
            )
            pred = apply_eef_delta(pred, d)
            action_plan.append(pred.copy())
        executed_count = 0
    abs_pred = action_plan.popleft()
    executed_count += 1
    if executed_count >= execute_horizon:
        action_plan.clear()
        executed_count = 0
    return postprocess_gripper_for_robot(abs_pred, binarize=cfg.binarize_gripper), executed_count


# --- RTC：_step_rtc 等价逻辑 ---
def step_rtc(rtc_client, observation, current_eef, rtc_pred_eef, cfg):
    result = rtc_client.infer(observation)
    delta = smooth_delta_action(
        sanitize_delta_action(result["actions"], max_delta_xyz=cfg.max_delta_xyz, max_delta_rpy=cfg.max_delta_rpy),
        smooth_alpha=cfg.smooth_alpha,
    )
    base = current_eef if rtc_pred_eef is None else rtc_pred_eef
    abs_pred = apply_eef_delta(base, delta)
    rtc_pred_eef = abs_pred.copy()
    return postprocess_gripper_for_robot(abs_pred, binarize=cfg.binarize_gripper), rtc_pred_eef


def get_action_dispatch(bridge, frame, prompt):
    if bridge.cfg.client_mode == "rtc":
        return bridge.control_step(frame, prompt)
    if len(bridge.action_plan) > 0:
        return bridge.control_step(None, prompt)
    return bridge.control_step(frame, prompt)
```

### 抽象实现

| 模式 | 采图 | Server | 累积变量 |
|------|------|--------|----------|
| standard | 仅 `action_plan` 空 | 同步 chunk | `pred_eef` 链在 deque 内 |
| rtc | 每 tick | slice + 后台 chunk | `rtc_pred_eef` |

**不变量**：不要用延迟真机 state 每步重锚定 xyz/rpy delta。

### 本仓库源码位置

- [openpi-on-LIFT2/deploy/client_lift2.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/openpi-on-LIFT2/deploy/client_lift2.py) — `OpenPIClientModel.step` / `_step_rtc` / `get_action`

---

## 9. GPU 策略服务（openpi 仓库内 `serve_policy.py`）

本节 **不是**「任意模型」，而是：**你必须用本仓库训练出的 LIFT2 π₀.₅ checkpoint**，由 `Policy` 保证 §0.3 的输入输出格式。机器人客户端 **只认** §0.3 的 WS 契约。

### 9.1 服务端必须满足的输出格式（自检）

在 GPU 机上启动服务后，用仓库内测试或 `openpi-on-LIFT2/test_client.py` 连一次，确认：

1. 首包 `metadata` 可读（含 `action_horizon` 或与首帧 `actions.shape[0]` 一致）。
2. `actions.dtype` 为 float，shape **`(H, 14)`**，`H` 等于训练时 `Pi0Config.action_horizon`。
3. RTC 模式：同一响应中 **`_rtc_model_actions.shape == actions.shape`**。
4. 用 `--debug-print` 时，非图像 obs 与 action 数值在合理范围（xyz delta 多为毫米级）。

若 shape 不是 14：说明 **config / checkpoint 不是 LIFT2 EEF**（例如 `action_dim=32`  padding 未截断）——应使用 `LeRobotLift2DataConfig` 训练的 config（`Lift2Outputs` 截断为 14）。

### 9.2 启动命令（在 openpi 仓库根目录）

**非 RTC**（与 `launch_profiles.yaml` → `default` 配对）：

```bash
cd /path/to/openpi
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
  --server-mode websocket \
  --port 7777 \
  policy:checkpoint \
  --policy.config=<你的 TrainConfig.name，如 pi05_0319_pick_and_place_block_30hz_chunk30_lora> \
  --policy.dir=./checkpoints/<exp>/<step>
```

**RTC**（与 `profile: rtc` 配对；**horizon 与客户端、训练一致**）：

```bash
CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_policy.py \
  --server-mode rtc \
  --port 7777 \
  --rtc-execution-horizon 10 \
  --rtc-inference-delay 4 \
  --rtc-prefix-attention-schedule exp \
  --rtc-max-guidance-weight 10.0 \
  policy:checkpoint \
  --policy.config=<同上> \
  --policy.dir=./checkpoints/<exp>/<step>
```

可选：`--default-prompt "..."`（与训练 `default_prompt` 一致）、`--fix-left-arm` / `--fix-right-arm`（单臂任务固定另一侧 state，见 `serve_policy.py`）。

### 9.3 输入 / 输出（`serve_policy.py` 视角）

| 阶段 | 输入 | 输出 |
|------|------|------|
| 加载 | `--policy.config` + `--policy.dir` | 内存中 `Policy`（含 norm_stats、transforms） |
| 握手 | TCP/WS 连接 | `metadata` msgpack |
| 每步 infer | 客户端 observation（§0.3） | `actions` (H,14) + timing；RTC 加 `_rtc_model_actions` |

### 9.4 抽象实现

**PolicyServer**：任意框架均可，只要实现 §0.3 WS 行为；**本仓库标准实现**为 `WebsocketPolicyServer` / `RTCPolicyServer` + `create_trained_policy`。

### 9.5 本仓库源码位置

- [https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/scripts/serve_policy.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/scripts/serve_policy.py)
- [https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/src/openpi/policies/lift2_policy.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/src/openpi/policies/lift2_policy.py)（`make_lift2_example` 为最小 obs 样例）
- [https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/src/openpi/policies/policy_config.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/src/openpi/policies/policy_config.py)（`create_trained_policy` transform 链）
- [https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/openpi-on-LIFT2/launch_profiles.yaml](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/openpi-on-LIFT2/launch_profiles.yaml)（机器人默认 host/port/rtc 参数）

---

## 9A. 端到端对接清单（仅 openpi 仓库）

按顺序打勾：

1. **数据 / 训练**（GPU）：HDF5 → `convert_hdf5_to_lerobot_eef.py` → `compute_norm_stats` → `config.py` 中 `LeRobotLift2DataConfig` + `action_horizon` → `train.py` 得到 checkpoint（见 [LIFT2_GUIDE.md](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/LIFT2_GUIDE.md)）。
2. **确认契约**：`make_lift2_example()` / 训练 config 的 `action_horizon`、prompt 与真机任务一致。
3. **起服务**（GPU）：§9.2，`0.0.0.0:7777` 监听；防火墙放行；`curl http://<ip>:7777/healthz` 可选。
4. **机器人 ROS**：双臂 + 三路 RealSense（见 `openpi-on-LIFT2/run_lift2.sh` / README）。
5. **起客户端**：拷贝 `openpi-on-LIFT2` 到机器人，`./launch.sh --profile default|rtc --task <preset>` 或 `client_lift2.py --host <GPU_IP> --port 7777`。
6. **参数对齐**：`execute_horizon` ≤ `action_horizon`；RTC 时 server CLI 与 `launch_profiles.yaml` 中 `rtc_*` 一致；`publish_rate` 与训练 fps（常 30Hz）一致。
7. **联调**：先 `test_client.py`（仅 WS，无 ROS），再真机 `client_lift2.py`；异常时开 server `--debug-print`。

---

## 10. ROS 真机层（`openpi-on-LIFT2`）

### 输入 / 输出（`RosOperator`）

| 回调 / 方法 | 输入（期待） | 输出 |
|-------------|----------------|------|
| `get_frame()` |  deque 中已有同步时刻的三路图 + 最新双臂 `PosCmd` | `(img_front, img_left, img_right, ..., arm_left, arm_right)` 或 `False` |
| `eef_arm_publish(left, right)` | 各 `list`/`ndarray` 长度 7：xyz, rpy, **grip 0–5** | ROS 发布 `PosCmd` 到 cmd 话题 |

**`get_frame` 失败**：客户端应等待，**不要**用空 obs 推理。

### 具体实现

```python
# frame 由 RosOperator.get_frame() 组装
frame = {
    "images": {"head": img_front, "left_wrist": img_left, "right_wrist": img_right},
    "eef": pose_to_eef(arm_left_pose, arm_right_pose),  # 或 pose_arrays_to_eef
}
cmd_14 = bridge.control_step(frame, prompt="pick up the cube")
left, right = cmd_14[:7], cmd_14[7:14]
ros_operator.eef_arm_publish(left.tolist(), right.tolist())
```

### 抽象实现

**HardwareAdapter**：`get_frame() -> RosFrame | None`，`publish(abs_14d)`。桥接层不依赖 ROS 消息类型。

### 本仓库源码位置

- [openpi-on-LIFT2/deploy/utils/rosoperator.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/openpi-on-LIFT2/deploy/utils/rosoperator.py)
- [openpi-on-LIFT2/deploy/utils/eef_action_executor.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/openpi-on-LIFT2/deploy/utils/eef_action_executor.py)

---

## 11. 源码索引

| 模块 | 链接 |
|------|------|
| client_lift2.py | [openpi-on-LIFT2/deploy/client_lift2.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/openpi-on-LIFT2/deploy/client_lift2.py) |
| rtc_client_policy.py | [openpi-on-LIFT2/openpi_client/rtc_client_policy.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/openpi-on-LIFT2/openpi_client/rtc_client_policy.py) |
| websocket_client_policy.py | [openpi-on-LIFT2/openpi_client/websocket_client_policy.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/openpi-on-LIFT2/openpi_client/websocket_client_policy.py) |
| rtc_policy_server.py | [src/openpi/serving/rtc_policy_server.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/src/openpi/serving/rtc_policy_server.py) |
| websocket_policy_server.py | [src/openpi/serving/websocket_policy_server.py](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/src/openpi/serving/websocket_policy_server.py) |
| COMPARISON.md | [openpi-on-LIFT2/COMPARISON.md](https://github.com/Infinity4B/openpi-lift2/blob/wx_dev/openpi-on-LIFT2/COMPARISON.md) |

*链接分支为 `wx_dev`。*

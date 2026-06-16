# OpenPI vs X-VLA on LIFT2：技术对比

本文档对比 OpenPI 和 X-VLA 在 LIFT2 机器人平台上的实现。

## 架构概览

### X-VLA-on-LIFT2
```
机器人 → ROS → HTTP客户端 → X-VLA服务器 → 末端位姿动作 → 机器人
         ↓
    末端位姿（6D旋转）
```

### OpenPI-on-LIFT2
```
机器人 → ROS → WebSocket客户端 → OpenPI服务器 → EEF增量动作 → PosCmd → 机器人
         ↓
    末端位姿（14维：每臂 xyz + rpy + gripper）
```

## 关键差异

| 方面 | X-VLA | OpenPI |
|------|-------|--------|
| **模型** | X-VLA（基于Florence-2） | π₀.₅（基于PaliGemma） |
| **控制空间** | 末端位姿（笛卡尔空间） | 末端位姿（EEF delta） |
| **动作维度** | 20（每臂10：xyz + 6D旋转 + 夹爪） | 14（每臂7：xyz + rpy + gripper） |
| **状态输入** | 末端6D旋转 | 末端 xyz/rpy + 归一化夹爪 |
| **通信方式** | HTTP REST API | WebSocket |
| **动作格式** | 绝对位姿 + 相对增量 | xyz/rpy 增量 + 夹爪命令 |
| **平滑处理** | 客户端插值（K帧） | 策略生成（隐式） |
| **夹爪阈值** | 0.45（归一化[0,1]） | 0.6（归一化[0,1]，默认二值化） |

## 控制空间对比

### X-VLA：末端位姿控制
```python
# 状态：末端位姿（20维）
state = [
    left_x, left_y, left_z,           # 左臂位置
    left_rot_6d (6个值),              # 左臂旋转（6D）
    left_gripper,                     # 左夹爪
    right_x, right_y, right_z,        # 右臂位置
    right_rot_6d (6个值),             # 右臂旋转（6D）
    right_gripper                     # 右夹爪
]

# 动作：增量 + 绝对
action[0:3] += state[0:3]      # 将增量加到当前位置
action[10:13] += state[10:13]  # 将增量加到当前位置
```

**优点**：
- 对操作任务直观
- 更容易指定任务约束
- 对运动学奇异点更鲁棒

**缺点**：
- 需要逆运动学
- 可能有IK求解失败
- 控制循环较慢

### OpenPI：末端增量控制
```python
# 状态：末端位姿（14维）
state = [
    left_x, left_y, left_z, left_roll, left_pitch, left_yaw, left_gripper,
    right_x, right_y, right_z, right_roll, right_pitch, right_yaw, right_gripper
]

# 动作：xyz/rpy 增量 + 夹爪命令
next_eef = apply_eef_delta(current_eef, delta_action)
ros_operator.eef_arm_publish(left_eef, right_eef)
```

**优点**：
- 与当前 LIFT2 客户端和训练数据的 EEF 表示一致
- 动作是相对增量，更适合连续闭环控制
- 通过 `arm_control/PosCmd` 直接发布末端命令
- 每步可做 NaN/Inf 检查和 xyz/rpy 限幅

**缺点**：
- 仍需要底层控制器完成末端命令执行
- 欧拉角表示需要注意角度连续性
- 单步限幅需要根据机器人实测调参

## 通信协议

### X-VLA：HTTP REST API
```python
query = {
    "proprio": json_numpy.dumps(state),
    "image0": json_numpy.dumps(img_head),
    "image1": json_numpy.dumps(img_left),
    "image2": json_numpy.dumps(img_right),
    "language_instruction": "拾取并放置",
    "steps": 10,
    "domain_id": 0
}
response = requests.post(url, json=query)
action = response.json()["action"]
```

**特点**：
- 无状态（每个请求独立）
- 易于用curl/Postman调试
- 标准HTTP工具可用
- 每个请求开销较高

### OpenPI：WebSocket
```python
client = websocket_client_policy.WebsocketClientPolicy(host, port)
observation = {
    "observation.images.head": img_head,
    "observation.images.left_wrist": img_left,
    "observation.images.right_wrist": img_right,
    "observation.state": state,
    "prompt": "把盘子里的东西放到左边"
}
result = client.infer(observation)
action = result["actions"]
```

**特点**：
- 持久连接
- 延迟更低
- 支持二进制协议
- 需要连接管理

## 动作平滑

### X-VLA：客户端插值
```python
# 在当前位姿和目标位姿之间生成K个插值帧
interpolated = np.linspace(current_pose, first_action, K+2)
transition_frames = interpolated[1:-1]
smoothed_actions = np.vstack([transition_frames, action_sequence])
```

**优点**：
- 机器人运动更平滑
- 减少抖动
- 可配置平滑度（K参数）

### OpenPI：策略生成
```python
# 策略直接生成平滑的动作序列
# 无需客户端平滑
action_chunk = policy(observation)  # (horizon, 14)
```

**优点**：
- 策略学习最优平滑度
- 无需手动调参
- 适应任务需求

## 夹爪控制

### X-VLA：归一化二值化
```python
THRESHOLD = 0.45  # 归一化 [0, 1]
OPEN = 4.9
CLOSED = 1.0

gripper = OPEN if raw_value > THRESHOLD else CLOSED
```

### OpenPI：归一化二值化
```python
THRESHOLD = 0.6  # 归一化 [0, 1]
OPEN = 4.9
CLOSED = 0.5

gripper = OPEN if raw_value > THRESHOLD else CLOSED
```

OpenPI 客户端当前提供两个开关：
- `--binarize_gripper`：默认启用，归一化输出 `>= 0.6` 发布为张开，否则发布为闭合
- `--no_binarize_gripper`：关闭二值化，连续归一化值反归一化到机器人夹爪范围 `[0, 5]`

## 性能特征

### 延迟分解

**X-VLA**：
- 图像编码：~5ms
- HTTP开销：~10ms
- 模型推理：~50-100ms
- 响应解析：~5ms
- **总计**：~70-120ms

**OpenPI**：
- 图像编码：~5ms
- WebSocket开销：~2ms
- 模型推理：~50-100ms
- 响应解析：~2ms
- **总计**：~60-110ms

### 控制频率

**X-VLA**：
- 典型：15 Hz
- 最大：30 Hz（带平滑）

**OpenPI**：
- 典型：30 Hz
- 最大：60 Hz（带动作分块）

## 代码复杂度

### X-VLA 实现
- 主客户端：~487行
- ROS操作器：~288行
- 旋转工具：~150行
- **总计**：~925行

### OpenPI 实现
- 主客户端：~450行
- ROS操作器：~220行
- **总计**：~670行

OpenPI更简单因为：
- 不需要 6D 旋转和欧拉角之间的转换
- 状态和动作都固定为 14 维 EEF 表示
- 机器人端只负责策略请求、增量累积和 `PosCmd` 发布

## 何时使用

### 使用 X-VLA 当：
- 任务需要笛卡尔空间推理
- 需要显式位姿控制
- 任务有几何约束
- 有良好的IK求解器

### 使用 OpenPI 当：
- 训练数据和策略输出使用 14 维 EEF 表示
- 想使用 WebSocket 远程推理和动作分块
- 希望机器人端逻辑尽量轻量，只做必要检查和发布
- 希望保留 EEF 控制语义，同时避免 6D 旋转转换

## 迁移指南

### 从 X-VLA 到 OpenPI

1. **改变状态表示**：
   ```python
   # X-VLA
   state = eef_6d(left_pose, right_pose)  # 20维

   # OpenPI
   state = pose_to_eef(left_pose, right_pose)  # 14维：xyz/rpy/gripper
   ```

2. **改变动作解释**：
   ```python
   # X-VLA
   action = abs_6d_2_abs_euler(action)  # 转换为欧拉角
   ros_operator.eef_arm_publish(left, right)

   # OpenPI
   # xyz/rpy 是相对增量，夹爪是命令值
   next_eef = apply_eef_delta(current_eef, delta_action)
   ros_operator.eef_arm_publish(left, right)
   ```

3. **更新ROS话题**：
   ```python
   # X-VLA
   --arm_left_pose_topic /arm_left/arm_status_ee

   # OpenPI
   --arm_left_pose_topic /arm_left/arm_status_ee
   --arm_right_pose_topic /arm_right/arm_status_ee
   --arm_left_cmd_topic /arm_left_cmd
   --arm_right_cmd_topic /arm_right_cmd
   ```

### 从 OpenPI 到 X-VLA

1. **改变末端姿态表示**：
   ```python
   # 从 xyz/rpy 改为 X-VLA 需要的 6D 旋转表示
   state = eef_rpy_to_eef_6d(left_pose, right_pose)
   ```

2. **改变动作后处理**：
   ```python
   # 将 X-VLA 输出转换为绝对欧拉角位姿后发布
   action = abs_6d_2_abs_euler(action)
   ros_operator.eef_arm_publish(left_action, right_action)
   ```

3. **更新通信**：
   ```python
   # 用HTTP替换WebSocket
   response = requests.post(url, json=query)
   ```

## 结论

两种实现都已准备好用于生产，各有优势：

- **X-VLA**：更适合需要显式位姿控制的任务
- **OpenPI**：EEF delta 表示更轻量，WebSocket 通信更适合远程策略推理

根据以下因素选择：
- 训练数据格式
- 任务需求
- 控制频率需求
- 可用计算资源

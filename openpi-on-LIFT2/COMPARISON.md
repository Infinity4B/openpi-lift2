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
机器人 → ROS → WebSocket客户端 → OpenPI服务器 → 关节动作 → 机器人
         ↓
    关节位置（14维）
```

## 关键差异

| 方面 | X-VLA | OpenPI |
|------|-------|--------|
| **模型** | X-VLA（基于Florence-2） | π₀.₅（基于PaliGemma） |
| **控制空间** | 末端位姿（笛卡尔空间） | 关节空间 |
| **动作维度** | 20（每臂10：xyz + 6D旋转 + 夹爪） | 14（每臂7：6关节 + 夹爪） |
| **状态输入** | 末端6D旋转 | 关节位置 |
| **通信方式** | HTTP REST API | WebSocket |
| **动作格式** | 绝对位姿 + 相对增量 | 绝对关节位置 |
| **平滑处理** | 客户端插值（K帧） | 策略生成（隐式） |
| **夹爪阈值** | 0.45（归一化[0,1]） | 3.5（原始值[1.0, 4.9]） |

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

### OpenPI：关节空间控制
```python
# 状态：关节位置（14维）
state = [
    left_j1, left_j2, left_j3, left_j4, left_j5, left_j6, left_gripper,
    right_j1, right_j2, right_j3, right_j4, right_j5, right_j6, right_gripper
]

# 动作：直接关节位置
action = [相同14维格式]
```

**优点**：
- 直接控制，无需IK
- 控制循环更快
- 无IK求解失败
- 更适合学习策略

**缺点**：
- 任务规范不够直观
- 难以强制笛卡尔约束
- 需要策略学习运动学

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

### OpenPI：原始值二值化
```python
THRESHOLD = 3.5  # 原始值 [1.0, 4.9]
OPEN = 4.9
CLOSED = 1.0

gripper = OPEN if raw_value > THRESHOLD else CLOSED
```

两者都支持多种模式：
- **hard**：高阈值（不敏感）
- **soft**：软二值化带过渡区间（平滑过渡）
- **low_threshold**：低阈值（更敏感）
- **raw**：不二值化（连续）

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
- 无旋转转换（6D ↔ 欧拉角）
- 无IK/FK计算
- 状态表示更简单

## 何时使用

### 使用 X-VLA 当：
- 任务需要笛卡尔空间推理
- 需要显式位姿控制
- 任务有几何约束
- 有良好的IK求解器

### 使用 OpenPI 当：
- 训练数据在关节空间
- 想要更快的控制循环
- 想避免IK问题
- 策略应该学习运动学

## 迁移指南

### 从 X-VLA 到 OpenPI

1. **改变状态表示**：
   ```python
   # X-VLA
   state = eef_6d(left_pose, right_pose)  # 20维

   # OpenPI
   state = np.concatenate([left_qpos, right_qpos])  # 14维
   ```

2. **改变动作解释**：
   ```python
   # X-VLA
   action = abs_6d_2_abs_euler(action)  # 转换为欧拉角
   ros_operator.eef_arm_publish(left, right)

   # OpenPI
   # 动作已经在关节空间
   ros_operator.joint_arm_publish(left, right)
   ```

3. **更新ROS话题**：
   ```python
   # X-VLA
   --arm_left_pose_topic /arm_left/arm_status_ee

   # OpenPI
   --arm_left_joint_topic /arm_left/joint_states
   ```

### 从 OpenPI 到 X-VLA

1. **添加正运动学**：
   ```python
   # 从关节计算末端位姿
   left_pose = forward_kinematics(left_qpos)
   right_pose = forward_kinematics(right_qpos)
   ```

2. **添加逆运动学**：
   ```python
   # 将动作转换为关节空间
   left_qpos = inverse_kinematics(left_action)
   right_qpos = inverse_kinematics(right_action)
   ```

3. **更新通信**：
   ```python
   # 用HTTP替换WebSocket
   response = requests.post(url, json=query)
   ```

## 结论

两种实现都已准备好用于生产，各有优势：

- **X-VLA**：更适合需要显式位姿控制的任务
- **OpenPI**：更简单、更快、更直接的控制

根据以下因素选择：
- 训练数据格式
- 任务需求
- 控制频率需求
- 可用计算资源

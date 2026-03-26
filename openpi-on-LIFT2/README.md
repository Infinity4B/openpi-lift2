# OpenPI on LIFT2

OpenPI 远程推理客户端，用于 ARX R5 双臂机器人（LIFT2 平台）。

本项目支持在 LIFT2 机器人平台上运行 OpenPI π₀.₅ 模型，采用远程策略推理架构。

## 概述

- **任务**：通用 LIFT2 双臂任务
- **机器人**：ARX R5 双臂机械臂
- **控制方式**：关节空间控制（14维：每臂7维）
- **相机**：3个RGB相机（头部、左腕、右腕）
- **架构**：远程策略服务器 + 机器人端客户端

## 目录结构

```
openpi-on-LIFT2/
├── deploy/
│   ├── client_lift2.py               # 主推理客户端
│   └── utils/
│       ├── __init__.py
│       └── rosoperator.py             # ROS接口封装
└── README.md
```

## 前置条件

### 策略服务器端

1. 已训练好的 OpenPI LIFT2 模型：
```bash
cd /path/to/openpi
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_lift2_lora \
    --policy.dir=checkpoints/pi05_lift2_lora/<exp_name>/<step>
```

### 机器人端

1. 安装了 ARX R5 软件包的 ROS 环境
2. OpenPI 客户端包：
```bash
cd /path/to/openpi/packages/openpi-client
pip install -e .
```

3. 所需 ROS 消息类型：
   - `arm_control/JointControl`
   - `arm_control/JointInformation`
   - `sensor_msgs/Image`

## 使用方法

### 1. 启动策略服务器（训练机）

```bash
# 启动 OpenPI 策略服务器
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_lift2_lora \
    --policy.dir=checkpoints/pi05_lift2_lora/<exp_name>/<step>
```

服务器默认监听 8000 端口。

### 2. 启动 ROS 系统（机器人）

```bash
# 启动机器人控制器
roslaunch arx_r5_controller open_double_arm.launch

# 启动相机
roslaunch realsense2_camera rs_multiple_devices.launch
```

### 3. 运行 OpenPI 客户端（机器人）

```bash
cd /path/to/openpi/openpi-on-LIFT2

# 基本用法
python deploy/client_lift2.py \
    --host <策略服务器IP> \
    --port 8000

# 自定义设置
python deploy/client_lift2.py \
    --host 192.168.1.100 \
    --port 8000 \
    --publish_rate 30 \
    --execute_horizon 10 \
    --gripper_mode low_threshold \
    --language_instruction "把盘子里的东西放到左边" \
    --verbose
```

## 命令行参数

### 策略服务器
- `--host`: 策略服务器IP地址（默认：localhost）
- `--port`: 策略服务器端口（默认：8000）

### 任务配置
- `--language_instruction`: 任务描述（默认："put items from plate to left"）
- `--max_publish_step`: 最大执行步数（默认：1000）

### 控制参数
- `--publish_rate`: 控制频率（Hz）（默认：30）
- `--execute_horizon`: 每次推理执行的帧数（默认：10）
- `--gripper_mode`: 夹爪处理模式
  - `low_threshold`: 低阈值二值化（默认，阈值3.5）
  - `hard`: 高阈值二值化（阈值4.25）
  - `soft`: 软二值化（带过渡区间）
  - `raw`: 不二值化

### 初始化
- `--auto_init`: 自动移动到初始位姿（默认：启用）
- `--no_auto_init`: 禁用自动初始化
- `--init_duration`: 到达初始位姿的时长（秒）（默认：3.0）
- `--wait_after_init`: 初始化后等待用户按Enter键
- `--left_init_pose`: 左臂初始关节位置（7个值）
- `--right_init_pose`: 右臂初始关节位置（7个值）

### ROS话题
- `--img_front_topic`: 头部相机话题（默认：/camera_h/color/image_raw）
- `--img_left_topic`: 左腕相机话题（默认：/camera_l/color/image_raw）
- `--img_right_topic`: 右腕相机话题（默认：/camera_r/color/image_raw）
- `--arm_left_joint_topic`: 左臂关节状态（默认：/arm_left/joint_states）
- `--arm_right_joint_topic`: 右臂关节状态（默认：/arm_right/joint_states）
- `--arm_left_cmd_topic`: 左臂命令话题（默认：/arm_left_cmd）
- `--arm_right_cmd_topic`: 右臂命令话题（默认：/arm_right_cmd）

### 日志
- `--verbose`: 启用详细日志
- `--log_latency`: 记录每次推理的延迟

## 使用示例

### 快速测试
```bash
# 在GPU机器上启动策略服务器
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_lift2_lora \
    --policy.dir=checkpoints/pi05_lift2_lora/<exp_name>/<step>

# 在机器人上
python deploy/client_lift2.py \
    --host 192.168.1.100 \
    --verbose
```

### 自定义初始位姿
```bash
python deploy/client_lift2.py \
    --host 192.168.1.100 \
    --left_init_pose -0.0008 0.0032 0.0055 -0.0037 -0.0025 0.0005 4.8946 \
    --right_init_pose 0.0 0.0 0.0 0.0 0.0 0.0 4.9 \
    --init_duration 5.0 \
    --wait_after_init
```

### 高频控制
```bash
python deploy/client_lift2.py \
    --host 192.168.1.100 \
    --publish_rate 60 \
    --execute_horizon 5 \
    --log_latency
```

## 架构细节

### 数据流

1. **观测采集**：
   - 3个相机的RGB图像（头部、左腕、右腕）
   - 双臂关节位置（14维）

2. **预处理**：
   - 图像调整为224x224（带padding）
   - 转换为uint8格式
   - 关节位置原样传递（服务器端归一化）

3. **远程推理**：
   - WebSocket连接到策略服务器
   - 服务器返回动作序列（action_horizon × 14）

4. **动作执行**：
   - 执行前N个动作（execute_horizon）
   - 丢弃剩余动作防止误差累积
   - 根据模式二值化夹爪值

5. **控制循环**：
   - 以指定频率发布关节命令
   - 动作队列耗尽时重新推理

### 夹爪处理

策略输出连续的夹爪值，会被二值化：

- **low_threshold**（默认）：阈值3.5
  - > 3.5 → 4.9（张开）
  - ≤ 3.5 → 1.0（闭合）

- **hard**：阈值4.25
  - > 4.25 → 4.9（张开）
  - ≤ 4.25 → 1.0（闭合）

- **soft**：软二值化，过渡区间0.5
  - > 4.0 → 4.9（张开）
  - < 3.0 → 1.0（闭合）
  - 3.0-4.0 → 线性插值

- **raw**：限制在[0.5, 5.0]范围内

## 与 X-VLA-on-LIFT2 对比

| 特性 | X-VLA-on-LIFT2 | OpenPI-on-LIFT2 |
|------|----------------|-----------------|
| 模型 | X-VLA | OpenPI π₀.₅ |
| 控制空间 | 末端位姿（6D） | 关节空间（7D每臂） |
| 动作表示 | 绝对位姿+增量 | 关节位置 |
| 通信方式 | HTTP REST API | WebSocket |
| 平滑处理 | 线性插值 | 无（策略生成） |
| 状态输入 | 末端6D旋转 | 关节位置 |

## 故障排除

### 连接问题
```bash
# 测试策略服务器连通性
curl http://<服务器IP>:8000/health

# 检查端口是否开放
telnet <服务器IP> 8000
```

### ROS话题问题
```bash
# 列出活动话题
rostopic list

# 检查相机数据
rostopic hz /camera_h/color/image_raw

# 检查关节状态
rostopic echo /arm_left/joint_states
```

### 夹爪不动
- 尝试不同夹爪模式：`--gripper_mode hard` 或 `--gripper_mode soft`
- 用 `--verbose` 检查日志中的夹爪值

### 延迟高
- 检查机器人和服务器之间的网络带宽
- 如需要可降低图像分辨率（需修改代码）
- 增加 `--execute_horizon` 减少推理频率

## 性能建议

1. **网络**：使用有线千兆以太网获得最低延迟
2. **控制频率**：推荐30Hz，快速任务可用60Hz
3. **执行范围**：10帧平衡响应性和稳定性
4. **夹爪模式**：从 `low_threshold` 开始，根据需要调整

## 安全提示

- 始终先用 `--wait_after_init` 测试
- 保持急停按钮可触及
- 操作期间监控夹爪力
- 从慢速开始（`--publish_rate 15`）

## 许可证

与 OpenPI 项目相同。

## 引用

如果使用此代码，请引用 OpenPI 论文：

```bibtex
@article{openpi2024,
  title={OpenPI: Open-Source Physical Intelligence},
  author={Physical Intelligence Team},
  year={2024}
}
```

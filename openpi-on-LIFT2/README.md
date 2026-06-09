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

# 启动 RTC 策略服务器（配合机器人端 --profile rtc）
uv run scripts/serve_policy.py \
    --server-mode rtc \
    --rtc-execution-horizon 10 \
    --rtc-inference-delay 2 \
    --rtc-prefix-attention-schedule exp \
    --rtc-max-guidance-weight 10.0 \
    policy:checkpoint \
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

# 默认 profile：30Hz，不上采样
bash launch.sh --task tube

# 上采样 profile：60Hz，30Hz -> 60Hz
bash launch.sh --profile upsample --task tube

# RTC profile：每个控制 tick 都输入当前观测，后台异步推理下一段 action chunk
bash launch.sh --profile rtc --task tube --verbose

# 覆盖 profile 中的 host
bash launch.sh --profile upsample --host <策略服务器IP> --task tube

# 启用 3 路相机视频录制（结束后只确认是否保留；确认后静默在后台转视频；Ctrl+C 中断后也会继续询问是否保留）
bash launch.sh --profile upsample --task tube --record_video

# 直接运行客户端
python deploy/client_lift2.py \
    --host 192.168.1.100 \
    --port 8000 \
    --task towel \
    --verbose

# 自定义文本会覆盖 --task 的默认描述
python deploy/client_lift2.py \
    --host 192.168.1.100 \
    --port 8000 \
    --task tube \
    --language_instruction "Transfer the test tube carefully." \
    --verbose
```

## 命令行参数

### 策略服务器
- `--host`: 策略服务器IP地址（默认：localhost）
- `--port`: 策略服务器端口（默认：8000）

### 任务配置
- `--task`: 任务简写，自动填充默认任务描述。支持：`tube`、`towel`、`wrench`、`power_strip`、`drum`、`dice`、`stack`
- `--language_instruction`: 自定义任务描述；如果同时传入，会覆盖 `--task` 的默认描述
- `--max_publish_step`: 最大执行步数（默认：1000）

默认任务描述：
- `tube`: `Transfer the test tube from the right rack to the left rack.`
- `towel`: `Flatten the towel.`
- `wrench`: `Open the toolbox, check the items inside one by one, and find the wrench.`
- `power_strip`: `Move the power strip with the left arm, and press the button of the power strip with the right arm.`
- `drum`: `Pick up two small drumsticks and hit the small drum.`
- `dice`: `Roll the dice and move the small stand the specified number of squares based on the number rolled.`
- `stack`: `Stack the building blocks one by one with the larger ones at the bottom.`

### 控制参数
- `launch.sh` 通过 `launch_profiles.yaml` 管理启动参数，支持 `--profile default`、`--profile upsample` 和 `--profile rtc`
- `default`: `host=192.168.101.101`、`port=7777`、`publish_rate=30`、`execute_horizon=30`、`action_chunk_size=30`、`source_hz=30`、`target_hz=30`
- `upsample`: `host=192.168.101.101`、`port=7777`、`publish_rate=60`、`execute_horizon=59`、`action_chunk_size=30`、`source_hz=30`、`target_hz=60`
- `rtc`: `client_mode=rtc`、`publish_rate=30`、`action_chunk_size=30`、`rtc_execution_horizon=10`、`rtc_inference_delay=2`
- `--client_mode`: 客户端执行模式，`standard` 为本地动作队列，`rtc` 为 RTC 单步客户端
- `--host` / `--port`: 可覆盖 profile 中的服务器地址
- `--publish_rate`: 控制频率（Hz）（默认：30）
- `--execute_horizon`: 每次推理执行的帧数（默认：10）
- `--rtc_action_horizon`: RTC 模型 action chunk 长度，默认等于 `action_chunk_size`
- `--rtc_execution_horizon`: RTC 后台请求下一段 chunk 的间隔，必须不超过 `rtc_action_horizon`
- `--rtc_inference_delay`: RTC 固定推理延迟（控制步数），需要与服务端参数一致
- `--rtc_control_period_s`: RTC 控制周期，默认 `1 / publish_rate`
- `--rtc_prefix_attention_schedule`: RTC 前缀 attention schedule，例如 `exp`
- `--rtc_max_guidance_weight`: RTC 前缀引导最大权重
- `--gripper_mode`: 夹爪处理模式
  - `low_threshold`: 低阈值二值化（默认，阈值3.5）
  - `hard`: 高阈值二值化（阈值4.25）
  - `soft`: 软二值化（带过渡区间）
  - `raw`: 不二值化

### 初始化
- `--auto_init`: 自动移动到初始位姿（默认：启用）；任务开始前会归位，任务正常结束后会再归位一次，按 **Ctrl+C** 中断时也会尝试自动归位
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
- `--record_video`: 先将 3 路 D405 相机帧保存到 `./pic/{task}/{seq}/`，结束后先询问是否保留本次采集；确认保留后会静默在后台转成视频保存到 `./video/{task}/{seq}/`，这样可以更快开始下一次采集；不保留时会同时删除本次图片和视频目录。若按 **Ctrl+C** 中断，客户端会先尝试自动归位，然后继续询问是否保留本次录像

### 图片转视频

如果已有图片帧目录，可以单独运行转换脚本，不需要启动推理客户端：

```bash
# 转换一次 LIFT2 录制目录，输入目录下包含 camera_h/camera_l/camera_r 子目录
python frames_to_video.py pic/tube/1 -o video/tube/1 --fps 60

# 也可以只转换单个图片文件夹
python frames_to_video.py pic/tube/1/camera_h -o video/tube/1/camera_h.mp4 --fps 60
```

## 使用示例

### 快速测试
```bash
# 在GPU机器上启动策略服务器
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_lift2_lora \
    --policy.dir=checkpoints/pi05_lift2_lora/<exp_name>/<step>

# 在机器人上：默认 30Hz profile
bash launch.sh \
    --task tube \
    --verbose

# 在机器人上：60Hz 上采样 profile
bash launch.sh \
    --profile upsample \
    --task tube \
    --verbose

# 在机器人上：RTC client profile（需要服务端以 --server-mode rtc 启动）
bash launch.sh \
    --profile rtc \
    --task tube \
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
   - standard 模式：服务器返回动作序列（action_horizon × 14），客户端本地排队执行
   - rtc 模式：客户端每个控制 tick 输入当前观测，后台请求下一段 chunk，并立即返回单帧可执行动作

4. **动作执行**：
   - standard 模式执行前N个动作（execute_horizon），丢弃剩余动作防止误差累积
   - rtc 模式持续执行当前 chunk，同时用 RTC prefix guidance 异步生成下一段 chunk
   - 根据模式二值化夹爪值

5. **控制循环**：
   - 以指定频率发布关节命令
   - 动作队列耗尽时重新推理

### 夹爪处理

策略输出连续的夹爪值，会被二值化：

- **low_threshold**（默认）：阈值3.5
  - > 3.5 → 4.9（张开）
  - ≤ 3.5 → 0.5（闭合）

- **hard**：阈值4.25
  - > 4.25 → 4.9（张开）
  - ≤ 4.25 → 0.5（闭合）

- **soft**：软二值化，过渡区间0.5
  - > 4.0 → 4.9（张开）
  - < 3.0 → 0.5（闭合）
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

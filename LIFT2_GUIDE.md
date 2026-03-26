# LIFT2 训练和部署指南

本指南适用于 LIFT2 双臂机器人的任何任务。

## 数据格式要求

HDF5 数据集应包含以下字段：
- `observations/eef`: [T, 14] 绝对 EEF 位置 (xyz, rpy, gripper) × 2
- `observations/images/head`: [T, N] JPEG 编码的头部相机图像
- `observations/images/left_wrist`: [T, N] JPEG 编码的左腕相机图像
- `observations/images/right_wrist`: [T, N] JPEG 编码的右腕相机图像

## 完整流程

**注意**：本指南中的所有 OpenPI 核心脚本都使用 `uv run` 运行。部分脚本使用直接的 `python` 命令是因为：
- 数据分析脚本（如 `analyze_dataset0319_left_arm.py`）：辅助工具，不是核心流程
- 测试脚本（如 `test_lift2_client.py`）：独立测试工具
- **机器人客户端（如 `client_lift2.py`）：在远程机器人上运行，有独立的 Python 环境，不使用 uv**

**架构说明**：
- **Policy Server**：在训练机/服务器上运行（使用 `uv run scripts/serve_policy.py`）
- **Robot Client**：在机器人上运行（使用 `python client_lift2.py`）
- 两者通过 WebSocket 网络连接（局域网或互联网）

### 步骤 0：分析数据集（可选但推荐）

在开始训练之前，建议先分析数据集以了解数据特征：

```bash
# 分析左臂数据（适用于单臂任务）
.venv/bin/python analyze_dataset0319_left_arm.py
```

这个脚本会分析：
- 左臂和右臂的运动范围
- 夹爪的开合状态分布
- 各关节的均值和标准差
- 识别是否为单臂任务（一侧手臂不动）

**为什么要分析数据？**
- 确认数据格式正确
- 了解任务特征（单臂/双臂）
- 检查夹爪值范围是否正常
- 发现潜在的数据问题（如某个关节一直不动）

示例输出：
```
左臂数据 (维度 0-6):
  均值: [-0.0008  0.0008  0.0066 -0.0037 -0.0033 -0.0001  4.3132]
  标准差: [0.0002 0.0003 0.0004 0.0001 0.0005 0.0002 1.565 ]

左臂运动分析:
  总运动量: 0.2014
  ⚠️  左臂几乎不动！
```

如果发现左臂不动（单臂任务），在训练和推理时需要特别注意状态归一化。

### 步骤 1：数据转换

将 HDF5 数据集转换为 LeRobot 格式：

```bash
uv run convert_hdf5_to_lerobot_eef.py \
    --data-dir ./datasets_mytask \
    --repo-id mytask_eef \
    --task-description "describe your task here"
```

如果原始数据是 **60Hz 采集**，想在转换时直接降采样为 **30Hz**：

```bash
uv run convert_hdf5_to_lerobot_eef.py \
    --data-dir ./dataset0319 \
    --repo-id 0319_pick_and_place_block_30hz \
    --task-description "Put the block on the plate." \
    --source-fps 60 \
    --fps 30
```

参数说明：
- `--data-dir`: 包含 `episode_*.hdf5` 文件的目录
- `--repo-id`: 数据集 ID，将保存到 `~/.cache/huggingface/lerobot/{repo_id}`
- `--task-description`: 任务描述，用于语言指令
- `--source-fps`: 原始 HDF5 数据的采集频率；高于 `--fps` 时会在转换时按整数倍降采样
- `--fps`: 输出 LeRobot 数据集的频率

转换后的数据集格式：
- 状态：14 维绝对 EEF + 归一化夹爪 [0, 1]
- 动作：14 维 delta EEF (xyz, rpy 为增量) + 归一化夹爪 [0, 1]

### 步骤 2：计算 Normalization Statistics

**重要：训练前必须完成此步骤**

有两种方法：

#### 方法 1：在 config.py 中创建任务专用配置（推荐）

在 `src/openpi/training/config.py` 中添加你的任务配置（参考 `pi05_plate2left_eef_lora` 的格式）：

```python
TrainConfig(
    name="pi05_mytask_lora",
    model=pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=16,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ),
    data=LeRobotLift2DataConfig(
        repo_id="mytask_eef",
        default_prompt="describe your task here",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    num_train_steps=20_000,
    batch_size=4,
    freeze_filter=pi0_config.Pi0Config(
        pi05=True,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    ).get_freeze_filter(),
    ema_decay=None,
),
```

然后运行：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_mytask_lora
```

#### 方法 2：直接修改通用配置中的 CHANGE_ME

在 `src/openpi/training/config.py` 中找到 `pi05_lift2_lora` 配置，临时修改：

```python
data=LeRobotLift2DataConfig(
    repo_id="mytask_eef",  # 改成你的 repo_id
    default_prompt="describe your task here",  # 改成你的任务描述
),
```

然后运行：

```bash
uv run scripts/compute_norm_stats.py --config-name pi05_lift2_lora
```

这个过程可能需要 10-20 分钟。完成后会在以下位置生成 norm_stats：
`assets/pi05_mytask_lora/mytask_eef/` 或 `assets/pi05_lift2_lora/mytask_eef/`

### 步骤 3：开始训练

使用你在步骤 2 中创建的配置名称：

```bash
# 如果你创建了专用配置 pi05_mytask_lora
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_mytask_lora \
    --exp-name mytask_exp1

# 或者如果你修改了通用配置 pi05_lift2_lora
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_lift2_lora \
    --exp-name mytask_exp1
```

训练参数说明：
- `--exp-name`: 实验名称
- 训练进度保存到 `checkpoints/pi05_mytask_lora/mytask_exp1/` 或 `checkpoints/pi05_lift2_lora/mytask_exp1/`
- 可在 Weights & Biases 查看训练进度

### 步骤 4：启动 Policy Server

训练完成后，启动 policy server：

```bash
# 使用训练好的 checkpoint (假设训练到 20000 步)
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_mytask_lora \
    --policy.dir=checkpoints/pi05_mytask_lora/mytask_exp1/20000

# 或者如果你用的是通用配置
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_lift2_lora \
    --policy.dir=checkpoints/pi05_lift2_lora/mytask_exp1/20000
```

服务器默认监听端口 8000。

#### 启用调试打印（可选）

如果需要查看模型的输入状态和输出动作，可以启用调试打印：

```bash
# 启用调试打印，每步都打印
uv run scripts/serve_policy.py \
    --debug-print \
    --debug-interval 1 \
    policy:checkpoint \
    --policy.config=pi05_mytask_lora \
    --policy.dir=checkpoints/pi05_mytask_lora/mytask_exp1/20000

# 每 10 步打印一次（减少输出量）
uv run scripts/serve_policy.py \
    --debug-print \
    --debug-interval 10 \
    policy:checkpoint \
    --policy.config=pi05_mytask_lora \
    --policy.dir=checkpoints/pi05_mytask_lora/mytask_exp1/20000
```

调试打印会显示：
- **模型输入**：state 的 shape、dtype、values、range、mean、std
- **模型输出**：actions 的 shape、dtype、values、range、mean、std
- **双臂分析**：自动识别 14 维数据，分别显示左臂 [0-6] 和右臂 [7-13]
- **推理时间**：每步的推理耗时（毫秒）

这对于调试数据格式问题、分析模型行为非常有用。

### 步骤 5：测试推理

**在测试机上**（可以是任何能连接到 policy server 的机器）：

1. 安装 openpi-client：
```bash
cd /path/to/openpi/packages/openpi-client
pip install -e .
```

2. 运行测试脚本：
```bash
python test_lift2_client.py \
    --host <policy_server_ip> \
    --port 8000 \
    --prompt "describe your task here"
```

**注意**：`<policy_server_ip>` 是运行 policy server 的机器的 IP 地址（步骤 4 启动时会显示）。

### 步骤 6：实际机器人部署

**在机器人上**运行部署客户端（不是在 policy server 上）。

推荐使用 `launch.sh` 启动脚本：

```bash
cd openpi-on-LIFT2

# 基本启动（默认启用 60Hz 上采样）
bash launch.sh --host <policy_server_ip> --language_instruction "describe your task here"

# 禁用上采样（使用 30Hz 控制）
bash launch.sh --host <policy_server_ip> --language_instruction "describe your task here" --no_upsample

# 启用单步调试模式（每步按 Enter 执行）
bash launch.sh --host <policy_server_ip> --language_instruction "describe your task here" --debug

# 启用详细日志
bash launch.sh --host <policy_server_ip> --language_instruction "describe your task here" --verbose
```

`launch.sh` 支持的参数：
- `--host IP`: Policy server IP 地址（默认 192.168.101.101）
- `--port PORT`: 端口（默认 7777）
- `--language_instruction TEXT`: 任务语言指令
- `--no_upsample`: 禁用动作上采样（默认启用 30Hz→60Hz 上采样）
- `--action_chunk_size N`: 从预测中使用的帧数（默认 10）
- `--verbose`: 详细日志
- `--debug`: 单步调试模式，每步按 Enter 执行，显示动作详情

**动作上采样说明**：
- 默认启用 30Hz → 60Hz 上采样，让机器人运动更平滑
- 模型预测 10 帧 @ 30Hz，上采样为 19 帧 @ 60Hz
- 执行时间：19 帧 @ 60Hz = 317ms，推理余量充足
- 如果不需要上采样，使用 `--no_upsample` 回退到 30Hz 控制

也可以直接运行 `client_lift2.py`（更多参数可用）：

```bash
cd openpi-on-LIFT2/deploy

# 默认 30Hz 模式
python client_lift2.py \
    --host <policy_server_ip> \
    --port 7777 \
    --language_instruction "describe your task here" \
    --publish_rate 30 \
    --execute_horizon 10

# 启用 60Hz 上采样模式
python client_lift2.py \
    --host <policy_server_ip> \
    --port 7777 \
    --language_instruction "describe your task here" \
    --publish_rate 60 \
    --execute_horizon 19 \
    --enable_upsample \
    --action_chunk_size 10 \
    --target_hz 60 \
    --source_hz 30
```

**重要**：
- `<policy_server_ip>` 是步骤 4 中启动 policy server 的机器的 IP 地址
- 确保机器人和 policy server 在同一局域网内，或者网络可达
- 机器人需要安装 `openpi-client` 包

可选参数：
- `--debug`: 启用调试模式，每步需要按 Enter 确认
- `--verbose`: 详细日志
- `--log_latency`: 记录推理延迟
- `--auto_init`: 自动移动到初始位置
- `--left_init_pose x y z roll pitch yaw gripper`: 左臂初始位置（gripper 归一化 0-1）
- `--right_init_pose x y z roll pitch yaw gripper`: 右臂初始位置（gripper 归一化 0-1）
- `--enable_upsample`: 启用动作上采样（30Hz → 60Hz）
- `--action_chunk_size N`: 从预测中使用的帧数
- `--target_hz N`: 目标控制频率
- `--source_hz N`: 模型预测频率

## 代码集成示例

在实际机器人代码中集成 policy：

```python
from openpi_client import image_tools
from openpi_client import websocket_client_policy

# 初始化客户端
client = websocket_client_policy.WebsocketClientPolicy(
    host="<policy_server_ip>",
    port=8000
)

# 在控制循环中
for step in range(num_steps):
    # 获取机器人观测
    head_img = get_head_camera()  # 480x640x3
    left_wrist_img = get_left_wrist_camera()  # 480x640x3
    right_wrist_img = get_right_wrist_camera()  # 480x640x3
    eef_state = get_eef_state()  # 14-dim [xyz, rpy, gripper] × 2

    # 构造观测字典
    observation = {
        "observation.images.head": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(head_img, 224, 224)
        ),
        "observation.images.left_wrist": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(left_wrist_img, 224, 224)
        ),
        "observation.images.right_wrist": image_tools.convert_to_uint8(
            image_tools.resize_with_pad(right_wrist_img, 224, 224)
        ),
        "observation.state": eef_state,
        "prompt": "describe your task here",
    }

    # 获取动作
    action_chunk = client.infer(observation)["actions"]

    # 执行动作 (通常每 N 步调用一次 policy，其余步骤执行 action chunk)
    execute_action(action_chunk[0])
```

## 数据格式说明

### HDF5 输入格式
- 图像：JPEG 压缩，解码后 480x640x3
- EEF 状态：14 维 [left_xyz(3), left_rpy(3), left_gripper(1), right_xyz(3), right_rpy(3), right_gripper(1)]
- 夹爪原始范围：[0, 5] (0=闭合, 5=打开)

### LeRobot 输出格式
- 图像：480x640x3 RGB
- 状态：14 维绝对 EEF + 归一化夹爪 [0, 1]
- 动作：14 维 delta EEF (xyz/rpy 为增量) + 归一化夹爪 [0, 1]

### 训练和推理
- 模型输入：绝对 EEF 状态 + 归一化夹爪
- 模型输出：delta EEF 动作 + 归一化夹爪
- 推理时：累积 delta 得到绝对位置，反归一化夹爪后发送给机器人

## 故障排除

### 训练内存不足
- 使用 LoRA 配置 (`pi05_lift2_lora`)
- 减小 batch_size (通过 `--batch-size` 参数)
- 使用 FSDP：`--fsdp-devices <num_gpus>`

### Norm stats 计算被中断
- 重新运行 `compute_norm_stats.py`
- 确保有足够的磁盘空间和内存

### Policy server 连接失败
- 检查防火墙设置
- 确认端口 8000 未被占用
- 验证网络连接

### 机器人动作异常
- 使用 `--debug` 模式逐步检查每个动作
- 检查初始位置是否正确
- 验证 EEF 状态的坐标系是否一致

### 客户端单步调试模式

在机器人客户端启用 `--debug` 参数，可以逐步执行推理动作，每一步都需要按 Enter 才会执行，方便观察每步的动作值是否合理：

```bash
python client_lift2.py \
    --host <policy_server_ip> \
    --language_instruction "describe your task here" \
    --debug
```

启用后：
- 每步会打印左右臂的 xyz、rpy 和夹爪值
- 需要按 **Enter** 才执行当前步的动作
- 按 **Ctrl+C** 可随时中止

输出示例：
```
[Debug Step    0] Left:  xyz=[0.1, 0.2, 0.3], rpy=[0.0, 0.0, 0.0], gripper=0.00
[Debug Step    0] Right: xyz=[0.1, 0.2, 0.3], rpy=[0.0, 0.0, 0.0], gripper=5.00
Step 0: Press Enter to execute (Ctrl+C to abort)...
```

### 调试模型输入输出

如果模型行为异常（如动作不合理、夹爪状态错误），使用调试打印功能：

1. **启动带调试的 policy server**：
```bash
uv run scripts/serve_policy.py \
    --debug-print \
    --debug-interval 1 \
    policy:checkpoint \
    --policy.config=pi05_mytask_lora \
    --policy.dir=checkpoints/pi05_mytask_lora/mytask_exp1/20000
```

2. **检查输出信息**：

调试打印会遍历 observation 和 action 的所有 key，对 numpy 数组显示 shape、dtype、range 和完整值：

```
[DEBUG] Step 0 - MODEL INPUT
obs['state']: shape=(14,), dtype=float64, range=[-0.1234, 5.0000]
  values: [0.1 0.2 ...]
obs['prompt']: pick up the cup

[DEBUG] Step 0 - MODEL OUTPUT
action['actions']: shape=(10, 14), dtype=float32, range=[-0.5, 0.8]
  values: [...]
```

   - **状态输入异常**：检查 `obs['state']` 的 range，确认是否在合理范围内
   - **动作输出异常**：检查 `action['actions']` 的 range，确认是否符合预期
   - **夹爪值检查**：确认夹爪值在 [0, 1] 范围内（归一化后）

3. **常见问题**：
   - 如果 state 的 mean 远离 0，可能是归一化统计有问题，需要重新计算 norm_stats
   - 如果 actions 的 range 异常大（如 >10），可能是模型未收敛或数据有问题
   - 如果左臂不动但 state 中左臂值变化很大，可能需要在推理时固定左臂状态

### 单臂任务的特殊处理

如果数据分析显示这是单臂任务（一侧手臂不动），推理时不动手臂的实际位姿可能与训练数据差异很大，导致归一化后的值超出 [-1, +1] 范围（Out-of-Distribution），影响模型输出。

**推荐方案**：在 policy server 启动时使用 `--fix-left-arm` 或 `--fix-right-arm` 参数，将不动的手臂 state 自动固定为训练数据的均值。

```bash
# 右臂任务（左臂不动）：固定左臂 state
uv run scripts/serve_policy.py \
    --fix-left-arm \
    policy:checkpoint \
    --policy.config=pi05_mytask_lora \
    --policy.dir=checkpoints/pi05_mytask_lora/mytask_exp1/20000

# 左臂任务（右臂不动）：固定右臂 state
uv run scripts/serve_policy.py \
    --fix-right-arm \
    policy:checkpoint \
    --policy.config=pi05_mytask_lora \
    --policy.dir=checkpoints/pi05_mytask_lora/mytask_exp1/20000
```

启动时会显示固定的 state 值：
```
INFO:root:Fixing left arm state to training mean: [-3.3424050e-04  1.4505737e-05  1.7553023e-03 ...]
```

这样可以避免不动的手臂位姿差异干扰模型推理，确保模型输入始终在训练分布范围内。

## 示例：Plate2Left 任务

作为参考，这里是 plate2left 任务的完整命令：

```bash
# 1. 数据转换
uv run convert_hdf5_to_lerobot_eef.py \
    --data-dir ./datasets_plate2left \
    --repo-id plate2left_eef \
    --task-description "put items from plate to left"

# 如果原始 HDF5 是 60Hz，需要转换成 30Hz 数据集：
# uv run convert_hdf5_to_lerobot_eef.py \
#     --data-dir ./dataset0319 \
#     --repo-id 0319_pick_and_place_block_30hz \
#     --task-description "Put the block on the plate." \
#     --source-fps 60 \
#     --fps 30

# 2. 计算 norm stats (使用已有的 pi05_plate2left_eef_lora 配置)
uv run scripts/compute_norm_stats.py --config-name pi05_plate2left_eef_lora

# 3. 训练
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_plate2left_eef_lora \
    --exp-name plate2left_exp1

# 4. 启动 server
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_plate2left_eef_lora \
    --policy.dir=checkpoints/pi05_plate2left_eef_lora/plate2left_exp1/20000

# 5. 部署
python client_lift2.py \
    --host localhost \
    --language_instruction "put items from plate to left"
```

## 文件清单

- `convert_hdf5_to_lerobot_eef.py`: 通用数据转换脚本
- `analyze_dataset0319_left_arm.py`: 数据集分析脚本（分析左右臂运动、夹爪状态等）
- `src/openpi/policies/lift2_policy.py`: 通用 policy 定义
- `src/openpi/policies/debug_policy.py`: 带调试打印的 policy 类（可选）
- `src/openpi/training/config.py`: 包含 `pi05_lift2` 和 `pi05_lift2_lora` 配置
- `src/openpi/serving/websocket_policy_server.py`: WebSocket policy server（支持调试打印）
- `scripts/serve_policy.py`: Policy server 启动脚本（支持 --debug-print 参数）
- `openpi-on-LIFT2/launch.sh`: 客户端启动脚本（支持 --debug 单步调试）
- `openpi-on-LIFT2/deploy/client_lift2.py`: 通用部署客户端
- `test_lift2_client.py`: 通用测试脚本

# OpenPI on LIFT2 - 快速入门指南

## 设置（一次性）

### 1. 在机器人端准备 OpenPI 客户端

```bash
# openpi-on-LIFT2 文件夹内自带 openpi_client，直接复制整个文件夹到机器人即可。
# 如需额外安装依赖，请按机器人 Python/ROS 环境安装 cv_bridge、numpy、PyYAML 等。
```

### 2. 验证 ROS 话题

```bash
# 检查相机话题
rostopic list | grep camera

# 预期输出：
# /camera_h/color/image_raw
# /camera_l/color/image_raw
# /camera_r/color/image_raw

# 检查机械臂话题
rostopic list | grep arm

# 预期输出：
# /arm_left/arm_status_ee
# /arm_right/arm_status_ee
# /arm_left_cmd
# /arm_right_cmd
```

## 运行系统

### 步骤1：启动策略服务器（GPU机器）

```bash
cd /path/to/openpi

# 启动策略服务器
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_lift2_lora \
    --policy.dir=checkpoints/pi05_lift2_lora/<exp_name>/<step>

# 默认 profile 会连接 7777 端口；请确保策略服务器监听 7777，或在客户端用 --port 覆盖。
# 你应该看到类似："Serving policy on 0.0.0.0:7777"
```

### 步骤2：启动机器人系统（机器人端）

```bash
# 终端1：启动机器人控制器
roslaunch arx_r5_controller open_double_arm.launch

# 终端2：启动相机
roslaunch realsense2_camera rs_multiple_devices.launch
```

### 步骤3：运行 OpenPI 客户端（机器人端）

```bash
cd /path/to/openpi/openpi-on-LIFT2

# 默认 profile：30Hz 直接发布
./launch.sh --task tube --verbose

# 平滑高频执行：policy/主循环 30Hz，按 policy waypoint 轨迹语义播放到 90Hz EEF command
./launch.sh --fast --task tube --verbose

# 覆盖 profile 中的 host，并启用平滑高频执行
./launch.sh --fast --host <服务器IP> --task tube --verbose

# 只保存普通三路相机视频
./launch.sh --fast --task tube --record_video

# 只保存 RTC/non-RTC compare 输出
./launch.sh --fast --task tube --compare

# 同时保存普通视频和 compare 输出
./launch.sh --fast --task tube --record_video --compare
```

## 无机器人测试

你可以使用测试脚本在没有真实机器人的情况下测试客户端逻辑：

```bash
cd /path/to/openpi/openpi-on-LIFT2
python test_client.py --host <服务器IP>
```

## 常见问题

### 问题："无法连接到策略服务器"
**解决方案**：
- 检查策略服务器端口是否可达：`nc -vz <服务器IP> 7777`
- 检查防火墙：`sudo ufw allow 7777`
- 验证网络连通性：`ping <服务器IP>`

### 问题："找不到ROS话题"
**解决方案**：
- 验证ROS是否运行：`rosnode list`
- 检查话题名称：`rostopic list`
- 如果话题名称不同，在命令行参数中调整

### 问题："夹爪无响应"
**解决方案**：
- 用 `--binarize_gripper` / `--no_binarize_gripper` 切换二值化或连续夹爪输出
- 用 `--verbose` 检查夹爪值
- 验证夹爪硬件是否正常

### 问题："延迟高"
**解决方案**：
- 使用有线连接而非WiFi
- 降低控制频率：`--publish_rate 15`
- 增加执行范围：`--execute_horizon 15`

## 安全检查清单

在真实机器人上运行前：

- [ ] 急停按钮可触及
- [ ] 机器人工作空间清空
- [ ] 相机位置正确
- [ ] 初始位姿安全
- [ ] 先用 `--wait_after_init` 测试
- [ ] 从低控制频率开始（15 Hz）

## 下一步

1. 先测试简单动作
2. 逐步增加控制频率
3. 根据需要调整夹爪二值化设置
4. 为你的任务微调 execute_horizon
5. 用 `--log_latency` 监控性能

## 获取帮助

- 用 `--verbose` 检查日志
- 查看 README.md 获取详细文档
- 测试各个组件（相机、EEF 状态话题、策略服务器）

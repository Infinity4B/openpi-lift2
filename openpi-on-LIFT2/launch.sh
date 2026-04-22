#!/bin/bash
# Launch script for OpenPI LIFT2 client

# Get script directory and change to it
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Default values
HOST="192.168.101.101"
PORT=7777
PUBLISH_RATE=60  # 上采样后的控制频率
EXECUTE_HORIZON=19  # 执行全部上采样后的帧（10 帧 @ 30Hz → 19 帧 @ 60Hz = 317ms）
ACTION_CHUNK_SIZE=10  # 从预测中取 10 帧（模型预测的全部）
MAX_PUBLISH_STEP=1000  # 最大步数，0为无限模式
TASK=""
LANGUAGE_INSTRUCTION=""
ENABLE_UPSAMPLE="--enable_upsample"  # 默认启用上采样
TARGET_HZ=60  # 目标控制频率
SOURCE_HZ=30  # 模型预测频率
VERBOSE=""
DEBUG=""

resolve_task_description() {
    case "$1" in
        tube)
            echo "Transfer the test tube from the right rack to the left rack."
            ;;
        towel)
            echo "Flatten the towel."
            ;;
        wrench)
            echo "Open the toolbox, check the items inside one by one, and find the wrench."
            ;;
        power_strip)
            echo "Move the power strip with the left arm, and press the button of the power strip with the right arm."
            ;;
        drum)
            echo "Pick up two small drumsticks and hit the small drum."
            ;;
        dice)
            echo "Roll the dice and move the small stand the specified number of squares based on the number rolled."
            ;;
        stack)
            echo "Stack the building blocks one by one with the larger ones at the bottom."
            ;;
        *)
            return 1
            ;;
    esac
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --language_instruction)
            LANGUAGE_INSTRUCTION="$2"
            shift 2
            ;;
        --infinite)
            MAX_PUBLISH_STEP=0
            shift
            ;;
        --no_upsample)
            ENABLE_UPSAMPLE=""
            PUBLISH_RATE=30
            EXECUTE_HORIZON=10
            shift
            ;;
        --action_chunk_size)
            ACTION_CHUNK_SIZE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --debug)
            DEBUG="--debug"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--host IP] [--port PORT] [--task {tube|towel|wrench|power_strip|drum|dice|stack}] [--language_instruction TEXT] [--infinite] [--no_upsample] [--action_chunk_size N] [--verbose] [--debug]"
            exit 1
            ;;
    esac
done

if [ -n "$TASK" ] && [ -z "$LANGUAGE_INSTRUCTION" ]; then
    if ! LANGUAGE_INSTRUCTION="$(resolve_task_description "$TASK")"; then
        echo "Unknown task: $TASK"
        echo "Supported tasks: tube, towel, wrench, power_strip, drum, dice, stack"
        exit 1
    fi
fi

if [ -z "$LANGUAGE_INSTRUCTION" ]; then
    LANGUAGE_INSTRUCTION="perform task"
fi

echo "=================================================="
echo "OpenPI LIFT2 Client Launcher"
echo "=================================================="
echo "Policy Server: $HOST:$PORT"
echo "Control Rate: $PUBLISH_RATE Hz"
echo "Execute Horizon: $EXECUTE_HORIZON frames"
echo "Max Steps: $([ "$MAX_PUBLISH_STEP" -le 0 ] && echo "Infinite" || echo "$MAX_PUBLISH_STEP")"
echo "Action Upsampling: $([ -n "$ENABLE_UPSAMPLE" ] && echo "Enabled (${SOURCE_HZ}Hz -> ${TARGET_HZ}Hz, chunk=$ACTION_CHUNK_SIZE)" || echo "Disabled")"
if [ -n "$TASK" ]; then
    echo "Task Preset: $TASK"
fi
echo "Language Instruction: $LANGUAGE_INSTRUCTION"
if [ -n "$DEBUG" ]; then
    echo "Debug Mode: ON (press Enter each step)"
fi
echo "=================================================="
echo ""

# Check if policy server is reachable
echo "Checking policy server connectivity..."
if timeout 2 bash -c "echo > /dev/tcp/$HOST/$PORT" 2>/dev/null; then
    echo "✓ Policy server is reachable"
else
    echo "✗ Cannot reach policy server at $HOST:$PORT"
    echo "  Please ensure the policy server is running:"
    echo "  uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_lift2_lora --policy.dir=<checkpoint_dir>"
    exit 1
fi

# Source ROS workspace
source ~/Desktop/LIFT/R5/ROS/R5_ws/devel/setup.bash

echo ""
echo "Starting OpenPI client..."
echo ""

# Run the client
CLIENT_ARGS=(
    --host "$HOST"
    --port "$PORT"
    --publish_rate "$PUBLISH_RATE"
    --execute_horizon "$EXECUTE_HORIZON"
    --max_publish_step "$MAX_PUBLISH_STEP"
    --action_chunk_size "$ACTION_CHUNK_SIZE"
    --target_hz "$TARGET_HZ"
    --source_hz "$SOURCE_HZ"
)

if [ -n "$TASK" ]; then
    CLIENT_ARGS+=(--task "$TASK")
fi

if [ -n "$LANGUAGE_INSTRUCTION" ]; then
    CLIENT_ARGS+=(--language_instruction "$LANGUAGE_INSTRUCTION")
fi

if [ -n "$ENABLE_UPSAMPLE" ]; then
    CLIENT_ARGS+=(--enable_upsample)
fi

if [ -n "$VERBOSE" ]; then
    CLIENT_ARGS+=(--verbose)
fi

if [ -n "$DEBUG" ]; then
    CLIENT_ARGS+=(--debug)
fi

python3 deploy/client_lift2.py "${CLIENT_ARGS[@]}"

#!/bin/bash
# OpenPI LIFT2 - Start robot & ROS environment
# Run this BEFORE launch.sh
# Reference: X-VLA-on-LIFT2-main/evaluation/LIFT2/run_lift2.sh

# Get script directory and change to it
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Terminal 1-3: Start CAN buses
gnome-terminal --title="CAN1" -- bash -c "cd '$SCRIPT_DIR'; cd ARX_CAN/arx_can && sudo ./arx_can1.sh; exec bash"
sleep 0.1
gnome-terminal --title="CAN3" -- bash -c "cd '$SCRIPT_DIR'; cd ARX_CAN/arx_can && sudo ./arx_can3.sh; exec bash"
sleep 0.1
gnome-terminal --title="CAN5" -- bash -c "cd '$SCRIPT_DIR'; cd ARX_CAN/arx_can && sudo ./arx_can5.sh; exec bash"
sleep 1

# Terminal 4: Start chassis controller
gnome-terminal --title="LIFT Body" -- bash -c "source /opt/ros/noetic/setup.bash 2>/dev/null || source /opt/ros/melodic/setup.bash 2>/dev/null; cd '$SCRIPT_DIR/body' && source devel/setup.bash && roslaunch arx_lift_controller lift.launch; exec bash"
sleep 1

# Terminal 5: Start dual-arm controller
gnome-terminal --title="R5 Arms" -- bash -c "source /opt/ros/noetic/setup.bash 2>/dev/null || source /opt/ros/melodic/setup.bash 2>/dev/null; cd '$SCRIPT_DIR/R5_ws' && source devel/setup.bash && roslaunch arx_r5_controller open_double_arm_xvla.launch; exec bash"
sleep 2

# Terminal 6: Start cameras
gnome-terminal --title="RealSense" -- bash -c "cd '$SCRIPT_DIR/realsense_camera' && bash realsense.sh; exec bash"
sleep 2

echo "All services started"
echo "Use ./launch.sh to start the client"

#!/bin/bash
# OpenPI LIFT2 - Start robot & ROS environment
# Run this BEFORE launch.sh
# Reference: X-VLA-on-LIFT2-main/evaluation/LIFT2/run_lift2.sh

# Use the fixed robot-side deployment path.
LIFT2_ROOT="/home/arx/Desktop/openpi-on-LIFT2"
cd "$LIFT2_ROOT"

# Terminal 1-3: Start CAN buses
gnome-terminal --title="CAN1" -- bash -c "cd '$LIFT2_ROOT/ARX_CAN/arx_can' && sudo ./arx_can1.sh; exec bash"
sleep 0.1
gnome-terminal --title="CAN3" -- bash -c "cd '$LIFT2_ROOT/ARX_CAN/arx_can' && sudo ./arx_can3.sh; exec bash"
sleep 0.1
gnome-terminal --title="CAN5" -- bash -c "cd '$LIFT2_ROOT/ARX_CAN/arx_can' && sudo ./arx_can5.sh; exec bash"
sleep 1

# Terminal 4: Start chassis controller
gnome-terminal --title="LIFT Body" -- bash -c "cd '$LIFT2_ROOT/body' && source devel/setup.bash && roslaunch '$LIFT2_ROOT/body/src/ARX_LIFT_ros/arx_lift_controller/launch/lift.launch'; exec bash"
sleep 1

# Terminal 5: Start dual-arm controller
gnome-terminal --title="R5 Arms" -- bash -c "cd '$LIFT2_ROOT/R5_ws' && source devel/setup.bash && roslaunch '$LIFT2_ROOT/R5_ws/src/arx_r5_ros/arx_r5_controller/launch/open_double_arm_xvla.launch'; exec bash"
sleep 2

# Terminal 6: Start cameras
gnome-terminal --title="RealSense" -- bash -c "cd '$LIFT2_ROOT/realsense_camera' && bash realsense.sh; exec bash"
sleep 2

echo "All services started"
echo "Use ./launch.sh to start the client"

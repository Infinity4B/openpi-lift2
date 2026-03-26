#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenPI LIFT2 Client for ARX R5 Dual-Arm Robot
End-effector delta control (delta_xyz + delta_rpy + gripper) with OpenPI remote inference
"""

import numpy as np
import time
import argparse
import collections

import rospy
import sys
import os

# Add parent directory to path to find deploy.utils
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
sys.path.append(parent_dir)

from openpi_client import image_tools
from openpi_client import websocket_client_policy

from deploy.utils.rotation import pose_to_eef, apply_eef_delta, denormalize_gripper
from deploy.utils.rosoperator import RosOperator

task_config = {
    'camera_names': ['head', 'left_wrist', 'right_wrist']
}


class OpenPIClientModel:
    """OpenPI Inference Client for EEF Delta Control"""

    def __init__(self, host, port, execute_horizon=10,
                 enable_upsample=False, action_chunk_size=10, target_hz=60, source_hz=30):
        """
        Args:
            host: Policy server host
            port: Policy server port
            execute_horizon: Number of frames to execute per inference
            enable_upsample: Enable action upsampling (30Hz -> 60Hz)
            action_chunk_size: Number of frames to use from prediction when upsampling
            target_hz: Target control frequency for upsampling
            source_hz: Source prediction frequency for upsampling
        """
        self.client = websocket_client_policy.WebsocketClientPolicy(
            host=host,
            port=port
        )
        self.execute_horizon = execute_horizon
        self.executed_count = 0
        self.enable_upsample = enable_upsample
        self.action_chunk_size = action_chunk_size
        self.target_hz = target_hz
        self.source_hz = source_hz
        self.reset()
        self.current_eef = None

    def reset(self):
        """Reset action queue at the start of each episode"""
        self.action_plan = collections.deque()
        self.executed_count = 0
        return None

    def set_current_eef(self, eef):
        """
        Update current EEF state
        Args:
            eef: (14,) Current dual-arm EEF state [xyz, rpy, gripper] × 2
        """
        self.current_eef = eef

    def upsample_actions(self, actions):
        """
        Upsample action sequence from source_hz to target_hz using linear interpolation.
        Gripper values (indices 6, 13) are not interpolated.

        Args:
            actions: List of (14,) action poses at source_hz

        Returns:
            upsampled_actions: List with interpolated frames at target_hz
        """
        if not self.enable_upsample or len(actions) == 0:
            return actions

        if self.target_hz % self.source_hz != 0:
            rospy.logwarn(f"[Upsample] target_hz ({self.target_hz}) must be integer multiple of source_hz ({self.source_hz})")
            return actions

        ratio = self.target_hz // self.source_hz
        if ratio == 1:
            return actions  # No upsampling needed

        actions = [np.array(a) for a in actions]
        upsampled = []

        for i in range(len(actions) - 1):
            current_action = actions[i]
            next_action = actions[i + 1]

            # Add current frame
            upsampled.append(current_action.copy())

            # Interpolate intermediate frames
            for j in range(1, ratio):
                alpha = j / ratio
                interpolated = current_action * (1 - alpha) + next_action * alpha

                # Keep gripper values from next action (no interpolation)
                interpolated[6] = next_action[6]    # Left gripper
                interpolated[13] = next_action[13]  # Right gripper

                upsampled.append(interpolated)

        # Add last frame
        upsampled.append(actions[-1].copy())

        return upsampled

    def step(self, obs, args):
        """
        Execute one inference step

        Args:
            obs: Observation dict with images and eef
            args: Command line arguments

        Returns:
            action: (14,) Single-frame action [xyz, rpy, gripper] × 2 (absolute pose)
        """
        if not self.action_plan:
            head_img = obs['images']['head']
            left_wrist_img = obs['images']['left_wrist']
            right_wrist_img = obs['images']['right_wrist']
            current_eef = self.current_eef.astype(np.float32)

            # Construct observation for OpenPI
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
                "observation.state": current_eef,
                "prompt": args.language_instruction,
            }

            # Call remote policy
            t0 = time.perf_counter()
            result = self.client.infer(observation)
            latency_ms = (time.perf_counter() - t0) * 1000

            if args.log_latency or args.verbose:
                rospy.loginfo(f"[Latency] single inference: {latency_ms:.1f} ms")

            action_chunk = result["actions"]  # Shape: (action_horizon, 14)
            # action_chunk contains delta actions

            # ✅ FIX: Accumulate deltas - each delta is relative to the previous predicted target
            # Training: action[i] = obs[i+1] - obs[i] (consecutive frame deltas)
            # Inference: Must accumulate deltas to get trajectory
            absolute_actions = []
            pred_eef = current_eef.copy()  # Start from current observation
            for delta_action in action_chunk:
                next_eef = apply_eef_delta(pred_eef, delta_action)  # Relative to previous prediction
                absolute_actions.append(next_eef)
                pred_eef = next_eef  # Update for next delta (accumulate)

            # Limit to action_chunk_size if upsampling is enabled
            if self.enable_upsample:
                absolute_actions = absolute_actions[:self.action_chunk_size]

            # Apply action upsampling if enabled
            if self.enable_upsample:
                original_count = len(absolute_actions)
                absolute_actions = self.upsample_actions(absolute_actions)
                if args.verbose:
                    rospy.loginfo(f"[Upsample] {self.source_hz}Hz -> {self.target_hz}Hz: {original_count} frames -> {len(absolute_actions)} frames")

            # Cache actions to queue
            self.action_plan.extend(absolute_actions)
            self.executed_count = 0

            if args.verbose:
                rospy.loginfo(f"[Inference] Generated {len(absolute_actions)} frames, will execute {self.execute_horizon}")
                rospy.loginfo(f"[Debug] Current EEF xyz: L={current_eef[:3]}, R={current_eef[7:10]}")
                rospy.loginfo(f"[Debug] First delta xyz: L={action_chunk[0, :3]}, R={action_chunk[0, 7:10]}")
                rospy.loginfo(f"[Debug] First delta magnitude: L={np.linalg.norm(action_chunk[0, :3]):.6f}m, R={np.linalg.norm(action_chunk[0, 7:10]):.6f}m")
                rospy.loginfo(f"[Debug] First target xyz: L={absolute_actions[0][:3]}, R={absolute_actions[0][7:10]}")

                # Check if deltas are reasonable (should be mm-scale for right arm)
                right_delta_norm = np.linalg.norm(action_chunk[0, 7:10])
                if right_delta_norm < 0.0001:  # < 0.1mm
                    rospy.logwarn(f"[Warning] Right arm delta is very small ({right_delta_norm*1000:.3f}mm), may indicate denormalization issue")
                elif right_delta_norm > 0.05:  # > 50mm
                    rospy.logwarn(f"[Warning] Right arm delta is very large ({right_delta_norm*1000:.1f}mm), may cause sudden movements")

        # Pop one action frame
        action_predict = np.array(self.action_plan.popleft())
        self.executed_count += 1

        # Clear queue after execute_horizon frames to force re-inference
        if self.executed_count >= self.execute_horizon:
            if len(self.action_plan) > 0:
                if args.verbose:
                    rospy.loginfo(f"[Policy] Executed {self.execute_horizon} frames, discarding {len(self.action_plan)} remaining")
                self.action_plan.clear()
                self.executed_count = 0

        # Gripper processing
        left_gripper_norm = action_predict[6]
        right_gripper_norm = action_predict[13]

        if args.binarize_gripper:
            # Binarize and map to robot gripper range [0, 5]
            # Avoid extreme values (1.0, 4.9) for safety, matching X-VLA convention
            # Threshold: < 0.6 = closed, >= 0.6 = open
            GRIPPER_THRESHOLD = 0.6  # Normalized threshold
            GRIPPER_CLOSED = 1.0
            GRIPPER_OPEN = 4.9
            action_predict[6] = GRIPPER_OPEN if left_gripper_norm >= GRIPPER_THRESHOLD else GRIPPER_CLOSED
            action_predict[13] = GRIPPER_OPEN if right_gripper_norm >= GRIPPER_THRESHOLD else GRIPPER_CLOSED
        else:
            # Continuous mode: denormalize from [0, 1] to [0, 5]
            from deploy.utils.rotation import denormalize_gripper
            action_predict[6] = denormalize_gripper(left_gripper_norm)
            action_predict[13] = denormalize_gripper(right_gripper_norm)

        return action_predict


def get_action(args, config, ros_operator, policy, t):
    """
    Get action with intelligent sensor query strategy

    Args:
        args: Command line arguments
        config: Configuration dict
        ros_operator: ROS operator instance
        policy: ClientModel instance
        t: Current timestep

    Returns:
        action: (14,) Action [xyz, rpy, gripper] × 2
    """
    print_flag = True
    rate = rospy.Rate(args.publish_rate)

    while True and not rospy.is_shutdown():
        # Case 1: Action queue has remaining frames, use directly
        if len(policy.action_plan) > 0:
            action = policy.step(None, args)
            return action

        # Case 2: Queue empty, query sensors and inference
        result = ros_operator.get_frame()
        if not result:
            if print_flag:
                rospy.logwarn("Sensor sync failed, waiting...")
                print_flag = False
            rate.sleep()
            continue

        print_flag = True
        (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
         arm_left_pose, arm_right_pose) = result

        # Construct observation dict
        obs = collections.OrderedDict()
        image_dict = {
            config['camera_names'][0]: img_front,
            config['camera_names'][1]: img_left,
            config['camera_names'][2]: img_right
        }
        obs['images'] = image_dict

        if args.use_depth_image:
            image_depth_dict = {
                config['camera_names'][0]: img_front_depth,
                config['camera_names'][1]: img_left_depth,
                config['camera_names'][2]: img_right_depth
            }
            obs['images_depth'] = image_depth_dict

        # Current EEF state
        obs['eef'] = pose_to_eef(arm_left_pose, arm_right_pose)

        # Inference
        policy.set_current_eef(obs['eef'])
        action = policy.step(obs, args)

        return action


def move_to_init_pose(ros_operator, left_init, right_init, duration=3.0, rate_hz=15):
    """
    Smoothly move to initial pose using end-effector control

    Args:
        ros_operator: ROS operator instance
        left_init: (7,) Left arm initial pose [x,y,z,roll,pitch,yaw,gripper(0-1)]
        right_init: (7,) Right arm initial pose
        duration: Movement duration (seconds)
        rate_hz: Control frequency
    """
    rospy.loginfo("Moving to initial pose...")

    # Wait for current pose
    rate = rospy.Rate(rate_hz)
    while len(ros_operator.arm_left_pose_deque) == 0 or len(ros_operator.arm_right_pose_deque) == 0:
        rospy.loginfo("Waiting for arm state data...")
        rate.sleep()

    # Get current pose
    left_current_msg = ros_operator.arm_left_pose_deque[-1]
    right_current_msg = ros_operator.arm_right_pose_deque[-1]

    left_current = pose_to_eef(left_current_msg, right_current_msg)[:7]
    right_current = pose_to_eef(left_current_msg, right_current_msg)[7:14]

    left_init = np.array(left_init)
    right_init = np.array(right_init)

    rospy.loginfo(f"Left arm: {left_current[:3]} → {left_init[:3]}")
    rospy.loginfo(f"Right arm: {right_current[:3]} → {right_init[:3]}")

    total_steps = int(duration * rate_hz)

    # Linear interpolation
    for step in range(total_steps + 1):
        alpha = step / total_steps

        left_target = left_current * (1 - alpha) + left_init * alpha
        right_target = right_current * (1 - alpha) + right_init * alpha

        # Denormalize gripper for publishing
        left_target_pub = left_target.copy()
        right_target_pub = right_target.copy()
        left_target_pub[6] = denormalize_gripper(left_target[6])
        right_target_pub[6] = denormalize_gripper(right_target[6])

        ros_operator.eef_arm_publish(left_target_pub.tolist(), right_target_pub.tolist())

        if step % 15 == 0:
            progress = int(alpha * 100)
            rospy.loginfo(f"Progress: {progress}% ({step}/{total_steps})")

        rate.sleep()

    rospy.loginfo("Reached initial pose")


def model_inference(args, config, ros_operator):
    """
    Main inference loop

    Args:
        args: Command line arguments
        config: Configuration dict
        ros_operator: ROS operator instance
    """
    policy = OpenPIClientModel(
        args.host,
        args.port,
        execute_horizon=args.execute_horizon,
        enable_upsample=args.enable_upsample,
        action_chunk_size=args.action_chunk_size,
        target_hz=args.target_hz,
        source_hz=args.source_hz
    )
    max_publish_step = config['episode_len']

    # Initial pose (normalized gripper [0, 1])
    left_init = args.left_init_pose if args.left_init_pose else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
    right_init = args.right_init_pose if args.right_init_pose else [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]

    # Auto move to initial pose
    if args.auto_init:
        move_to_init_pose(ros_operator, left_init, right_init,
                         duration=args.init_duration, rate_hz=args.publish_rate)
        if args.wait_after_init:
            input("Press Enter to start inference...")

    # Main inference loop
    start_time = time.time()
    count = 0

    policy.reset()
    t = 0
    rate = rospy.Rate(args.publish_rate)

    # Infinite loop if max_publish_step is 0 or negative
    run_infinite = max_publish_step <= 0

    while (run_infinite or t < max_publish_step) and not rospy.is_shutdown():
        action = get_action(args, config, ros_operator, policy, t)

        duration = time.time() - start_time
        count += 1

        if args.verbose and t % 50 == 0:
            rospy.loginfo(f"Average Hz: {count/duration:.2f}")

        # Split dual-arm actions
        left_action = action[:7]
        right_action = action[7:14]

        # Gripper values are already in [0, 5] range from step() method
        # No additional processing needed

        # Debug mode: print action details and wait for keypress
        if args.debug:
            rospy.loginfo(f"[Debug Step {t:4d}] Left:  xyz={left_action[:3]}, rpy={left_action[3:6]}, gripper={left_action[6]:.2f}")
            rospy.loginfo(f"[Debug Step {t:4d}] Right: xyz={right_action[:3]}, rpy={right_action[3:6]}, gripper={right_action[6]:.2f}")
            try:
                input(f"Step {t}: Press Enter to execute (Ctrl+C to abort)...")
            except (KeyboardInterrupt, EOFError):
                rospy.loginfo("Debug mode: user aborted")
                break

        # Publish to ROS
        ros_operator.eef_arm_publish(left_action, right_action)

        if t % 10 == 0:
            rospy.loginfo(f"[Step {t:4d}] L_gripper={left_action[6]:.2f}, R_gripper={right_action[6]:.2f}")

        t += 1
        rate.sleep()

    if run_infinite:
        rospy.loginfo("Infinite mode interrupted, executed {t} steps")
    else:
        rospy.loginfo(f"Episode completed, executed {t} steps")


def get_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='OpenPI LIFT2 Client - EEF Delta Control')

    # Policy server
    parser.add_argument('--host', type=str, default="192.168.101.101",
                        help='Policy server host')
    parser.add_argument('--port', type=int, default=7777,
                        help='Policy server port')

    # Task configuration
    parser.add_argument('--language_instruction', type=str,
                        default='perform task',
                        help='Language instruction')
    parser.add_argument('--max_publish_step', type=int, default=1000,
                        help='Maximum execution steps (0 or negative for infinite mode)')

    # Control parameters
    parser.add_argument('--publish_rate', type=int, default=30,
                        help='Control frequency (Hz)')
    parser.add_argument('--execute_horizon', type=int, default=10,
                        help='Frames to execute per inference')

    # Action upsampling (30Hz -> 60Hz)
    parser.add_argument('--enable_upsample', action='store_true', default=False,
                        help='Enable action upsampling (e.g., 30Hz -> 60Hz)')
    parser.add_argument('--action_chunk_size', type=int, default=10,
                        help='Number of frames to use from prediction when upsampling (default: 10)')
    parser.add_argument('--target_hz', type=int, default=60,
                        help='Target control frequency for upsampling (default: 60)')
    parser.add_argument('--source_hz', type=int, default=30,
                        help='Source prediction frequency for upsampling (default: 30)')

    # Initialization
    parser.add_argument('--auto_init', action='store_true', default=True,
                        help='Auto move to initial pose on startup')
    parser.add_argument('--no_auto_init', action='store_false', dest='auto_init',
                        help='Disable auto initialization')
    parser.add_argument('--init_duration', type=float, default=3.0,
                        help='Duration to move to initial pose (seconds)')
    parser.add_argument('--wait_after_init', action='store_true', default=False,
                        help='Wait for user confirmation after reaching initial pose')
    parser.add_argument('--left_init_pose', type=float, nargs=7, default=None,
                        help='Left arm initial pose [x y z roll pitch yaw gripper(0-1)]')
    parser.add_argument('--right_init_pose', type=float, nargs=7, default=None,
                        help='Right arm initial pose [x y z roll pitch yaw gripper(0-1)]')

    # Camera topics
    parser.add_argument('--img_front_topic', type=str, default='/camera_h/color/image_raw',
                        help='Front camera topic (head camera)')
    parser.add_argument('--img_left_topic', type=str, default='/camera_l/color/image_raw',
                        help='Left wrist camera topic')
    parser.add_argument('--img_right_topic', type=str, default='/camera_r/color/image_raw',
                        help='Right wrist camera topic')

    parser.add_argument('--img_front_depth_topic', type=str, default='/camera_h/depth/image_rect_raw',
                        help='Front depth camera topic')
    parser.add_argument('--img_left_depth_topic', type=str, default='/camera_l/depth/image_rect_raw',
                        help='Left wrist depth camera topic')
    parser.add_argument('--img_right_depth_topic', type=str, default='/camera_r/depth/image_rect_raw',
                        help='Right wrist depth camera topic')

    parser.add_argument('--use_depth_image', action='store_true', help='Use depth images')

    # Arm topics (EEF pose)
    parser.add_argument('--arm_left_pose_topic', type=str, default='/arm_left/arm_status_ee',
                        help='Left arm end-effector pose topic')
    parser.add_argument('--arm_right_pose_topic', type=str, default='/arm_right/arm_status_ee',
                        help='Right arm end-effector pose topic')

    parser.add_argument('--arm_left_cmd_topic', type=str, default='/arm_left_cmd',
                        help='Left arm command topic')
    parser.add_argument('--arm_right_cmd_topic', type=str, default='/arm_right_cmd',
                        help='Right arm command topic')

    # Debug mode
    parser.add_argument('--debug', action='store_true', default=False,
                        help='Debug mode: press Enter to execute each action step')

    # Gripper control
    parser.add_argument('--binarize_gripper', action='store_true', default=True,
                        help='Binarize gripper to 0/1 (default: True)')
    parser.add_argument('--no_binarize_gripper', action='store_false', dest='binarize_gripper',
                        help='Keep continuous gripper values [0,1]')

    # Verbose / latency
    parser.add_argument('--verbose', action='store_true', default=False,
                        help='Enable verbose logging')
    parser.add_argument('--log_latency', action='store_true', default=False,
                        help='Log single-inference latency (ms) each time')

    args = parser.parse_args()
    return args


def main():
    """Main function"""
    args = get_arguments()

    rospy.init_node('openpi_lift2_client', anonymous=True)

    rospy.loginfo("="*50)
    rospy.loginfo("OpenPI LIFT2 Client Starting (EEF Delta Control)")
    rospy.loginfo(f"Policy server: {args.host}:{args.port}")
    rospy.loginfo(f"Control rate: {args.publish_rate} Hz")
    rospy.loginfo(f"Execute horizon: {args.execute_horizon} frames")
    rospy.loginfo(f"Action upsampling: {'Enabled' if args.enable_upsample else 'Disabled'}")
    if args.enable_upsample:
        rospy.loginfo(f"  {args.source_hz}Hz -> {args.target_hz}Hz (chunk size: {args.action_chunk_size})")
    rospy.loginfo(f"Auto initialization: {'Enabled' if args.auto_init else 'Disabled'}")
    if args.auto_init:
        rospy.loginfo(f"  Init duration: {args.init_duration}s")
        if args.left_init_pose:
            rospy.loginfo(f"  Left target: {args.left_init_pose}")
        if args.right_init_pose:
            rospy.loginfo(f"  Right target: {args.right_init_pose}")
    rospy.loginfo(f"Language instruction: {args.language_instruction}")
    if args.debug:
        rospy.loginfo("** DEBUG MODE: Press Enter to execute each step **")
    rospy.loginfo("="*50)

    # Initialize ROS operator
    ros_operator = RosOperator(args)

    # Configuration
    config = {
        'episode_len': args.max_publish_step,
        'camera_names': task_config['camera_names'],
    }

    # Start inference
    model_inference(args, config, ros_operator)


if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Program interrupted")
    except KeyboardInterrupt:
        rospy.loginfo("User terminated program")
    except Exception as e:
        rospy.logerr(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

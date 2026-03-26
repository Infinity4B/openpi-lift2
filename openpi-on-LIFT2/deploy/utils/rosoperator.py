#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS Operator for OpenPI LIFT2 Client (EEF Control)
Handles sensor data collection and end-effector pose control for ARX R5 dual-arm robot
"""

import rospy
from sensor_msgs.msg import Image
from arm_control.msg import PosCmd
from cv_bridge import CvBridge
from collections import deque
import numpy as np


class RosOperator:
    """ROS Operator: Manages all ROS topic subscriptions and publications"""

    def __init__(self, args):
        """
        Initialize ROS operator

        Args:
            args: Command line arguments containing all topic names
        """
        self.args = args
        self.bridge = CvBridge()

        # Initialize data queues
        self.img_left_deque = deque()
        self.img_right_deque = deque()
        self.img_front_deque = deque()
        self.img_left_depth_deque = deque()
        self.img_right_depth_deque = deque()
        self.img_front_depth_deque = deque()

        # End-effector pose queues
        self.arm_left_pose_deque = deque()
        self.arm_right_pose_deque = deque()

        # Publishers
        self.arm_left_cmd_publisher = None
        self.arm_right_cmd_publisher = None

        # Initialize ROS topics
        self.init_ros()

    def init_ros(self):
        """Initialize ROS subscribers and publishers"""
        # Note: rospy.init_node is called in main(), not here

        # ========== Subscribe to camera topics ==========
        rospy.Subscriber(self.args.img_left_topic, Image, self.img_left_callback,
                        queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_right_topic, Image, self.img_right_callback,
                        queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.img_front_topic, Image, self.img_front_callback,
                        queue_size=1000, tcp_nodelay=True)

        if hasattr(self.args, 'use_depth_image') and self.args.use_depth_image:
            rospy.Subscriber(self.args.img_left_depth_topic, Image, self.img_left_depth_callback,
                            queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_right_depth_topic, Image, self.img_right_depth_callback,
                            queue_size=1000, tcp_nodelay=True)
            rospy.Subscriber(self.args.img_front_depth_topic, Image, self.img_front_depth_callback,
                            queue_size=1000, tcp_nodelay=True)

        # ========== Subscribe to arm end-effector pose topics ==========
        rospy.Subscriber(self.args.arm_left_pose_topic, PosCmd,
                        self.arm_left_pose_callback, queue_size=1000, tcp_nodelay=True)
        rospy.Subscriber(self.args.arm_right_pose_topic, PosCmd,
                        self.arm_right_pose_callback, queue_size=1000, tcp_nodelay=True)

        # ========== Create publishers (control commands) ==========
        self.arm_left_cmd_publisher = rospy.Publisher(self.args.arm_left_cmd_topic,
                                                      PosCmd, queue_size=10)
        self.arm_right_cmd_publisher = rospy.Publisher(self.args.arm_right_cmd_topic,
                                                       PosCmd, queue_size=10)

        rospy.loginfo("ROS Operator initialized (EEF Control)")

    # ==================== Camera callbacks ====================
    def img_left_callback(self, msg):
        if len(self.img_left_deque) >= 2000:
            self.img_left_deque.popleft()
        self.img_left_deque.append(msg)

    def img_right_callback(self, msg):
        if len(self.img_right_deque) >= 2000:
            self.img_right_deque.popleft()
        self.img_right_deque.append(msg)

    def img_front_callback(self, msg):
        if len(self.img_front_deque) >= 2000:
            self.img_front_deque.popleft()
        self.img_front_deque.append(msg)

    def img_left_depth_callback(self, msg):
        if len(self.img_left_depth_deque) >= 2000:
            self.img_left_depth_deque.popleft()
        self.img_left_depth_deque.append(msg)

    def img_right_depth_callback(self, msg):
        if len(self.img_right_depth_deque) >= 2000:
            self.img_right_depth_deque.popleft()
        self.img_right_depth_deque.append(msg)

    def img_front_depth_callback(self, msg):
        if len(self.img_front_depth_deque) >= 2000:
            self.img_front_depth_deque.popleft()
        self.img_front_depth_deque.append(msg)

    # ==================== Arm end-effector pose callbacks ====================
    def arm_left_pose_callback(self, msg):
        """
        Callback for left arm end-effector pose

        Args:
            msg: PosCmd message with x, y, z, roll, pitch, yaw, gripper
        """
        if len(self.arm_left_pose_deque) >= 2000:
            self.arm_left_pose_deque.popleft()
        self.arm_left_pose_deque.append(msg)

    def arm_right_pose_callback(self, msg):
        """
        Callback for right arm end-effector pose

        Args:
            msg: PosCmd message with x, y, z, roll, pitch, yaw, gripper
        """
        if len(self.arm_right_pose_deque) >= 2000:
            self.arm_right_pose_deque.popleft()
        self.arm_right_pose_deque.append(msg)

    # ==================== Synchronized data acquisition ====================
    def get_frame(self):
        """
        Get all sensor data with timestamp synchronization

        Returns:
            tuple: (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                    arm_left_pose, arm_right_pose)
            False: Data not ready or timestamps not aligned
        """
        # Check if basic data is ready
        if len(self.img_left_deque) == 0 or len(self.img_right_deque) == 0 or len(self.img_front_deque) == 0:
            return False

        if hasattr(self.args, 'use_depth_image') and self.args.use_depth_image:
            if len(self.img_left_depth_deque) == 0 or len(self.img_right_depth_deque) == 0 or len(self.img_front_depth_deque) == 0:
                return False

        if len(self.arm_left_pose_deque) == 0 or len(self.arm_right_pose_deque) == 0:
            return False

        # Get minimum timestamp
        timestamps = [
            self.img_left_deque[-1].header.stamp.to_sec(),
            self.img_right_deque[-1].header.stamp.to_sec(),
            self.img_front_deque[-1].header.stamp.to_sec(),
        ]

        if hasattr(self.args, 'use_depth_image') and self.args.use_depth_image:
            timestamps.extend([
                self.img_left_depth_deque[-1].header.stamp.to_sec(),
                self.img_right_depth_deque[-1].header.stamp.to_sec(),
                self.img_front_depth_deque[-1].header.stamp.to_sec(),
            ])

        frame_time = min(timestamps)

        # Check if all data has reached this timestamp
        if self.img_left_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.img_right_deque[-1].header.stamp.to_sec() < frame_time:
            return False
        if self.img_front_deque[-1].header.stamp.to_sec() < frame_time:
            return False

        # Pop old data and get synchronized data
        while self.img_left_deque[0].header.stamp.to_sec() < frame_time:
            self.img_left_deque.popleft()
        img_left = self.bridge.imgmsg_to_cv2(self.img_left_deque.popleft(), 'passthrough')

        while self.img_right_deque[0].header.stamp.to_sec() < frame_time:
            self.img_right_deque.popleft()
        img_right = self.bridge.imgmsg_to_cv2(self.img_right_deque.popleft(), 'passthrough')

        while self.img_front_deque[0].header.stamp.to_sec() < frame_time:
            self.img_front_deque.popleft()
        img_front = self.bridge.imgmsg_to_cv2(self.img_front_deque.popleft(), 'passthrough')

        # Depth images (optional)
        img_left_depth = None
        img_right_depth = None
        img_front_depth = None
        if hasattr(self.args, 'use_depth_image') and self.args.use_depth_image:
            while self.img_left_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_left_depth_deque.popleft()
            img_left_depth = self.bridge.imgmsg_to_cv2(self.img_left_depth_deque.popleft(), 'passthrough')

            while self.img_right_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_right_depth_deque.popleft()
            img_right_depth = self.bridge.imgmsg_to_cv2(self.img_right_depth_deque.popleft(), 'passthrough')

            while self.img_front_depth_deque[0].header.stamp.to_sec() < frame_time:
                self.img_front_depth_deque.popleft()
            img_front_depth = self.bridge.imgmsg_to_cv2(self.img_front_depth_deque.popleft(), 'passthrough')

        # End-effector poses (take latest)
        arm_left_pose = self.arm_left_pose_deque[-1]
        arm_right_pose = self.arm_right_pose_deque[-1]

        return (img_front, img_left, img_right, img_front_depth, img_left_depth, img_right_depth,
                arm_left_pose, arm_right_pose)

    # ==================== Action publishing ====================
    def eef_arm_publish(self, left, right):
        """
        Publish dual-arm end-effector pose control commands

        Args:
            left: (7,) or list - Left arm pose [x, y, z, roll, pitch, yaw, gripper]
            right: (7,) or list - Right arm pose [x, y, z, roll, pitch, yaw, gripper]
        """
        # Left arm
        pos_cmd_left = PosCmd()
        pos_cmd_left.x = left[0]
        pos_cmd_left.y = left[1]
        pos_cmd_left.z = left[2]
        pos_cmd_left.roll = left[3]
        pos_cmd_left.pitch = left[4]
        pos_cmd_left.yaw = left[5]
        pos_cmd_left.gripper = left[6]

        # Right arm
        pos_cmd_right = PosCmd()
        pos_cmd_right.x = right[0]
        pos_cmd_right.y = right[1]
        pos_cmd_right.z = right[2]
        pos_cmd_right.roll = right[3]
        pos_cmd_right.pitch = right[4]
        pos_cmd_right.yaw = right[5]
        pos_cmd_right.gripper = right[6]

        # Publish
        self.arm_left_cmd_publisher.publish(pos_cmd_left)
        self.arm_right_cmd_publisher.publish(pos_cmd_right)

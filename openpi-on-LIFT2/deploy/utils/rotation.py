#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rotation and EEF utilities for OpenPI LIFT2 client
Standard EEF delta representation (delta_xyz + delta_rpy + gripper)

Gripper range:
  Raw: [0, 5] where 0=fully closed, 5=fully open
  Normalized: [0, 1] where 0=fully closed, 1=fully open
"""

import numpy as np
from scipy.spatial.transform import Rotation as R

# Gripper normalization constants (matching X-VLA LIFT2 and training data convention)
GRIPPER_MIN = 0.0  # Fully closed
GRIPPER_MAX = 5.0  # Fully open


def normalize_gripper(gripper_raw):
    """
    将夹爪值从 [0, 5] 归一化到 [0, 1]

    参数:
        gripper_raw: 原始夹爪值

    返回:
        归一化后的夹爪值 [0, 1]
    """
    return np.clip(gripper_raw, GRIPPER_MIN, GRIPPER_MAX) / GRIPPER_MAX


def denormalize_gripper(gripper_norm):
    """
    将夹爪值从 [0, 1] 反归一化到 [0, 5]

    参数:
        gripper_norm: 归一化的夹爪值 [0, 1]

    返回:
        原始夹爪值 [0, 5]
    """
    return np.clip(gripper_norm, 0.0, 1.0) * GRIPPER_MAX


def pose_to_eef(arm_left_pose, arm_right_pose):
    """
    从ROS消息中提取双臂末端执行器的标准EEF表示（绝对位姿）

    参数:
        arm_left_pose: PosCmd 消息（左臂末端位姿）
        arm_right_pose: PosCmd 消息（右臂末端位姿）

    返回:
        (14,) [left_xyz(3), left_rpy(3), left_grip(1), right_xyz(3), right_rpy(3), right_grip(1)]
        注意：夹爪值已归一化到 [0, 1]
    """
    # 左臂
    left_xyz = np.array([arm_left_pose.x, arm_left_pose.y, arm_left_pose.z])
    left_rpy = np.array([arm_left_pose.roll, arm_left_pose.pitch, arm_left_pose.yaw])
    left_gripper = normalize_gripper(arm_left_pose.gripper)

    # 右臂
    right_xyz = np.array([arm_right_pose.x, arm_right_pose.y, arm_right_pose.z])
    right_rpy = np.array([arm_right_pose.roll, arm_right_pose.pitch, arm_right_pose.yaw])
    right_gripper = normalize_gripper(arm_right_pose.gripper)

    return np.concatenate([
        left_xyz, left_rpy, [left_gripper],
        right_xyz, right_rpy, [right_gripper]
    ])


def eef_to_pose(eef):
    """
    将EEF表示转换回双臂位姿数组

    参数:
        eef: (14,) [left_xyz(3), left_rpy(3), left_grip(1), right_xyz(3), right_rpy(3), right_grip(1)]

    返回:
        left_pose: (7,) [x, y, z, roll, pitch, yaw, gripper]
        right_pose: (7,) [x, y, z, roll, pitch, yaw, gripper]
        注意：夹爪值为归一化的 [0, 1]
    """
    left_pose = eef[:7]
    right_pose = eef[7:14]
    return left_pose, right_pose


def apply_eef_delta(current_eef, delta_eef):
    """
    将EEF delta应用到当前EEF状态

    参数:
        current_eef: (14,) 当前EEF状态 [xyz, rpy, gripper] × 2
        delta_eef: (14,) EEF delta [delta_xyz, delta_rpy, gripper] × 2

    返回:
        next_eef: (14,) 下一个EEF状态
    """
    next_eef = np.zeros(14, dtype=np.float32)

    # 左臂
    next_eef[0:3] = current_eef[0:3] + delta_eef[0:3]  # xyz + delta_xyz
    next_eef[3:6] = current_eef[3:6] + delta_eef[3:6]  # rpy + delta_rpy
    next_eef[6] = delta_eef[6]  # gripper (absolute)

    # 右臂
    next_eef[7:10] = current_eef[7:10] + delta_eef[7:10]  # xyz + delta_xyz
    next_eef[10:13] = current_eef[10:13] + delta_eef[10:13]  # rpy + delta_rpy
    next_eef[13] = delta_eef[13]  # gripper (absolute)

    return next_eef

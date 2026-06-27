#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import numpy as np

from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import JointState

from builtin_interfaces.msg import Duration

from moveit_msgs.msg import (
    RobotState,
    RobotTrajectory,
    MoveItErrorCodes,
    Constraints,
    JointConstraint,
)
from moveit_msgs.srv import GetPositionIK, GetMotionPlan, GetPositionFK
from moveit_msgs.action import ExecuteTrajectory


from typing import Union

from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from sensor_msgs.msg import JointState as js

import torch
import time
from curobo.types.math import Pose
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig

class MinimalPublisher(Node):

    def __init__(self):
        super().__init__('minimal_publisher')
        self.publisher_ = self.create_publisher(js, 'joint_states', 10)


def main(args=None):
    rclpy.init(args=args)
    world_config = {
        # "mesh": {
        #     "base_scene": {
        #         "pose": [10.5, 0.080, 1.6, 0.043, -0.471, 0.284, 0.834],
        #         "file_path": "scene/nvblox/srl_ur10_bins.obj",
        #     },
        # },
        "cuboid": {
            "box": {
                "dims": [0.2, 0.2, 0.4],  # x, y, z
                "pose": [-0.4, 0.0, 0.2, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
            },

            "table": {
                "dims": [5.0, 5.0, 0.2],  # x, y, z
                "pose": [0.0, 0.0, -0.1, 1, 0, 0, 0.0],  # x, y, z, qw, qx, qy, qz
            },
        },
    }

    motion_gen_config = MotionGenConfig.load_from_robot_config(
        "ur5e.yml",
        world_config,
        interpolation_dt=0.01,
    )
    motion_gen = MotionGen(motion_gen_config)
    motion_gen.warmup()

    retract_cfg = motion_gen.get_retract_config()

    state = motion_gen.rollout_fn.compute_kinematics(
        JointState.from_position(retract_cfg.view(1, -1))
    )

    goal_pose = Pose.from_list([-0.45, 0.4, 0.25, 0.0, 0.0, 1.0, 0.0])  # x, y, z, qw, qx, qy, qz
    initial_joint_states = torch.zeros(1, 6).cuda()
    initial_joint_states[0][0] = 38.0*np.pi/180.0
    initial_joint_states[0][1] = -61.0*np.pi/180.0
    initial_joint_states[0][2] = 77.0*np.pi/180.0
    initial_joint_states[0][3] = -106.0*np.pi/180.0
    initial_joint_states[0][4] = -87.0*np.pi/180.0
    initial_joint_states[0][5] = -51.0*np.pi/180.0
    # print(initial_joint_states[0][0])

    start_state = JointState.from_position(
        initial_joint_states,
        joint_names=[
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        ],
    )
    start = time.time()

    result = motion_gen.plan_single(start_state, goal_pose, MotionGenPlanConfig(max_attempts=1))
    traj = result.get_interpolated_plan()  # result.optimized_dt has the dt between timesteps
    positions = traj.position.to('cpu').cpu().numpy()
    velocities = traj.velocity.to('cpu').cpu().numpy()
    accelerations = traj.acceleration.to('cpu').cpu().numpy()
    dt = result.optimized_dt.to('cpu').cpu().numpy()
    dts = [float(dt)*i for i in range(len(positions))]

    trajectory_ur5 = FollowJointTrajectory.Goal()
    joint_trajectory = JointTrajectory()

    joint_names=[
        "arm_0_shoulder_pan_joint",
        "arm_0_shoulder_lift_joint",
        "arm_0_elbow_joint",
        "arm_0_wrist_1_joint",
        "arm_0_wrist_2_joint",
        "arm_0_wrist_3_joint",
    ]

    joint_trajectory.joint_names = joint_names
    joint_publisher = MinimalPublisher()
    for i in range(len(positions)):
        positions_i = positions[i]
        new_positions_i = []
        for num in positions_i:
            new_positions_i.append(float(num))

        velocities_i = velocities[i]
        new_velocities_i = []
        for num in positions_i:
            new_velocities_i.append(float(num))

        accelerations_i = accelerations[i]
        new_accelerations_i = []
        for num in positions_i:
            new_accelerations_i.append(float(num))

        traj_pt = JointTrajectoryPoint()
        traj_pt.positions = new_positions_i
        traj_pt.velocities = new_velocities_i
        traj_pt.accelerations = new_accelerations_i
        # dur = Duration()
        # dur.sec = dts[i]
        # traj_pt.time_from_start = dur
        
        joint_trajectory.points.append(traj_pt)
        msg = js()
        msg.name = joint_names
        # request.ik_request.pose_stamped.header.frame_id =self.base_
        # request.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()

        msg.header.frame_id = "base_link"
        msg.header.stamp = joint_publisher.get_clock().now().to_msg()
        msg.position = new_positions_i
        joint_publisher.publisher_.publish(msg)
        input("asdas")

if __name__ == "__main__":
    main()
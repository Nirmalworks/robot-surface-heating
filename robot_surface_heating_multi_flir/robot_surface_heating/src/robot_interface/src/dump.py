#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import numpy as np

from sensor_msgs.msg import JointState

from moveit_msgs.msg import (
    RobotState,
    RobotTrajectory,
    MoveItErrorCodes,
    Constraints,
    JointConstraint,
)
from moveit_msgs.srv import GetPositionIK, GetMotionPlan, GetPositionFK, GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory


from typing import Union

from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener

class RobotInterfaceNode(Node):
    timeout_sec_ = 5.0

    # move_group_name_ = "ur5"
    namespace_ = ""

    joint_state_topic_ = "/joint_states"
    plan_srv_name_ = "plan_kinematic_path"
    ik_srv_name_ = "compute_ik"
    fk_srv_name_ = "compute_fk"
    execute_action_name_ = "execute_trajectory"

    def __init__(self, move_group_name, joint_state_topic, trajectory_server, base_name, eef_name) -> None:
        super().__init__("robot_interface_node_"+move_group_name, namespace=self.namespace_)
        # argument parameter assignments
        self.move_group_name_ = move_group_name
        self.joint_state_topic_ = joint_state_topic
        self.trajectory_server_ = trajectory_server
        self.base_ = base_name
        self.end_effector_ = eef_name
        # self.gripper_srv_name = gripper_action

        self.cartesian_path_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')

        ##### assign parameters #####

        ##########  ##########  ##########

        ##### create service/action clients #####

        self.ik_client_ = self.create_client(GetPositionIK, self.ik_srv_name_)
        if not self.ik_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("IK service not available.")
            exit(1)

        self.fk_client_ = self.create_client(GetPositionFK, self.fk_srv_name_)
        if not self.fk_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("FK service not available.")
            exit(1)

        self.plan_client_ = self.create_client(GetMotionPlan, self.plan_srv_name_)
        if not self.plan_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("Plan service not available.")
            exit(1)

        self.execute_client_ = ActionClient(
            self, ExecuteTrajectory, self.execute_action_name_
        )
        if not self.execute_client_.wait_for_server(timeout_sec=self.timeout_sec_):
            self.get_logger().error("Execute action not available.")
            exit(1)

        self.srv = self.create_service(PoseToPlan, move_group_name+'_move_to_pose', self.execute_motion_plan)

        self.traj_client = ActionClient(self, FollowJointTrajectory, self.trajectory_server_)

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        ##########  ##########  ##########

    def get_ik(self, target_pose: Pose) -> JointState | None:
        request = GetPositionIK.Request()

        request.ik_request.group_name = self.move_group_name_
        tf_prefix = self.get_namespace()[1:]
        request.ik_request.pose_stamped.header.frame_id =self.base_
        request.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        request.ik_request.pose_stamped.pose = target_pose
        request.ik_request.avoid_collisions = True

        # self.get_logger().error(f"request: {request}")

        future = self.ik_client_.call_async(request)

        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error("Failed to get IK solution")
            return None

        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            # self.get_logger().error("IK solution was not successful")
            return None

        return response.solution.joint_state
    
    def get_fk(self) -> Pose | None:
        current_joint_state_set, current_joint_state = wait_for_message(
            JointState, self, self.joint_state_topic_, time_to_wait=1.0
        )
        if not current_joint_state_set:
            self.get_logger().error("Failed to get current joint state")
            return None

        current_robot_state = RobotState()
        current_robot_state.joint_state = current_joint_state

        request = GetPositionFK.Request()

        request.header.frame_id = self.base_
        request.header.stamp = self.get_clock().now().to_msg()

        request.fk_link_names.append(self.end_effector_)
        request.robot_state = current_robot_state

        future = self.fk_client_.call_async(request)

        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error("Failed to get FK solution")
            return None

        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"Failed to get FK solution: {response.error_code.val}"
            )
            return None
        
        return response.pose_stamped[0].pose

    def sum_of_square_diff(
        self, joint_state_1: JointState, joint_state_2: JointState
    ) -> float:
        return np.sum(
            np.square(np.subtract(joint_state_1.position, joint_state_2.position))
        )

    def check_same_pose(
            self, pose_1: Pose, pose_2: Pose
    ) -> bool:
        x_diff = np.abs(pose_1.position.x - pose_2.position.x)
        y_diff = np.abs(pose_1.position.y - pose_2.position.y)
        z_diff = np.abs(pose_1.position.z - pose_2.position.z)

        pos_diff = np.asarray([x_diff, y_diff, z_diff])

        qx_diff = np.abs(pose_1.orientation.x - pose_2.orientation.x)
        qy_diff = np.abs(pose_1.orientation.y - pose_2.orientation.y)
        qz_diff = np.abs(pose_1.orientation.z - pose_2.orientation.z)
        qw_diff = np.abs(pose_1.orientation.w - pose_2.orientation.w)

        quat_diff = np.asarray([qx_diff, qy_diff, qz_diff, qw_diff])

        return np.sum(np.square(pos_diff)) < 0.001 and np.sum(np.square(quat_diff)) < 0.001

    def get_best_ik(self, target_pose: Pose, attempts: int = 100) -> JointState | None:
        current_joint_state_set, current_joint_state = wait_for_message(
            JointState, self, self.joint_state_topic_, time_to_wait=1.0
        )
        if not current_joint_state_set:
            self.get_logger().error("Failed to get current joint state")
            return None

        best_cost = np.inf
        best_joint_state = None

        for _ in range(attempts):
            joint_state = self.get_ik(target_pose)
            if joint_state is None:
                continue

            # cost = self.sum_of_square_diff(current_joint_state, joint_state)
            # if cost < best_cost:
            #     best_cost = cost
            best_joint_state = joint_state

        if not best_joint_state:
            self.get_logger().error("Failed to get IK solution")

        return best_joint_state

    def get_motion_plan(
        self, target_pose: Pose, linear: bool = False, attempts: int = 10
    ) -> RobotTrajectory | None:

        self.get_logger().info("start of get motion plan")

        current_pose = self.get_fk()
        # self.get_logger().info(f"current pose: {current_pose}")
        if not current_pose:
            self.get_logger().error("Failed to get current pose")

        if self.check_same_pose(current_pose, target_pose):
            self.get_logger().info("already at pose")
            return RobotTrajectory()
        
        current_joint_state_set, current_joint_state = wait_for_message(
            JointState, self, self.joint_state_topic_, time_to_wait=1.0
        )
        if not current_joint_state_set:
            self.get_logger().error("Failed to get current joint state")
            return None

        self.get_logger().info(f"current joint state: {current_joint_state}")

        current_robot_state = RobotState()
        current_robot_state.joint_state.position = current_joint_state.position

        target_joint_state = self.get_best_ik(target_pose)
        if target_joint_state is None:
            self.get_logger().error("Failed to get target joint state")
            return None

        target_constraint = Constraints()
        for i in range(len(target_joint_state.position)):
            joint_constraint = JointConstraint()
            joint_constraint.joint_name = target_joint_state.name[i]
            joint_constraint.position = target_joint_state.position[i]
            joint_constraint.tolerance_above = 0.001
            joint_constraint.tolerance_below = 0.001
            joint_constraint.weight = 1.0
            target_constraint.joint_constraints.append(joint_constraint)

        request = GetMotionPlan.Request()
        request.motion_plan_request.group_name = self.move_group_name_
        request.motion_plan_request.start_state = current_robot_state
        request.motion_plan_request.goal_constraints.append(target_constraint)
        request.motion_plan_request.num_planning_attempts = 10
        request.motion_plan_request.allowed_planning_time = 5.0
        request.motion_plan_request.max_velocity_scaling_factor = 0.5
        request.motion_plan_request.max_acceleration_scaling_factor = 0.2

        if linear:
            request.motion_plan_request.pipeline_id = "pilz_industrial_motion_planner"
            request.motion_plan_request.planner_id = "LIN"
        else:
            request.motion_plan_request.pipeline_id = "ompl"
            request.motion_plan_request.planner_id = "RRTkConfigDefault"

        for _ in range(attempts):
            plan_future = self.plan_client_.call_async(request)
            rclpy.spin_until_future_complete(self, plan_future)

            if plan_future.result() is None:
                self.get_logger().error("Failed to get motion plan")

            response = plan_future.result()
            if response.motion_plan_response.error_code.val != MoveItErrorCodes.SUCCESS:
                self.get_logger().error(
                    f"Failed to get motion plan: {response.motion_plan_response.error_code.val}"
                )
            else:
                self.get_logger().info("get motion plan success")
                return response.motion_plan_response.trajectory
        return None

    def execute_motion_plan(self, traj, target_pose):        

        if(traj == None):
            self.get_logger().error("NO MOTION PLAN FOUND")
            response.success = False
        else:
            self.get_logger().info("A MOTION PLAN IS CREATED")
            traj_goal = FollowJointTrajectory.Goal()
            traj_goal.trajectory = traj.joint_trajectory
            self.traj_client.wait_for_server()
            result = self.traj_client.send_goal_async(traj_goal)
            reached = False

            while(reached == False):
                curr_pose = self.get_fk()
                x_pos_diff = abs(curr_pose.position.x - target_pose.position.x)
                y_pos_diff = abs(curr_pose.position.x - target_pose.position.x)
                z_pos_diff = abs(curr_pose.position.x - target_pose.position.x)
                x_ori_diff = abs(curr_pose.orientation.x - target_pose.orientation.x)
                y_ori_diff = abs(curr_pose.orientation.y - target_pose.orientation.y)
                z_ori_diff = abs(curr_pose.orientation.z - target_pose.orientation.z)
                w_ori_diff = abs(curr_pose.orientation.w - target_pose.orientation.w)

                if(x_pos_diff <=0.01 and y_pos_diff <=0.01 and z_pos_diff <=0.01): # and x_ori_diff <=0.01 and y_ori_diff <=0.01 and z_ori_diff <=0.01 and w_ori_diff <=0.01):
                    reached = True

            self.get_logger().info("MOTION PLAN EXECUTION COMPLETED")

    def get_motion_execute_client(self) -> ActionClient:
        return self.execute_client_

    def send_action_sequence_poses(self, items: list[torch.Tensor], from_idx: int, lan: int) -> None:
        """
        Creates a trajectory of poses from the list of actions, then sends it to Pilz/cartesian
        planner for splining execution.

        Testing on both arms (UR5 and UR10e).
        """
        return self.send_action_sequence_new(items, from_idx, lan)

    def get_cartesian_trajectory(self, items: list[torch.Tensor], from_idx: int, lan: int) -> RobotTrajectory:

        # visualize poses of waypoints
        from tf2_ros import TransformBroadcaster
        from geometry_msgs.msg import TransformStamped, PoseArray, Pose
        self.tf_broadcaster = TransformBroadcaster(self)
        t = TransformStamped()
        for idx in range(len(items)):
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = 'world'
            t.child_frame_id = f'waypoint_{idx}'

            # items_idx = items[idx]
            # print(f"items idx: {items_idx}")

            t.transform.translation.x = float(items[idx]["action"][0])
            t.transform.translation.y = float(items[idx]["action"][1])
            t.transform.translation.z = float(items[idx]["action"][2])
            t.transform.rotation.x = float(items[idx]["action"][3])
            t.transform.rotation.y = float(items[idx]["action"][4])
            t.transform.rotation.z = float(items[idx]["action"][5])
            t.transform.rotation.w = float(items[idx]["action"][6])

            self.tf_broadcaster.sendTransform(t)

        waypoint_list = []

        for item_idx in range(from_idx, from_idx+lan):

            # create waypoint constraints
            arm_off = 0
            for arm in self.follower_arms:

                # extract position values from pose data
                xyz = [items[item_idx]["action"][arm_off], items[item_idx]["action"][arm_off+1], items[item_idx]["action"][arm_off+2]]
                quat = items[item_idx]["action"][arm_off+3:arm_off+7]


                waypoint = Pose()
                waypoint.position.x = float(xyz[0])
                waypoint.position.y = float(xyz[1])
                waypoint.position.z = float(xyz[2])
                waypoint.orientation.x = float(quat[0])
                waypoint.orientation.y = float(quat[1])
                waypoint.orientation.z = float(quat[2])
                waypoint.orientation.w = float(quat[3])
                waypoint_list.append(waypoint)

                # move to pose and gripper joint data for next arm
                arm_off += 8

        from std_msgs.msg import Header
        request = GetCartesianPath.Request(
            header=Header(frame_id=self.base_, stamp=self.get_clock().now().to_msg()),
            max_step=0.01,
            jump_threshold = 0.0,
            group_name="ur10e",
            waypoints=waypoint_list
        )

        plan_future = self.cartesian_path_client.call_async(request)
        rclpy.spin_until_future_complete(self, plan_future)
        response = plan_future.result()

        return response.solution.joint_trajectory

    def send_action_sequence_new(self, items: list[torch.Tensor], from_idx: int, lan: int) -> None:
        print("here")
        cartesian_path = self.get_cartesian_trajectory(items, from_idx, lan)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = cartesian_path
        self.traj_client.wait_for_server()
        goal_future = self.traj_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, goal_future)
        rclpy.spin_until_future_complete(self, goal_future)

def old_demo(args=None):
    rclpy.init(args=args)

    robot_interface_node_ur10e = RobotInterfaceNode("ur10e", "joint_states", "scaled_joint_trajectory_controller/follow_joint_trajectory", "base_link", "tool0")
    pose_ur10e = Pose()
    pose_ur10e.position.x = 0.297
    pose_ur10e.position.y = -0.493
    pose_ur10e.position.z = 0.846
    pose_ur10e.orientation.x = 0.797
    pose_ur10e.orientation.y = 0.054
    pose_ur10e.orientation.z = -0.225
    pose_ur10e.orientation.w = 0.558

    traj_ur10e = robot_interface_node_ur10e.get_motion_plan(pose_ur10e)
    robot_interface_node_ur10e.execute_motion_plan(traj_ur10e, pose_ur10e)
    rclpy.shutdown()

if __name__ == "__main__":
    old_demo()

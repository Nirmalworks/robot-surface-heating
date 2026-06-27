#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import numpy as np

from geometry_msgs.msg import Pose, Point, Quaternion, PoseArray, PoseStamped
from sensor_msgs.msg import JointState

from moveit_msgs.msg import (
    RobotState,
    RobotTrajectory,
    MoveItErrorCodes,
    Constraints,
    JointConstraint,
    PositionConstraint,
)
from moveit_msgs.srv import GetPositionIK, GetMotionPlan, GetPositionFK, GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory


from typing import Union

from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec

from control_msgs.action import FollowJointTrajectory, GripperCommand
from control_msgs.msg import GripperCommand as Gc_Msg
from robot_interface.srv import PoseToPlan
# from robot_interface.msg import GripperResult
from rclpy import Parameter
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from shape_msgs.msg import SolidPrimitive

import threading
from rclpy.executors import MultiThreadedExecutor

from ros2_aruco_interfaces.msg import ArucoMarkers
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs
import asyncio

import ctypes
import os
import random
from scipy.spatial.transform import Rotation as R

#### python bindings with ctypes ####

# dir_path = os.path.dirname(os.path.realpath(__file__))
# dll_file = os.path.join(dir_path,'dual_motion.dll')

# lib = ctypes.CDLL(dll_file)
# /home/cam/ur_bimanual/install/robot_interface/lib/robot_interface/libdual_motion.so
lib = ctypes.cdll.LoadLibrary('/home/cam/ur_bimanual/install/robot_interface/lib/robot_interface/libdual_motion.so')

PosePos = ctypes.c_float * 3
PoseOrt = ctypes.c_float * 4

lib.dual_ur_motion_old.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), \
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]   
lib.dual_ur_motion_old.restype = ctypes.c_void_p

"""dual_ur_motion -> C++ function that plans dual motion for a combined movegroup using the provided solution joint states (found in dual_motion.cpp)
Input:
- The solution joint states for both arms, separated into arrays for joint names ([str]), positions ([float]), velocities ([float]), and efforts ([float])
- The name of the movegroup used for motion planning (str)
- The name/planner ID of the motion planner used for planning (str)
Output:
- The status code of the motion planning (int)
"""
lib.dual_ur_motion.argtypes = [ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_float), \
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_char_p), ctypes.POINTER(ctypes.c_float), \
    ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float), ctypes.c_char_p, ctypes.c_char_p]
# lib.dual_ur_motion.restype = ctypes.c_void_p
lib.dual_ur_motion.restype = ctypes.c_int

######## ########

def wait_for_message(
    msg_type,
    node: 'Node',
    topic: str,
    *,
    qos_profile: Union[QoSProfile, int] = 1,
    time_to_wait=-1
):
    """
    Wait for the next incoming message.

    :param msg_type: message type
    :param node: node to initialize the subscription on
    :param topic: topic name to wait for message
    :param qos_profile: QoS profile to use for the subscription
    :param time_to_wait: seconds to wait before returning
    :returns: (True, msg) if a message was successfully received, (False, None) if message
        could not be obtained or shutdown was triggered asynchronously on the context.
    """
    context = node.context
    wait_set = _rclpy.WaitSet(1, 1, 0, 0, 0, 0, context.handle)
    wait_set.clear_entities()

    sub = node.create_subscription(msg_type, topic, lambda _: None, qos_profile=qos_profile)
    try:
        wait_set.add_subscription(sub.handle)
        sigint_gc = SignalHandlerGuardCondition(context=context)
        wait_set.add_guard_condition(sigint_gc.handle)

        timeout_nsec = timeout_sec_to_nsec(time_to_wait)
        wait_set.wait(timeout_nsec)

        subs_ready = wait_set.get_ready_entities('subscription')
        guards_ready = wait_set.get_ready_entities('guard_condition')

        if guards_ready:
            if sigint_gc.handle.pointer in guards_ready:
                return False, None

        if subs_ready:
            if sub.handle.pointer in subs_ready:
                msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
                if msg_info is not None:
                    return True, msg_info[0]
    finally:
        node.destroy_subscription(sub)

    return False, None

class RobotInterfaceNode(Node):
    timeout_sec_ = 5.0

    # move_group_name_ = "ur5"
    namespace_ = ""

    joint_state_topic_ = "/joint_states"
    plan_srv_name_ = "plan_kinematic_path"
    ik_srv_name_ = "compute_ik"
    fk_srv_name_ = "compute_fk"
    execute_action_name_ = "execute_trajectory"
    # trajectory_server_ = '/arm_0/_controller/follow_joint_trajectory'

    # base_ = "arm_0_base_link"
    # end_effector_ = "arm_0_tool0"

    # gripper_srv_name = "/robotiq/robotiq_gripper_controller/gripper_cmd"

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


        # Declare and read parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                (f"{self.move_group_name_}_params.object_approach_ort", [0.0, 0.0, 0.0, 1.0]),
                (f"{self.move_group_name_}_params.gripper_service", "/gripper_actuate"),
                (f"{self.move_group_name_}_params.gripper_grab_pos", 0.0),
                (f"{self.move_group_name_}_params.gripper_open_pos", 0.0),
                (f"{self.move_group_name_}_params.gripper_default_effort", 0.0),
                (f"{self.move_group_name_}_params.gripper_offset", [0.0, 0.0, 0.0]),
                (f"{self.move_group_name_}_params.prelift_height", 0.0),
                (f"{self.move_group_name_}_params.lift_height", 0.0),
                (f"{self.move_group_name_}_params.dropoff_pos", [0.0, 0.0, 0.0]),
                (f"{self.move_group_name_}_params.home_tool_pos", [0.0, 0.0, 0.0]),
                (f"{self.move_group_name_}_params.home_joint_state_name", ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_3_joint", "wrist_2_joint"]),
                (f"{self.move_group_name_}_params.home_joint_state_pos", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                (f"{self.move_group_name_}_params.home_joint_state_vel", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                (f"{self.move_group_name_}_params.home_joint_state_effort", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            ]
        )

        ##### assign parameters #####

        # orientation for picking up the item
        self.OBJECT_APPROACH_ORIENTATION = self.get_parameter(f"{self.move_group_name_}_params.object_approach_ort").value

        # gripper defaults
        self.GRIPPER_GRAB_POSITION = self.get_parameter(f"{self.move_group_name_}_params.gripper_grab_pos").value
        self.GRIPPER_OPEN_POSITION = self.get_parameter(f"{self.move_group_name_}_params.gripper_open_pos").value
        self.GRIPPER_DEFAULT_EFFORT = self.get_parameter(f"{self.move_group_name_}_params.gripper_default_effort").value
        
        # gripper offsets to account for motion aiming towards origin of target frame
        self.GRIPPER_OFFSET = self.get_parameter(f"{self.move_group_name_}_params.gripper_offset").value

        # height to which eef is lowered to prepare for gripper close
        self.PRELIFT_HEIGHT = self.get_parameter(f"{self.move_group_name_}_params.prelift_height").value
        # height to which item is lifted before movement to dropoff location
        self.LIFT_HEIGHT = self.get_parameter(f"{self.move_group_name_}_params.lift_height").value

        # dropoff position for item
        self.DROPOFF_POSITION = self.get_parameter(f"{self.move_group_name_}_params.dropoff_pos").value
        self.DROPOFF_EXTERN = Pose()
        self.DROPOFF_EXTERN.position.x = self.DROPOFF_POSITION[0]
        self.DROPOFF_EXTERN.position.y = self.DROPOFF_POSITION[1]
        self.DROPOFF_EXTERN.position.z = self.LIFT_HEIGHT
        self.DROPOFF_EXTERN.orientation.x = self.OBJECT_APPROACH_ORIENTATION[0]
        self.DROPOFF_EXTERN.orientation.y = self.OBJECT_APPROACH_ORIENTATION[1]
        self.DROPOFF_EXTERN.orientation.z = self.OBJECT_APPROACH_ORIENTATION[2]
        self.DROPOFF_EXTERN.orientation.w = self.OBJECT_APPROACH_ORIENTATION[3]

        # home position of end effector and home joint states of arm
        self.HOME_POSITION = self.get_parameter(f"{self.move_group_name_}_params.home_tool_pos").value
        self.HOME_POSE = Pose()
        # self.HOME_POSE.position = Point()
        self.HOME_POSE.position.x, self.HOME_POSE.position.y, self.HOME_POSE.position.z = self.HOME_POSITION[0], self.HOME_POSITION[1], self.HOME_POSITION[2]
        # self.HOME_POSE.orientation = Quaternion()
        self.HOME_POSE.orientation.x, self.HOME_POSE.orientation.y, self.HOME_POSE.orientation.z, self.HOME_POSE.orientation.w = \
            self.OBJECT_APPROACH_ORIENTATION[0], self.OBJECT_APPROACH_ORIENTATION[1], self.OBJECT_APPROACH_ORIENTATION[2], \
            self.OBJECT_APPROACH_ORIENTATION[3]


        self.HOME_JOINT_STATE = JointState()
        self.HOME_JOINT_STATE.name = self.get_parameter(f"{self.move_group_name_}_params.home_joint_state_name").value
        self.HOME_JOINT_STATE.position = self.get_parameter(f"{self.move_group_name_}_params.home_joint_state_pos").value
        self.HOME_JOINT_STATE.velocity = self.get_parameter(f"{self.move_group_name_}_params.home_joint_state_vel").value
        self.HOME_JOINT_STATE.effort = self.get_parameter(f"{self.move_group_name_}_params.home_joint_state_effort").value

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
            
#                 self.get_logger().info(f"pos diff not enough: {x_pos_diff if x_pos_diff > 0.01 else 'ok'} \
# {y_pos_diff if y_pos_diff > 0.01 else 'ok'} {z_pos_diff if z_pos_diff > 0.01 else 'ok'} {x_ori_diff if x_ori_diff > 0.01 else 'ok'} \
# {y_ori_diff if y_ori_diff > 0.01 else 'ok'} {z_ori_diff if z_ori_diff > 0.01 else 'ok'} {w_ori_diff if w_ori_diff > 0.01 else 'ok'}")

            self.get_logger().info("MOTION PLAN EXECUTION COMPLETED")

    def get_motion_execute_client(self) -> ActionClient:
        return self.execute_client_

    def return_to_home(self, linear: bool=False, attempts: int=10) -> None:
        self.get_logger().info("RETURNING TO HOME")

        home_pose = self.HOME_POSE

        current_pose = self.get_fk()
        # self.get_logger().info(f"current pose: {current_pose}")
        if not current_pose:
            self.get_logger().error("Failed to get current pose")

        if self.check_same_pose(current_pose, home_pose):
            self.get_logger().info("already at pose")
            return RobotTrajectory()

        current_joint_state_set, current_joint_state = wait_for_message(
            JointState, self, self.joint_state_topic_, time_to_wait=1.0
        )
        if not current_joint_state_set:
            self.get_logger().error("Failed to get current joint state")
            return None

        # self.get_logger().info(f"current joint state: {current_joint_state}")

        current_robot_state = RobotState()
        current_robot_state.joint_state.position = current_joint_state.position

        target_joint_state = self.HOME_JOINT_STATE
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
                self.execute_motion_plan(response.motion_plan_response.trajectory, home_pose)
                return
        self.get_logger().error(f"Failed to get homing motion plan after {attempts} attempts.")

    def transform_to_pose(self, target_pose, target_transform, target_id) -> Pose:
        # get target pose and transform to get to target
        marker_pose, t = target_pose, target_transform

        # apply transformation matrix from camera frame to ur5 frame
        self.get_logger().info(f"marker detected: id {target_id} pos.x {marker_pose.position.x} pos.y {marker_pose.position.y} \
pos.z {marker_pose.position.z} ort.x {marker_pose.orientation.x} ort.y {marker_pose.orientation.y} \
ort.z {marker_pose.orientation.z} ort.w {marker_pose.orientation.w}", once=True)
        
        self.get_logger().info(f"transform found: {t.transform.translation.x} {t.transform.translation.y} {t.transform.translation.z} \
{t.transform.rotation.x} {t.transform.rotation.y} {t.transform.rotation.z} {t.transform.rotation.w}")

        # apply rotation quaternion to flip target pose for better tool approach orientation
        rot_quat_x = Quaternion()
        rot_quat_x.x = 1.0
        rot_quat_x.y = 0.0
        rot_quat_x.z = 0.0
        rot_quat_x.w = 0.0

        def quaternion_multiply(q1, q2):
            new_quat = Quaternion()
            new_quat.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y
            new_quat.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x
            new_quat.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
            new_quat.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
            return new_quat

        marker_pose.orientation = quaternion_multiply(rot_quat_x, marker_pose.orientation)
        marker_pose.position.x -= self.GRIPPER_OFFSET[0]
        # marker_pose.position.x = marker_pose.position.x + self.GRIPPER_OFFSET[0] if marker_pose.position.x > 0 else marker_pose.position.x - self.GRIPPER_OFFSET[0]
        marker_pose.position.y -= self.GRIPPER_OFFSET[1]
        # marker_pose.position.y = marker_pose.position.y + self.GRIPPER_OFFSET[0] if marker_pose.position.y > 0 else marker_pose.position.y - self.GRIPPER_OFFSET[0]

        # apply transformation to aruco pose to obtain target pose for arm
        pose_transformed = tf2_geometry_msgs.do_transform_pose(marker_pose, t)

        # adjust target pose position and orientation to default approach gripper offset
        # pose_transformed.position.x += self.GRIPPER_OFFSET[0]
        # pose_transformed.position.y += self.GRIPPER_OFFSET[1]
        pose_transformed.position.z = self.GRIPPER_OFFSET[2]

        self.get_logger().info(f"transformed pose: id {target_id} pos.x {pose_transformed.position.x} pos.y {pose_transformed.position.y} \
pos.z {pose_transformed.position.z} ort.x {pose_transformed.orientation.x} ort.y {pose_transformed.orientation.y} \
ort.z {pose_transformed.orientation.z} ort.w {pose_transformed.orientation.w}", once=True)

        return pose_transformed

    def send_action_sequence_poses(self, items: list[torch.Tensor], from_idx: int, lan: int) -> None:
        """
        Creates a trajectory of poses from the list of actions, then sends it to Pilz/cartesian
        planner for splining execution.

        Testing on both arms (UR5 and UR10e).
        """
        return self.send_action_sequence_new(items, from_idx, lan)

    def interpolate_cartesian_trajectory(self, start_pose: Pose, end_pose: Pose, steps: int = 10) -> None:
        """
        Interpolates linearly between start_pose and end_pose over 'steps' steps,
        computes the Cartesian path, and sends it to the trajectory client.
        """

        def lerp_vec3(a, b, t):
            return a + (b - a) * t

        # Convert quaternions to scipy Rotation objects for SLERP
        r_start = R.from_quat([
            start_pose.orientation.x,
            start_pose.orientation.y,
            start_pose.orientation.z,
            start_pose.orientation.w,
        ])
        r_end = R.from_quat([
            end_pose.orientation.x,
            end_pose.orientation.y,
            end_pose.orientation.z,
            end_pose.orientation.w,
        ])

        # Prepare interpolation
        waypoints = []
        for i in range(steps + 1):
            t = i / steps
            # Interpolate position
            pos = np.array([
                lerp_vec3(start_pose.position.x, end_pose.position.x, t),
                lerp_vec3(start_pose.position.y, end_pose.position.y, t),
                lerp_vec3(start_pose.position.z, end_pose.position.z, t),
            ])

            # Interpolate orientation via SLERP
            r_interp = R.slerp(t, [r_start, r_end])
            quat = r_interp.as_quat()  # [x, y, z, w]

            # Create Pose waypoint
            waypoint = Pose()
            waypoint.position.x = pos[0]
            waypoint.position.y = pos[1]
            waypoint.position.z = pos[2]
            waypoint.orientation.x = quat[0]
            waypoint.orientation.y = quat[1]
            waypoint.orientation.z = quat[2]
            waypoint.orientation.w = quat[3]

            waypoints.append(waypoint)

        # Create request for GetCartesianPath service
        from std_msgs.msg import Header
        request = GetCartesianPath.Request(
            header=Header(frame_id=self.base_, stamp=self.get_clock().now().to_msg()),
            group_name=self.move_group_name_,
            waypoints=waypoints,
            max_step=0.01,        # max distance between computed points
            jump_threshold=0.0    # disable jump threshold (tune if needed)
        )

        self.get_logger().info(f"Calling GetCartesianPath with {len(waypoints)} waypoints")

        # Call the service
        future = self.cartesian_path_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        response = future.result()

        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"GetCartesianPath failed with error code {response.error_code.val}")
            return

        traj = response.solution.joint_trajectory

        # Send trajectory to the trajectory client
        from control_msgs.action import FollowJointTrajectory
        traj_goal = FollowJointTrajectory.Goal()
        traj_goal.trajectory = traj

        self.traj_client.wait_for_server()
        send_future = self.traj_client.send_goal_async(traj_goal)
        rclpy.spin_until_future_complete(self, send_future)
        self.get_logger().info("Cartesian path execution sent to the controller.")

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

    # robot_interface_node_ur5 = RobotInterfaceNode("ur5", "/arm_0/joint_states", "/arm_0/scaled_joint_trajectory_controller/follow_joint_trajectory", "arm_0_base_link", "arm_0_tool0")
    robot_interface_node_ur10e = RobotInterfaceNode("ur10e", "joint_states", "scaled_joint_trajectory_controller/follow_joint_trajectory", "base_link", "tool0")
    pose_ur5 = Pose()
    # pose_ur5.position.x = -0.49
    # pose_ur5.position.y = -0.41
    # pose_ur5.position.z  = 0.47
    # pose_ur5.orientation.x = -0.42
    # pose_ur5.orientation.y = 0.76
    # pose_ur5.orientation.z = -0.42
    # pose_ur5.orientation.w = -0.25

    # pose_ur5.position.x = -0.516
    # pose_ur5.position.y = -0.395
    # pose_ur5.position.z  = 0.485
    # pose_ur5.orientation.x = -0.437
    # pose_ur5.orientation.y = 0.751
    # pose_ur5.orientation.z = -0.434
    # pose_ur5.orientation.w = -0.235

    # cur_pose = robot_interface_node_ur5.get_fk()
    # robot_interface_node_ur5.get_logger().info(f"\n\ncurrent pose: {robot_interface_node_ur5.get_fk()}\n\n")

    # pose_ur5.position.x = -0.6213153728572185
    # pose_ur5.position.y = 0.08013202555988705
    # pose_ur5.position.z  = 0.016466777605816052 + 0.05
    # pose_ur5.orientation.x = cur_pose.orientation.x
    # pose_ur5.orientation.y = cur_pose.orientation.y
    # pose_ur5.orientation.z = cur_pose.orientation.z
    # pose_ur5.orientation.w = cur_pose.orientation.w

    # traj_ur5 =robot_interface_node_ur5.get_motion_plan(pose_ur5)

    pose_ur10e = Pose()
    pose_ur10e.position.x = 0.297
    pose_ur10e.position.y = -0.493
    pose_ur10e.position.z = 0.846
    pose_ur10e.orientation.x = 0.797
    pose_ur10e.orientation.y = 0.054
    pose_ur10e.orientation.z = -0.225
    pose_ur10e.orientation.w = 0.558

    # pose_ur10e.position.x = 0.277
    # pose_ur10e.position.y = -0.473
    # pose_ur10e.position.z = 0.855
    # pose_ur10e.orientation.x = 0.776
    # pose_ur10e.orientation.y = 0.074
    # pose_ur10e.orientation.z = -0.235
    # pose_ur10e.orientation.w = 0.568

    traj_ur10e = robot_interface_node_ur10e.get_motion_plan(pose_ur10e)

    # robot_interface_node_ur5.execute_motion_plan(traj_ur5, pose_ur5)
    robot_interface_node_ur10e.execute_motion_plan(traj_ur10e, pose_ur10e)
    rclpy.shutdown()

if __name__ == "__main__":
    old_demo()

    # rclpy.init()
    # rin_ur10e = RobotInterfaceNode("ur10e", "/arm_1/joint_states", "scaled_joint_trajectory_controller/follow_joint_trajectory", "base_link", "tool0")   

    # # Use MultiThreadedExecutor to handle multiple callbacks concurrently
    # executor = MultiThreadedExecutor()
    # executor.add_node(rin_ur10e)

    # try:
    #     # Run the executor to process callbacks
    #     # while rclpy.ok():
    #         # executor.spin_once(timeout_sec=0.1)  # Adjust the timeout as needed
    #     executor.spin()
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     # rclpy.spin(dual_rin)
    #     rin_ur10e.sync_thread.join()
    #     rin_ur10e.destroy_node()
    #     rclpy.shutdown()
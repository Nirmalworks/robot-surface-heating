#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import numpy as np

from geometry_msgs.msg import Pose, Quaternion
from sensor_msgs.msg import JointState

from moveit_msgs.msg import (
    MoveItErrorCodes,
    RobotState,
)
from moveit_msgs.srv import GetPositionIK, GetPositionFK

#### wait_for_message() dependencies ####
from typing import Union
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec
######## ########

from control_msgs.action import GripperCommand
from control_msgs.msg import GripperCommand as Gc_Msg
from rclpy import Parameter

import threading
from rclpy.executors import MultiThreadedExecutor

#### vision package imports ####
from ros2_aruco_interfaces.msg import ArucoMarkers
from centroid_identifier_interfaces.msg import CentrArray
######## ########

#### PyBullet/BCD Primitives imports ####
from pybullet_plan_interfaces.msg import PyBulletPlan, PlanStep
from motion_primitive_interfaces.msg import PrimitivePlan, PrimitivePlanStep
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
######## ########

from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
import tf2_geometry_msgs
import asyncio
import math

import ctypes
import os

#### python bindings with ctypes ####
lib = ctypes.cdll.LoadLibrary('/home/cam/ur_bimanual/install/robot_interface/lib/robot_interface/libdual_motion.so')

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

#### Quaternion Euler Utility Functions ####
def quaternion_multiply(q1, q2):
    new_quat = Quaternion()
    new_quat.x = q1.w * q2.x + q1.x * q2.w + q1.y * q2.z - q1.z * q2.y
    new_quat.y = q1.w * q2.y - q1.x * q2.z + q1.y * q2.w + q1.z * q2.x
    new_quat.z = q1.w * q2.z + q1.x * q2.y - q1.y * q2.x + q1.z * q2.w
    new_quat.w = q1.w * q2.w - q1.x * q2.x - q1.y * q2.y - q1.z * q2.z
    return new_quat

def quaternion_from_euler(roll, pitch, yaw):
    """
    Converts euler roll, pitch, yaw to quaternion
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = Quaternion()
    q.w = cy * cp * cr + sy * sp * sr
    q.x = cy * cp * sr - sy * sp * cr
    q.y = sy * cp * sr + cy * sp * cr
    q.z = sy * cp * cr - cy * sp * sr
    return q 
######## ########


######### start of RobotInterface #########
class RobotInterfaceNode(Node):
    """Class instance for the robot interface of an individual robot
    Members:
    - move_group_name: move group associated with the robot, also serves as the namespace for parameter extraction
    - ik_srv: name of the IK solver service used by the robot
    - base_name: name of the frame used by the robot's base
    - eef_name: name of the frame used by the robot's end effector
    """
    timeout_sec_ = 5.0
    namespace_ = ""

    def __init__(self, move_group_name, ik_srv, fk_srv, base_name, eef_name) -> None:
        super().__init__("robot_interface_node_"+move_group_name, namespace=self.namespace_)
        # argument parameter assignments
        self.move_group_name_ = move_group_name
        self.base_ = base_name
        self.end_effector_ = eef_name
        self.ik_srv_name_ = ik_srv
        self.fk_srv_name_ = fk_srv

        # Declare and read parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                (f"{self.move_group_name_}_params.object_approach_ort", [0.0, 0.0, 0.0, 1.0]),
                (f"{self.move_group_name_}_params.gripper_service", "/gripper_actuate"),
                (f"{self.move_group_name_}_params.joint_state_topic", "scaled_joint_trajectory_controller/follow_joint_trajectory"),
                (f"{self.move_group_name_}_params.gripper_grab_pos", 0.0),
                (f"{self.move_group_name_}_params.gripper_open_pos", 0.0),
                (f"{self.move_group_name_}_params.gripper_default_effort", 0.0),
                (f"{self.move_group_name_}_params.gripper_offset", [0.0, 0.0, 0.0]),
                (f"{self.move_group_name_}_params.prelift_height", 0.0),
                (f"{self.move_group_name_}_params.lift_height", 0.0),
                (f"{self.move_group_name_}_params.dropoff_pos", [0.0, 0.0, 0.0]),
                (f"{self.move_group_name_}_params.home_tool_pos", [0.0, 0.0, 0.0]),
                (f"{self.move_group_name_}_params.joint_state_name", ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_3_joint", "wrist_2_joint"]),
                (f"{self.move_group_name_}_params.home_joint_state_pos", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                (f"{self.move_group_name_}_params.home_joint_state_vel", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
                (f"{self.move_group_name_}_params.home_joint_state_effort", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
            ]
        )

        ##### assign parameters #####

        # joint state topic
        self.joint_state_topic_ = self.get_parameter(f"{self.move_group_name_}_params.joint_state_topic").value

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
        self.HOME_POSE.position.x, self.HOME_POSE.position.y, self.HOME_POSE.position.z = self.HOME_POSITION[0], self.HOME_POSITION[1], self.HOME_POSITION[2]
        self.HOME_POSE.orientation.x, self.HOME_POSE.orientation.y, self.HOME_POSE.orientation.z, self.HOME_POSE.orientation.w = \
            self.OBJECT_APPROACH_ORIENTATION[0], self.OBJECT_APPROACH_ORIENTATION[1], self.OBJECT_APPROACH_ORIENTATION[2], \
            self.OBJECT_APPROACH_ORIENTATION[3]


        self.HOME_JOINT_STATE = JointState()
        self.HOME_JOINT_STATE.name = self.get_parameter(f"{self.move_group_name_}_params.joint_state_name").value
        self.HOME_JOINT_STATE.position = self.get_parameter(f"{self.move_group_name_}_params.home_joint_state_pos").value
        self.HOME_JOINT_STATE.velocity = self.get_parameter(f"{self.move_group_name_}_params.home_joint_state_vel").value
        self.HOME_JOINT_STATE.effort = self.get_parameter(f"{self.move_group_name_}_params.home_joint_state_effort").value

        ##########  ##########  ##########

        ##### create service/action clients #####

        # IK solver client
        self.ik_client_ = self.create_client(GetPositionIK, self.ik_srv_name_)
        if not self.ik_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("IK service not available.")
            exit(1)

        # FK solver client
        self.fk_client_ = self.create_client(GetPositionFK, self.fk_srv_name_)
        if not self.fk_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("FK service not available.")
            exit(1)

        # gripper action client and storage members
        self.gripper_srv_name = self.get_parameter(f"{self.move_group_name_}_params.gripper_service").value
        self.gripper_action_client_ = ActionClient(self, GripperCommand, self.gripper_srv_name)
        self.gripper_result_position = None
        self.gripper_result_effort = None
        self.gripper_result_stalled = None
        self.gripper_result_reached_goal = None

        # TF listener
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        ##########  ##########  ##########

    def actuate_gripper(self, position=0.0, max_effort=0.0):
        """Makes an action request to the associated gripper controller server to change the gripper's position"""
        gripper_goal = GripperCommand.Goal()
        gripper_goal.command = Gc_Msg()
        gripper_goal.command.position = position
        gripper_goal.command.max_effort = max_effort

        self.get_logger().info(f"ACTUATING GRIPPER: position={gripper_goal.command.position}, max_effort={gripper_goal.command.max_effort}")
        self.gripper_action_client_.wait_for_server()
        self.goal_future = self.gripper_action_client_.send_goal_async(gripper_goal)
        self.goal_future.add_done_callback(self.gripper_response_callback)

    def gripper_response_callback(self, future):
        """Action response handler for actuate_gripper"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('GRIPPER GOAL REJECTED')
            return

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.gripper_result_callback)
    
    def gripper_result_callback(self, future):
        """Action result handler for actuate_gripper"""
        result = future.result().result
        
        if result.stalled:
            self.get_logger().info(f"GRIPPER STOPPED DURING MOVEMENT: position={result.position}, effort={result.effort}")
        elif result.reached_goal:
            self.get_logger().info(f"GRIPPER COMPLETED MOVEMENT: position={result.position}, effort={result.effort}")
        else:
            self.get_logger().error(f"GRIPPER IN UNKNOWN STATE: position={result.position}, effort={result.effort}")
        
        # publish gripper result data -> maybe TODO
        self.gripper_result_position = result.position
        self.gripper_result_effort = result.effort
        self.gripper_result_stalled = result.stalled
        self.gripper_result_reached_goal = result.reached_goal

    def gripper_wrapper(self, position=0.0, effort=0.0):
        """debugging function for gripper action client/server interfacing"""
        self.actuate_gripper(position, effort)
        self.get_logger().info(f"data: position {self.gripper_result_position} \
effort {self.gripper_result_effort} stalled {self.gripper_result_stalled} \
reached_goal {self.gripper_result_reached_goal}")
        
    def get_fk(self) -> Pose | None:
        """Makes an action request to the forward kinematics client to obtain the current
        pose of the robot"""

        current_joint_state_set, current_joint_state = wait_for_message(
            JointState, self, self.joint_state_topic_, time_to_wait=1.0
        )
        if not current_joint_state_set:
            self.get_logger().error("FK: Failed to get current joint state")
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

    def vision_identify(self, target_id=0, msg_topic='/aruco_markers', msg_type=ArucoMarkers, tool_frame=False):
        # listener to get base to camera transform from static transform publisher

        # look for transformation between camera and base frames
        target_frame = self.end_effector_ if tool_frame else self.base_
        child_frame = 'camera_color_optical_frame'
        while True:
            try:
                tf_future = self.tf_buffer.wait_for_transform_async(
                    target_frame,
                    child_frame,
                    rclpy.time.Time())
                rclpy.spin_until_future_complete(self, tf_future)
                self.get_logger().info("found transform future")

                t = asyncio.run(self.tf_buffer.lookup_transform_async(
                    target_frame,
                    child_frame,
                    rclpy.time.Time()
                ))
            except TransformException as ex:
                self.get_logger().error(f"Could not transform target frame \"{target_frame}\" to child frame \"{child_frame}\": {ex}", once=True)
                continue

            # find the marker
            while True:
                marker_found, markers = wait_for_message(
                    ArucoMarkers, self, '/aruco_markers', time_to_wait=1.0
                )
                if not marker_found:
                    self.get_logger().info("Failed to find any marker")
                    continue
            
                for idx in range(len(markers.marker_ids)):
                    if markers.marker_ids[idx] == target_id:
                        return markers.poses[idx], t, f"aruco_target_{markers.marker_ids[idx]}"
                self.get_logger().info(f"Failed to find marker with ID {target_id}")
                continue

    def get_transform_frame(self, child_frame):
        """returns the transform from the frame of the robot's base to the child frame"""
        while True:
            try:
                tf_future = self.tf_buffer.wait_for_transform_async(
                    self.base_,
                    child_frame,
                    rclpy.time.Time())
                rclpy.spin_until_future_complete(self, tf_future)
                self.get_logger().info("found transform future")

                t = asyncio.run(self.tf_buffer.lookup_transform_async(
                    self.base_,
                    child_frame,
                    rclpy.time.Time()
                ))
                return t
            except TransformException as ex:
                self.get_logger().error(f"Could not transform target frame \"{self.base_}\" to child frame \"{child_frame}\": {ex}", once=True)
                continue

    def transform_to_pose(self, target_pose, target_transform, target_id, desired_offset=[0.0, 0.0, 0.0],
        desired_rotation=[-np.pi, 0.0, 0.0], override_posz=True) -> Pose:
        """transforms an obtained target pose to the robot's frame
        desired_rotation is a list containing RPY Euler angles"""
        # get target pose and transform to get to target
        marker_pose, t = target_pose, target_transform

        # apply transformation matrix from camera frame to ur5 frame
        self.get_logger().info(f"marker detected: id {target_id} pos.x {marker_pose.position.x} pos.y {marker_pose.position.y} \
pos.z {marker_pose.position.z} ort.x {marker_pose.orientation.x} ort.y {marker_pose.orientation.y} \
ort.z {marker_pose.orientation.z} ort.w {marker_pose.orientation.w}", once=True)
        
        self.get_logger().info(f"transform found: {t.transform.translation.x} {t.transform.translation.y} {t.transform.translation.z} \
{t.transform.rotation.x} {t.transform.rotation.y} {t.transform.rotation.z} {t.transform.rotation.w}")

        # apply rotation quaternion to flip target pose for better tool approach orientation
        rot_quat_x = quaternion_from_euler(desired_rotation[0], desired_rotation[1], desired_rotation[2])

        marker_pose.orientation = quaternion_multiply(rot_quat_x, marker_pose.orientation)

        # apply configured position offsets
        marker_pose.position.x -= self.GRIPPER_OFFSET[0]
        marker_pose.position.y -= self.GRIPPER_OFFSET[1]

        # apply transformation to aruco pose to obtain target pose for arm
        pose_transformed = tf2_geometry_msgs.do_transform_pose(marker_pose, t)

        # adjust target pose position and orientation to default approach gripper offset or as desired
        pose_transformed.position.x += desired_offset[0]
        pose_transformed.position.y += desired_offset[1]
        if override_posz:
            pose_transformed.position.z = self.GRIPPER_OFFSET[2]
        else:
            pose_transformed.position.z += desired_offset[2]

        self.get_logger().info(f"transformed pose: id {target_id} pos.x {pose_transformed.position.x} pos.y {pose_transformed.position.y} \
pos.z {pose_transformed.position.z} ort.x {pose_transformed.orientation.x} ort.y {pose_transformed.orientation.y} \
ort.z {pose_transformed.orientation.z} ort.w {pose_transformed.orientation.w}", once=True)

        return pose_transformed

    def transform_to_pose_2(self, target_pose, target_transform):
        """transforms the target pose to a new frame using the target transform"""
        return tf2_geometry_msgs.do_transform_pose(target_pose, target_transform)

######### end of RobotInterface #########

######### start of DualRobotInterface #########
class DualRobotInterfaceNode(Node):
    """Class instance for the dual motion robot interface
    Members:
    - both_mg_name_: the name of the move group used for dual motion planning
    - arm_0_: the RobotInterfaceNode associated with the first arm (UR5)
    - arm_1_: the RobotInterfaceNode associated with the second arm (UR10e)
    """
    namespace_ = ""
    FK_FAIL_CODE = 2
    IK_FAIL_CODE = 3

    def __init__(self, move_group_name: str, arm_0_node: RobotInterfaceNode, arm_1_node: RobotInterfaceNode, 
            do_fake_publish=False, sync_routine=None, sync_routine_args=None, chosen_planner=None) -> None:
        super().__init__("dual_robot_interface_node", namespace=self.namespace_)
        self.both_mg_name_ = move_group_name
        self.arm_0_ = arm_0_node
        self.arm_1_ = arm_1_node
        self.function_ = sync_routine
        self.function_args_ = sync_routine_args
        self.do_fake_publish = do_fake_publish
        self.planner_id = chosen_planner if chosen_planner else "BFMTkConfigDefault"

        #### fake joint state publishers ####
        if self.do_fake_publish:
            self.fake_publisher_ = self.create_publisher(JointState, '/joint_states', 10)
            self.fake_arm0_publisher_ = self.create_publisher(JointState, '/arm_0/joint_states', 10)
            self.fake_arm1_publisher_ = self.create_publisher(JointState, '/arm_1/joint_states', 10)
            self.fake_joints, self.fake_arm0_joints, self.fake_arm1_joints = JointState(), JointState(), JointState()
            self.load_fake_joints(self.arm_0_.HOME_JOINT_STATE, self.arm_1_.HOME_JOINT_STATE)

            self.timer = self.create_timer(0.01, self.publish_fake_joint_states)
        
        #### synchronous function to be run during node execution (optional) ####
        if sync_routine:
            # start synchronous task in separate thread
            self.sync_thread = threading.Thread(target=self.threaded_function_)
            self.sync_thread.start()


        solo_callback_group_1 = MutuallyExclusiveCallbackGroup()
        solo_callback_group_2 = MutuallyExclusiveCallbackGroup()
        ##### PyBullet interface subscriber #####
        # TODO transfer subscribed topic to config parameter
        self.plan_subscription = self.create_subscription(
            PyBulletPlan,
            '/pybullet_plan',
            self.pybullet_subscriber_callback,
            1,
            # callback_group = solo_callback_group
        )
        self.plan_subscription  # prevent unused variable warning

        ##### BCD Primitive Plan subscriber #####
        # TODO transfer subscribed topic to config parameter
        self.bcd_plan_subscription = self.create_subscription(
            PrimitivePlan,
            '/battery_disassembly_plan',
            self.bcd_subscriber_callback,
            1,
            # callback_group = solo_callback_group_1
        )
        self.bcd_plan_subscription  # prevent unused variable warning

        # ##### LeRobot output subscriber #####
        # # TODO transfer subscribed topic to config parameter
        # self.bcd_plan_subscription = self.create_subscription(
        #     JointState,
        #     '/lerobot_replay_joint_states',
        #     self.lerobot_subscriber_callback,
        #     1,
        #     callback_group = solo_callback_group_2
        # )
        # self.bcd_plan_subscription  # prevent unused variable warning

        print("getting fk")
        print("arm 1 fk", self.arm_1_.get_fk())

    def load_fake_joints(self, arm_0_js, arm_1_js):
        self.fake_arm0_joints.name = arm_0_js.name
        self.fake_arm0_joints.position = arm_0_js.position
        self.fake_arm0_joints.velocity = arm_0_js.velocity
        self.fake_arm0_joints.effort = arm_0_js.effort
        self.fake_arm1_joints.name = arm_1_js.name
        self.fake_arm1_joints.position = arm_1_js.position
        self.fake_arm1_joints.velocity = arm_1_js.velocity
        self.fake_arm1_joints.effort = arm_1_js.effort
        self.fake_joints.name = arm_0_js.name + arm_1_js.name
        self.fake_joints.position = arm_0_js.position + arm_1_js.position
        self.fake_joints.velocity = arm_0_js.velocity + arm_1_js.velocity
        self.fake_joints.effort = arm_0_js.effort + arm_1_js.effort

    def publish_fake_joint_states(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = self.fake_joints.name
        msg.position = self.fake_joints.position
        msg.velocity = self.fake_joints.velocity
        msg.effort = self.fake_joints.effort

        msg_0 = JointState()
        msg_0.header.stamp = msg.header.stamp
        msg_0.name = self.fake_arm0_joints.name
        msg_0.position = self.fake_arm0_joints.position
        msg_0.velocity = self.fake_arm0_joints.velocity
        msg_0.effort = self.fake_arm0_joints.effort

        msg_1 = JointState()
        msg_1.header.stamp = msg.header.stamp
        msg_1.name = self.fake_arm1_joints.name
        msg_1.position = self.fake_arm1_joints.position
        msg_1.velocity = self.fake_arm1_joints.velocity
        msg_1.effort = self.fake_arm1_joints.effort

        self.fake_publisher_.publish(msg)
        self.fake_arm0_publisher_.publish(msg_0)
        self.fake_arm1_publisher_.publish(msg_1)

    def threaded_function_(self):
        while rclpy.ok():
            # self.function_(self.function_args_[0])
            self.function_(self)

    """wrappers for individual arm functionality"""
    def arm_0_vision_identify(self, target_id: int):
        return self.arm_0_.vision_identify(target_id)

    def arm_1_vision_identify(self, target_id: int):
        return self.arm_1_.vision_identify(target_id)

    def arm_0_actuate_gripper(self, position: float, max_effort: float):
        self.arm_0_.actuate_gripper(position, max_effort)
    
    def arm_1_actuate_gripper(self, position: float, max_effort: float):
        self.arm_1_.actuate_gripper(position, max_effort)

    def arm_0_transform_to_pose(self, target_pose, target_transform, target_id, desired_offset=[0.0, 0.0, 0.0],
            desired_rotation=[-np.pi, 0.0, 0.0], override_posz=True) -> Pose:
        return self.arm_0_.transform_to_pose(target_pose, target_transform, target_id, desired_offset, desired_rotation, override_posz)

    def arm_1_transform_to_pose(self, target_pose, target_transform, target_id, desired_offset=[0.0, 0.0, 0.0],
            desired_rotation=[-np.pi, 0.0, 0.0], override_posz=True) -> Pose:
        return self.arm_1_.transform_to_pose(target_pose, target_transform, target_id, desired_offset, desired_rotation, override_posz)
    """"""

    def get_dual_ik(self, target_pose_0: Pose, target_pose_1: Pose) -> (JointState | None, JointState | None):
        """Makes a PositionIKRequest to the ComputeIK service for both arms individually, ignoring collisions, 
        returning two solution joint states, one for each arm"""
        #### get IK request for arm 0 ####
        request_0 = GetPositionIK.Request()

        request_0.ik_request.group_name = self.arm_0_.move_group_name_
        tf_prefix = self.arm_0_.get_namespace()[1:]
        request_0.ik_request.pose_stamped.header.frame_id = self.arm_0_.base_
        request_0.ik_request.pose_stamped.header.stamp = self.arm_0_.get_clock().now().to_msg()

        # apply noise to cause the request to differ during repeat attempts
        request_0.ik_request.pose_stamped.pose = target_pose_0
        request_0.ik_request.avoid_collisions = False

        future_0 = self.arm_0_.ik_client_.call_async(request_0)

        rclpy.spin_until_future_complete(self.arm_0_, future_0)
        if future_0.result() is None:
            self.get_logger().error("Failed to get IK solution 0")
            return None, None

        response_0 = future_0.result()
        if response_0.error_code.val != MoveItErrorCodes.SUCCESS:
            return None, None

        ######## ########

        #### get IK request for arm 1 ####

        request_1 = GetPositionIK.Request()

        request_1.ik_request.group_name = self.arm_1_.move_group_name_
        tf_prefix = self.arm_1_.get_namespace()[1:]
        request_1.ik_request.pose_stamped.header.frame_id = self.arm_1_.base_
        request_1.ik_request.pose_stamped.header.stamp = self.arm_1_.get_clock().now().to_msg()

        # apply noise to cause the request to differ during repeat attempts
        request_1.ik_request.pose_stamped.pose = target_pose_1
        request_1.ik_request.avoid_collisions = False

        future_1 = self.arm_1_.ik_client_.call_async(request_1)

        rclpy.spin_until_future_complete(self.arm_1_, future_1)
        if future_1.result() is None:
            self.get_logger().error("Failed to get IK solution 1")
            return response_0.solution.joint_state, None

        response_1 = future_1.result()
        if response_1.error_code.val != MoveItErrorCodes.SUCCESS:
            return response_0.solution.joint_state, None
        
        ######### #########

        return response_0.solution.joint_state, response_1.solution.joint_state

    def get_best_dual_ik(self, target_pose_0: Pose, target_pose_1: Pose, attempts: int = 100) -> (JointState | None, JointState | None):
        """Gets an IK solution for each arm multiple times, returning the best pair of joint states"""
        best_cost = np.inf
        best_joint_state_0, best_joint_state_1 = None, None

        for _ in range(attempts):
            joint_state_0, joint_state_1 = self.get_dual_ik(target_pose_0, target_pose_1)
            if joint_state_0 is None or joint_state_1 is None:
                continue

            best_joint_state_0 = joint_state_0
            best_joint_state_1 = joint_state_1

        if not best_joint_state_0 or not best_joint_state_1:
            arm_0_report = "0" if not best_joint_state_0 else ""
            arm_1_report = "1" if not best_joint_state_1 else ""
            self.get_logger().error(f"Failed to get IK solution for arm(s) {arm_0_report} {arm_1_report}")

        return best_joint_state_0, best_joint_state_1

    def pybullet_subscriber_callback(self, msg):
        """Subscriber callback function for executing the motion plan generated through the PyBullet simulation"""
        # TODO determine a good planner framework for executing the PyBullet multistage motion plan (maybe a Pilz linear/splining planner?)
        # TODO figure out when it is necessary to delay this callback/delay the output of the pybullet interface node
        for plan_step in msg.plan_steps:
            step_result = self.joints_to_motion(plan_step.arm_0_joint_state, plan_step.arm_1_joint_state, self.planner_id) 
            if step_result != MoveItErrorCodes.SUCCESS:
                self.get_logger().error(f"Error in PyBullet plan execution!: {step_result}; Abandoning motion plan!")
                return

    def primitive_handler(self, arm_node: RobotInterfaceNode, primitive_id: int, primitive_pose: Pose, fk_attempts: int = 10) -> Pose:
        """Helper function to the BCD subscriber that returns a pose that can be sent to
        dual motion planning for the given primitive instruction and data"""
        match primitive_id:
            case 0: # no-op
                for _ in range(fk_attempts):
                    current_pose = arm_node.get_fk()
                    if current_pose is not None:
                        return current_pose     # maintain the current pose for no-ops
                raise RuntimeError(f"Fk solving for arm {arm_node.get_name()} failed.")
            case 1: # go to pose
                # assumption is that the input pose is as desired in the world frame
                t = arm_node.get_transform_frame("world")  # get transform from world frame to the arm's base frame
                return arm_node.transform_to_pose_2(primitive_pose, t)  # return the transformed pose
            case 2: # approach vector (communicated identically to robot interface)
                # assumption is that the input pose is as desired in the world frame
                t = arm_node.get_transform_frame("world")  # get transform from world frame to the arm's base frame
                return arm_node.transform_to_pose_2(primitive_pose, t)  # return the transformed pose
            case 3: # open gripper
                # make call to gripper action service
                arm_node.actuate_gripper(arm_node.GRIPPER_OPEN_POSITION, arm_node.GRIPPER_DEFAULT_EFFORT)
                for _ in range(fk_attempts):
                    current_pose = arm_node.get_fk()
                    if current_pose is not None:
                        return current_pose     # maintain the current arm pose for gripper actuation
                raise RuntimeError(f"Fk solving for arm {arm_node.get_name()} failed.")
            case 4: # close gripper
                # make call to gripper action service
                arm_node.actuate_gripper(arm_node.GRIPPER_GRAB_POSITION, arm_node.GRIPPER_DEFAULT_EFFORT)
                for _ in range(fk_attempts):
                    current_pose = arm_node.get_fk()
                    if current_pose is not None:
                        return current_pose     # maintain the current arm pose for gripper actuation
                raise RuntimeError(f"Fk solving for arm {arm_node.get_name()} failed.")
            case _:
                raise ValueError(f"Invalid primitive ID: {primitive_id}")

    def bcd_subscriber_callback(self, msg: PrimitivePlan):
        """Subscriber callback function for executing the motion plan generated through the BCD algorithm"""
        # TODO figure out when it is necessary to delay this callback/delay the output of the BCD node
        # TODO implement primitive handling (will involve calls to IK motion planner, gripper service)
        planner_id = "BFMTkConfigDefault"

        # make sure the step lengths are equivalent by padding the shorter plan with empty steps
        if len(msg.arm_0_plan_steps) < len(msg.arm_1_plan_steps):      # handle empty step sequences for arm 0
            for _ in range(len(msg.arm_1_plan_steps) - len(msg.arm_0_plan_steps)):
                dummy_step = PrimitivePlanStep()
                dummy_step.seq_length = 0
                msg.arm_0_plan_steps.append(dummy_step)
        elif len(msg.arm_1_plan_steps) < len(msg.arm_0_plan_steps):      # handle empty step sequences for arm 1
            for _ in range(len(msg.arm_0_plan_steps) - len(msg.arm_1_plan_steps)):
                dummy_step = PrimitivePlanStep()
                dummy_step.seq_length = 0
                msg.arm_1_plan_steps.append(dummy_step)

        for step_idx in range(msg.plan_length):
            # make sure the step lengths are equivalent by padding the shorter plan with no-ops
            if msg.arm_0_plan_steps[step_idx].seq_length < msg.arm_1_plan_steps[step_idx].seq_length:
                for _ in range(msg.arm_1_plan_steps[step_idx].seq_length - msg.arm_0_plan_steps[step_idx].seq_length):
                    msg.arm_0_plan_steps[step_idx].primitive_ids.append(0)
                    msg.arm_0_plan_steps[step_idx].poses.append(Pose())
                msg.arm_0_plan_steps[step_idx].seq_length = msg.arm_1_plan_steps[step_idx].seq_length
            elif msg.arm_1_plan_steps[step_idx].seq_length < msg.arm_0_plan_steps[step_idx].seq_length:
                for _ in range(msg.arm_0_plan_steps[step_idx].seq_length - msg.arm_1_plan_steps[step_idx].seq_length):
                    msg.arm_1_plan_steps[step_idx].primitive_ids.append(0)
                    msg.arm_1_plan_steps[step_idx].poses.append(Pose())
                msg.arm_1_plan_steps[step_idx].seq_length = msg.arm_0_plan_steps[step_idx].seq_length
            # translate each primitive operation in the pair of plan steps to actionable poses, then execute them
            for primitive_idx in range(msg.arm_0_plan_steps[step_idx].seq_length):
                arm0_target_pose = self.primitive_handler(self.arm_0_, msg.arm_0_plan_steps[step_idx].primitive_ids[primitive_idx],
                                                            msg.arm_0_plan_steps[step_idx].poses[primitive_idx])
                arm1_target_pose = self.primitive_handler(self.arm_1_, msg.arm_1_plan_steps[step_idx].primitive_ids[primitive_idx],
                                                            msg.arm_1_plan_steps[step_idx].poses[primitive_idx])
                if self.get_dual_motion_plan(arm0_target_pose, arm1_target_pose, planner_id) != MoveItErrorCodes.SUCCESS:
                    self.get_logger().error(f"Execution of plan with length {msg.plan_length} failed.")
                    self.destroy_subscription(self.bcd_plan_subscription)
                    return

        # cleanup by destroying the subscriber (temporary solution to input delay/frequency problem)
        self.get_logger().info(f"Successfully completed execution of plan with length {msg.plan_length}.")
        self.destroy_subscription(self.bcd_plan_subscription)
        self.destroy_node()

    def lerobot_subscriber_callback(self, msg: JointState):
        """Subscriber callback function for actions requested from the LeRobot package"""
        planner_id = "LIN"
        # planner_id = "BFMTkConfigDefault"

        arm0_js, arm1_js = JointState(), JointState()
        arm0_js.header.stamp, arm1_js.header.stamp = msg.header.stamp, msg.header.stamp
        # retrieve arm0 joint state goal
        arm0_js.name = msg.name[:6]
        arm0_js.position = msg.position[:6]
        arm0_js.velocity = msg.velocity[:6]
        arm0_js.effort = msg.effort[:6]
        # retrieve arm0 gripper joint state goal
        arm0_gripper_val = msg.position[6]
        # retrieve arm1 joint state goal
        arm1_js.name = msg.name[7:13]
        arm1_js.position = msg.position[7:13]
        arm1_js.velocity = msg.velocity[7:13]
        arm1_js.effort = msg.effort[7:13]
        # retrieve arm1 gripper joint state goal
        arm1_gripper_val = msg.position[13]

        # plan and execute dual arm motion
        if self.joints_to_motion(arm0_js, arm1_js, planner_id) != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(f"Execution of plan with length {msg.plan_length} failed.")
            self.destroy_subscription(self.bcd_plan_subscription)
            return
        
        # actuate grippers
        self.arm_0_actuate_gripper(arm0_gripper_val, self.arm_0_.GRIPPER_DEFAULT_EFFORT)
        self.arm_1_actuate_gripper(arm1_gripper_val, self.arm_1_.GRIPPER_DEFAULT_EFFORT)

        # cleanup
        self.get_logger().info(f"Successfully completed execution of LeRobot action request.")

    def joints_to_motion(self, arm_0_target_joint_state, arm_1_target_joint_state, planner_id: str, attempts: int = 10) -> int:
        """Passes the input joint states into a MoveGroupInterface for dual motion planning 
        and execution, returning the error code for the planning"""
        # encode joint names for ctypes transfer
        arm_0_target_joint_state_encoded = [x.encode('utf_8') for x in arm_0_target_joint_state.name]
        arm_1_target_joint_state_encoded = [x.encode('utf_8') for x in arm_1_target_joint_state.name]

        for k in range(attempts):
            # send joint states to move group interface via ctypes python binding
            plan_result = lib.dual_ur_motion(
                (ctypes.c_char_p * 12)(*arm_0_target_joint_state_encoded), (ctypes.c_float* 12)(*arm_0_target_joint_state.position),
                (ctypes.c_float* 12)(*arm_0_target_joint_state.velocity), (ctypes.c_float* 12)(*arm_0_target_joint_state.effort),
                (ctypes.c_char_p * 12)(*arm_1_target_joint_state_encoded), (ctypes.c_float* 12)(*arm_1_target_joint_state.position),
                (ctypes.c_float* 12)(*arm_1_target_joint_state.velocity), (ctypes.c_float* 12)(*arm_1_target_joint_state.effort), 
                ctypes.c_char_p(self.both_mg_name_.encode('utf-8')), ctypes.c_char_p(planner_id.encode('utf-8'))
            )
            if plan_result == MoveItErrorCodes.SUCCESS:
                self.get_logger().info("Dual motion planning succeeded")
                if self.do_fake_publish:
                    fake_joints_0, fake_joints_1 = JointState(), JointState()
                    fake_joints_0.name = arm_0_target_joint_state.name
                    fake_joints_0.position = arm_0_target_joint_state.position
                    fake_joints_0.velocity = arm_0_target_joint_state.velocity
                    fake_joints_0.effort = arm_0_target_joint_state.effort
                    fake_joints_1.name = arm_1_target_joint_state.name
                    fake_joints_1.position = arm_1_target_joint_state.position
                    fake_joints_1.velocity = arm_1_target_joint_state.velocity
                    fake_joints_1.effort = arm_1_target_joint_state.effort
                    self.load_fake_joints(fake_joints_0, fake_joints_1)
                    self.publish_fake_joint_states()
                return plan_result
            elif plan_result == MoveItErrorCodes.GOAL_STATE_INVALID or plan_result == MoveItErrorCodes.GOAL_IN_COLLISION:
                self.get_logger().error(f"Dual motion planning attempt {k} of {attempts} failed by invalid goal state: {plan_result}; delaying then exiting...")
                time.sleep(10)
                return plan_result
            elif plan_result == MoveItErrorCodes.TIMED_OUT:
                self.get_logger().error(f"Dual motion planning attempt {k} of {attempts} failed by timing out: {plan_result}; exiting now...")
                return plan_result
            else:
                self.get_logger().error(f"Dual motion planning attempt {k} of {attempts} failed: {plan_result}; trying again...")
        self.get_logger().error("Dual motion planning ultimately failed")
        return plan_result


    def get_dual_motion_plan(self, arm_0_target_pose: Pose, arm_1_target_pose: Pose, planner_id: str, linear: bool = False, attempts: int = 10) -> int:
        """Obtains the best IK solutions for both arms separately, then passes the solution joint states into 
        a MoveGroupInterface for dual motion planning and execution, returning the error code for the planning"""

        self.get_logger().info(f"Getting motion plan with planner {planner_id}...")
        plan_result = 0

        for j in range(attempts):
            #### get two independent IK solutions, ignoring collisions ####
            for i in range(attempts):
                arm_0_target_joint_state, arm_1_target_joint_state = self.get_best_dual_ik(arm_0_target_pose, arm_1_target_pose)
                if arm_0_target_joint_state and arm_1_target_joint_state:
                    self.get_logger().info("Sucessfully obtained Arms target joint state")
                    break
                self.get_logger().error(f"Failed to get target Arms target joint state, attempt {i}")
            if arm_0_target_joint_state is None or arm_1_target_joint_state is None:
                self.get_logger().error("Ultimately failed to get target Arms target joint state")
                return self.IK_FAIL_CODE

            ######## ########

            #### send IK solution joint states to move group interface motion planning ####
            # mid_0, mid_1 = int(len(arm_0_target_joint_state.name)/2), int(len(arm_1_target_joint_state.name)/2)
            # arm_0_target_joint_state.name = arm_0_target_joint_state.name[:mid_0]           # remove gripper joints from motion planning
            # arm_0_target_joint_state.position = arm_0_target_joint_state.position[:mid_0]   #
            # arm_0_target_joint_state.velocity = arm_0_target_joint_state.velocity[:mid_0]   #
            # arm_0_target_joint_state.effort = arm_0_target_joint_state.effort[:mid_0]       #
            # arm_1_target_joint_state.name = arm_1_target_joint_state.name[:mid_1]           # remove gripper joints from motion planning
            # arm_1_target_joint_state.position = arm_1_target_joint_state.position[:mid_1]   #
            # arm_1_target_joint_state.velocity = arm_1_target_joint_state.velocity[:mid_1]   #
            # arm_1_target_joint_state.effort = arm_1_target_joint_state.effort[:mid_1]       #
            # plan_result = self.joints_to_motion(arm_0_target_joint_state, arm_1_target_joint_state, planner_id, linear, attempts)
            # if plan_result == MoveItErrorCodes.SUCCESS:
            #         self.get_logger().info("Dual motion planning succeeded")
            #         if self.do_fake_publish:
            #             fake_joints_0, fake_joints_1 = JointState(), JointState()
            #             fake_joints_0.name = arm_0_target_joint_state.name[:mid_0]
            #             fake_joints_0.position = arm_0_target_joint_state.position[:mid_0]
            #             fake_joints_0.velocity = arm_0_target_joint_state.velocity[:mid_0]
            #             fake_joints_0.effort = arm_0_target_joint_state.effort[:mid_0]
            #             fake_joints_1.name = arm_1_target_joint_state.name[mid_1:]
            #             fake_joints_1.position = arm_1_target_joint_state.position[mid_1:]
            #             fake_joints_1.velocity = arm_1_target_joint_state.velocity[mid_1:]
            #             fake_joints_1.effort = arm_1_target_joint_state.effort[mid_1:]
            #             self.load_fake_joints(fake_joints_0, fake_joints_1)
            #             self.publish_fake_joint_states()
            #         return plan_result
            #     elif plan_result == MoveItErrorCodes.GOAL_STATE_INVALID or plan_result == MoveItErrorCodes.GOAL_IN_COLLISION:
            #         self.get_logger().error(f"Dual motion planning attempt {k} of {attempts} failed by invalid goal state: {plan_result}; delaying then exiting...")
            #         time.sleep(10)
            #         return plan_result
            #     elif plan_result == MoveItErrorCodes.TIMED_OUT:
            #         self.get_logger().error(f"Dual motion planning attempt {k} of {attempts} failed by timing out: {plan_result}; exiting now...")
            #         return plan_result
            #     else:
            #         self.get_logger().error(f"Dual motion planning attempt {k} of {attempts} failed: {plan_result}; trying again...")
            # self.get_logger().error("Dual motion planning ultimately failed")
            # return plan_result
            ######## ########

            #### send IK solution joint states to move group interface motion planning ####

            # grab midpoints to separate modified joint states for each arm from unmodified joint states in each solution
            mid_0, mid_1 = int(len(arm_0_target_joint_state.name)/2), int(len(arm_1_target_joint_state.name)/2)

            # encode joint names for ctypes transfer
            arm_0_target_joint_state_encoded = [x.encode('utf_8') for x in arm_0_target_joint_state.name]
            arm_1_target_joint_state_encoded = [x.encode('utf_8') for x in arm_1_target_joint_state.name]

            for k in range(attempts):
                # send joint states to move group interface via ctypes python binding
                plan_result = lib.dual_ur_motion((ctypes.c_char_p * 12)(*arm_0_target_joint_state_encoded[:mid_0]), (ctypes.c_float* 12)(*arm_0_target_joint_state.position[:mid_0]),
                    (ctypes.c_float* 12)(*arm_0_target_joint_state.velocity[:mid_0]), (ctypes.c_float* 12)(*arm_0_target_joint_state.effort[:mid_0]),
                    (ctypes.c_char_p * 12)(*arm_1_target_joint_state_encoded[mid_1:]), (ctypes.c_float* 12)(*arm_1_target_joint_state.position[mid_1:]),
                    (ctypes.c_float* 12)(*arm_1_target_joint_state.velocity[mid_1:]), (ctypes.c_float* 12)(*arm_1_target_joint_state.effort[mid_1:]), 
                    ctypes.c_char_p(self.both_mg_name_.encode('utf-8')), ctypes.c_char_p(planner_id.encode('utf-8')))
                if plan_result == MoveItErrorCodes.SUCCESS:
                    self.get_logger().info("Dual motion planning succeeded")
                    if self.do_fake_publish:
                        fake_joints_0, fake_joints_1 = JointState(), JointState()
                        fake_joints_0.name = arm_0_target_joint_state.name[:mid_0]
                        fake_joints_0.position = arm_0_target_joint_state.position[:mid_0]
                        fake_joints_0.velocity = arm_0_target_joint_state.velocity[:mid_0]
                        fake_joints_0.effort = arm_0_target_joint_state.effort[:mid_0]
                        fake_joints_1.name = arm_1_target_joint_state.name[mid_1:]
                        fake_joints_1.position = arm_1_target_joint_state.position[mid_1:]
                        fake_joints_1.velocity = arm_1_target_joint_state.velocity[mid_1:]
                        fake_joints_1.effort = arm_1_target_joint_state.effort[mid_1:]
                        self.load_fake_joints(fake_joints_0, fake_joints_1)
                        self.publish_fake_joint_states()
                    return plan_result
                elif plan_result == MoveItErrorCodes.GOAL_STATE_INVALID or plan_result == MoveItErrorCodes.GOAL_IN_COLLISION:
                    self.get_logger().error(f"Dual motion planning attempt {k} of {attempts} failed by invalid goal state: {plan_result}; delaying then exiting...")
                    time.sleep(10)
                    return plan_result
                elif plan_result == MoveItErrorCodes.TIMED_OUT:
                    self.get_logger().error(f"Dual motion planning attempt {k} of {attempts} failed by timing out: {plan_result}; exiting now...")
                    return plan_result
                else:
                    self.get_logger().error(f"Dual motion planning attempt {k} of {attempts} failed: {plan_result}; trying again...")
        self.get_logger().error("Dual motion planning ultimately failed")
        return plan_result

        ######## ########
######### end of DualRobotInterface #########
    
def dm_pnp_demo(dual_rin, args=None):
    """full scale dual motion planning pick and place demonstration, compatible with simulation"""
    # initialization
    BLOCK_HEIGHT = 0.0635 # length of cube face, in meters
    planner_name = "BFMTkConfigDefault"
    DELAY_VAL = 1
    DELAY_VAL_GRIPPER = 1


    dual_rin.get_logger().info("\n\n\n#####\nstart of demo\n#####\n\n\n")

    # reset to home positions
    result = dual_rin.get_dual_motion_plan(dual_rin.arm_0_.HOME_POSE, dual_rin.arm_1_.HOME_POSE, planner_name)
    if result != 1:
        dual_rin.get_logger().info("Task failed, shutting down")
        return
    dual_rin.arm_0_actuate_gripper(position=dual_rin.arm_0_.GRIPPER_OPEN_POSITION, max_effort=dual_rin.arm_0_.GRIPPER_DEFAULT_EFFORT)
    dual_rin.arm_1_actuate_gripper(position=dual_rin.arm_1_.GRIPPER_OPEN_POSITION, max_effort=dual_rin.arm_1_.GRIPPER_DEFAULT_EFFORT)
    time.sleep(DELAY_VAL_GRIPPER)

    # ids for all 10 blocks (ur5 gets 0-4, ur10e gets 5-10 w/o 9)
    ids_ur5, ids_ur10e = [0, 1, 2, 3, 4], [5, 6, 7, 8, 10]

    # start moving blocks
    idx = 0
    while idx < 5:
        # identify blocks
        pose_target, t, pose_frame = dual_rin.arm_0_vision_identify(ids_ur5[idx])
        pose_ur5 = dual_rin.arm_0_transform_to_pose(pose_target, t, pose_frame)
        pose_target, t, pose_frame = dual_rin.arm_1_vision_identify(ids_ur10e[idx])
        pose_ur10e = dual_rin.arm_1_transform_to_pose(pose_target, t, pose_frame)
        time.sleep(DELAY_VAL)

        # move to blocks
        result = dual_rin.get_dual_motion_plan(pose_ur5, pose_ur10e, planner_name, attempts=20)
        time.sleep(DELAY_VAL)
        if result != 1:
            dual_rin.get_logger().info("Missed planning, getting new pose...")
            misses = 0
            while misses < 10:
                pose_target, t, pose_frame = dual_rin.arm_0_vision_identify(ids_ur5[idx])
                pose_ur5 = dual_rin.arm_0_transform_to_pose(pose_target, t, pose_frame)
                pose_target, t, pose_frame = dual_rin.arm_1_vision_identify(ids_ur10e[idx])
                pose_ur10e = dual_rin.arm_1_transform_to_pose(pose_target, t, pose_frame)
                time.sleep(DELAY_VAL)
                result = dual_rin.get_dual_motion_plan(pose_ur5, pose_ur10e, planner_name, attempts=20)
                time.sleep(DELAY_VAL)
                if result != 1:
                    dual_rin.get_logger().info(f"New plan {misses} failed with code {result}, getting new pose...")
                    misses += 1
                else:
                    break
            if result != 1:
                dual_rin.get_logger().info("Task failed, shutting down")
                return
        time.sleep(DELAY_VAL)

        # lower and pick up blocks
        pose_ur5.position.z = dual_rin.arm_0_.PRELIFT_HEIGHT
        pose_ur10e.position.z = dual_rin.arm_1_.PRELIFT_HEIGHT
        result = dual_rin.get_dual_motion_plan(pose_ur5, pose_ur10e, planner_name, attempts=20)
        if result != 1:
            dual_rin.get_logger().info("Grab positions are too close, reseting back to home...")
            result = dual_rin.get_dual_motion_plan(dual_rin.arm_0_.HOME_POSE, dual_rin.arm_1_.HOME_POSE, planner_name)
            if result != 1:
                dual_rin.get_logger().info("Task failed, shutting down")
                return
            # idx -= 1
            continue
            time.sleep(DELAY_VAL)
        dual_rin.arm_0_actuate_gripper(position=dual_rin.arm_0_.GRIPPER_GRAB_POSITION, max_effort=dual_rin.arm_0_.GRIPPER_DEFAULT_EFFORT)
        dual_rin.arm_1_actuate_gripper(position=dual_rin.arm_1_.GRIPPER_GRAB_POSITION, max_effort=dual_rin.arm_1_.GRIPPER_DEFAULT_EFFORT)
        time.sleep(DELAY_VAL_GRIPPER)

        # pick up and carry blocks
        pose_ur5.position.z = dual_rin.arm_0_.LIFT_HEIGHT
        pose_ur10e.position.z = dual_rin.arm_1_.LIFT_HEIGHT
        result = dual_rin.get_dual_motion_plan(pose_ur5, pose_ur10e, planner_name, attempts=10)
        if result != 1:
            dual_rin.get_logger().info("Task failed, shutting down")
            return
        time.sleep(DELAY_VAL)
        pose_ur5.position.x, pose_ur5.position.y = dual_rin.arm_0_.DROPOFF_EXTERN.position.x, dual_rin.arm_0_.DROPOFF_EXTERN.position.y + (BLOCK_HEIGHT + 0.05)*idx
        pose_ur10e.position.x, pose_ur10e.position.y = dual_rin.arm_1_.DROPOFF_EXTERN.position.x, dual_rin.arm_1_.DROPOFF_EXTERN.position.y  + (BLOCK_HEIGHT+0.05)*idx
        result = dual_rin.get_dual_motion_plan(pose_ur5, pose_ur10e, planner_name, attempts=20)
        if result != 1:
            dual_rin.get_logger().info("Task failed, shutting down")
            return
        time.sleep(DELAY_VAL)

        # lower and drop blocks
        pose_ur5.position.z = dual_rin.arm_0_.PRELIFT_HEIGHT
        pose_ur10e.position.z = dual_rin.arm_1_.PRELIFT_HEIGHT
        result = dual_rin.get_dual_motion_plan(pose_ur5, pose_ur10e, planner_name, attempts=10)
        if result != 1:
            dual_rin.get_logger().info("Task failed, shutting down")
            return
        time.sleep(DELAY_VAL)
        dual_rin.arm_0_actuate_gripper(position=dual_rin.arm_0_.GRIPPER_OPEN_POSITION, max_effort=dual_rin.arm_0_.GRIPPER_DEFAULT_EFFORT)
        dual_rin.arm_1_actuate_gripper(position=dual_rin.arm_1_.GRIPPER_OPEN_POSITION, max_effort=dual_rin.arm_1_.GRIPPER_DEFAULT_EFFORT)
        time.sleep(DELAY_VAL_GRIPPER)

        # return to lift height
        pose_ur5.position.z =  dual_rin.arm_0_.LIFT_HEIGHT
        pose_ur10e.position.z =  dual_rin.arm_1_.LIFT_HEIGHT
        result = dual_rin.get_dual_motion_plan(pose_ur5, pose_ur10e, planner_name, attempts=10)
        if result != 1:
            dual_rin.get_logger().info("Task failed, shutting down")
            return
        time.sleep(DELAY_VAL)

        idx += 1

    # cleanup
    dual_rin.get_logger().info("End of task, shutting down...")

    dual_rin.get_logger().info("\n\n\n#####\nend of demo\n#####\n\n\n")
    return

def gripper_test(dual_rin, args=None):
    """simple actuation test for each gripper
    DEPRECATED -> USE DEDICATED GRIPPER TEST NODE
    """
    dual_rin.arm_0_.actuate_gripper(position=dual_rin.arm_0_.GRIPPER_OPEN_POSITION, max_effort=dual_rin.arm_0_.GRIPPER_DEFAULT_EFFORT)
    dual_rin.arm_1_.actuate_gripper(position=dual_rin.arm_1_.GRIPPER_OPEN_POSITION, max_effort=dual_rin.arm_1_.GRIPPER_DEFAULT_EFFORT)
    time.sleep(3)
    dual_rin.arm_0_.actuate_gripper(position=dual_rin.arm_0_.GRIPPER_GRAB_POSITION, max_effort=dual_rin.arm_0_.GRIPPER_DEFAULT_EFFORT)
    dual_rin.arm_1_.actuate_gripper(position=dual_rin.arm_1_.GRIPPER_GRAB_POSITION, max_effort=dual_rin.arm_1_.GRIPPER_DEFAULT_EFFORT)
    time.sleep(3)

def cell_side_grip_test(dual_rin, args=None):
    # initialization
    planner_name = "BFMTkConfigDefault"
    DELAY_VAL = 0
    DELAY_VAL_GRIPPER = 0

    dual_rin.get_logger().info("\n\n\n#####\nstart of demo\n#####\n\n\n")

    # reset to home positions
    result = dual_rin.get_dual_motion_plan(dual_rin.arm_0_.HOME_POSE, dual_rin.arm_1_.HOME_POSE, planner_name)
    if result != 1:
        dual_rin.get_logger().info("Task failed, shutting down")
        return
    dual_rin.arm_0_actuate_gripper(position=dual_rin.arm_0_.GRIPPER_OPEN_POSITION, max_effort=dual_rin.arm_0_.GRIPPER_DEFAULT_EFFORT)
    dual_rin.arm_1_actuate_gripper(position=dual_rin.arm_1_.GRIPPER_OPEN_POSITION, max_effort=dual_rin.arm_1_.GRIPPER_DEFAULT_EFFORT)
    time.sleep(DELAY_VAL_GRIPPER)

    # identify cells
    # pose_target, t, pose_frame = dual_rin.arm_0_vision_identify(10)
    # pose_ur5 = dual_rin.arm_0_transform_to_pose(pose_target, t, pose_frame, desired_rotation=[0.0, -np.pi/2, 0.0], desired_offset=[-0.1, 0.0, 0.1305], override_posz=True) 
    #     #, desired_offset=[-0.1, 0.0, 0.1035],
    #     # desired_rotation=[-np.pi/2, 0.0, 0.0], override_posz=True)
    pose_target, t, pose_frame = dual_rin.arm_1_vision_identify(1)
    # pose_ur10e = dual_rin.arm_1_transform_to_pose(pose_target, t, pose_frame, desired_offset=[0.0, 0.1, 0.0], desired_rotation=[-np.pi/2, np.pi/2, 0.0])
    pose_ur10e = dual_rin.arm_1_transform_to_pose(pose_target, t, pose_frame, desired_offset=[0.0, 0.1, 0.0], desired_rotation=[-np.pi/2, 0.0, np.pi])
        #, desired_offset=[-0.1, 0.0, 0.1035],
        # desired_rotation=[-np.pi/2, 0.0, 0.0], override_posz=True)
    time.sleep(DELAY_VAL)
    pose_ur5 = Pose()
    pose_ur5.position.x = -0.01
    pose_ur5.position.y = -0.08837589475267559
    pose_ur5.position.z = 0.5803693935859697
    pose_ur5.orientation = quaternion_from_euler(0.0, np.pi/2, 0.0)
    
    # pose_ur10e = Pose()
    # pose_ur10e.position.x = 
    # pose_ur10e.position.y = -0.09049836683455582
    # pose_ur10e.position.z = 0.6154254575812302
    # pose_ur10e.orientation = quaternion_from_euler(0.0, 0.0, 0.0)

    # move to pregrasp positions
    result = dual_rin.get_dual_motion_plan(pose_ur5, pose_ur10e, planner_name, attempts=20)
    time.sleep(DELAY_VAL)
    if result != 1:
        dual_rin.get_logger().info("Missed planning, getting new pose...")
        misses = 0
        while misses < 10:
            pose_target, t, pose_frame = dual_rin.arm_0_vision_identify(10)
            pose_ur5 = dual_rin.arm_0_transform_to_pose(pose_target, t, pose_frame, desired_offset=[0.0, -0.2, 0.0], desired_rotation=[-np.pi/2, np.pi/2, 0.0], override_posz=True)
            pose_target, t, pose_frame = dual_rin.arm_1_vision_identify(1)
            pose_ur10e = dual_rin.arm_1_transform_to_pose(pose_target, t, pose_frame, desired_offset=[0.0, 0.1, 0.0], desired_rotation=[-np.pi/2, np.pi/2, 0.0])
            time.sleep(DELAY_VAL)
            result = dual_rin.get_dual_motion_plan(pose_ur5, pose_ur10e, planner_name, attempts=20)
            time.sleep(DELAY_VAL)
            if result != 1:
                dual_rin.get_logger().info(f"New plan {misses} failed with code {result}, getting new pose...")
                misses += 1
            else:
                break
        if result != 1:
            dual_rin.get_logger().info("Task failed, shutting down")
            return
    time.sleep(DELAY_VAL)

    # move in and grasp cells
    ur5_init_x = pose_ur5.position.x
    ur10e_init_x = pose_ur10e.position.x
    pose_ur5.position.x += 0.005
    pose_ur10e.position.x += 0.005
    result = dual_rin.get_dual_motion_plan(pose_ur5, pose_ur10e, planner_name, attempts=20)
    if result != 1:
        dual_rin.get_logger().info("Grab positions are too close, reseting back to home...")
        result = dual_rin.get_dual_motion_plan(dual_rin.arm_0_.HOME_POSE, dual_rin.arm_1_.HOME_POSE, planner_name)
        if result != 1:
            dual_rin.get_logger().info("Task failed, shutting down")
            return
        return
    dual_rin.arm_0_actuate_gripper(position=dual_rin.arm_0_.GRIPPER_GRAB_POSITION, max_effort=dual_rin.arm_0_.GRIPPER_DEFAULT_EFFORT)
    dual_rin.arm_1_actuate_gripper(position=dual_rin.arm_1_.GRIPPER_GRAB_POSITION, max_effort=dual_rin.arm_1_.GRIPPER_DEFAULT_EFFORT)
    time.sleep(DELAY_VAL_GRIPPER)

    # separate the cells
    pose_ur5.position.x = ur5_init_x
    pose_ur10e.position.x = ur10e_init_x
    result = dual_rin.get_dual_motion_plan(pose_ur5, pose_ur10e, planner_name, attempts=10)
    if result != 1:
        dual_rin.get_logger().info("Task failed, shutting down")
        return
    time.sleep(DELAY_VAL)
    # pose_ur5.position.x, pose_ur5.position.y = dual_rin.arm_0_.DROPOFF_EXTERN.position.x, dual_rin.arm_0_.DROPOFF_EXTERN.position.y + (BLOCK_HEIGHT + 0.05)*idx
    # pose_ur10e.position.x, pose_ur10e.position.y = dual_rin.arm_1_.DROPOFF_EXTERN.position.x, dual_rin.arm_1_.DROPOFF_EXTERN.position.y  + (BLOCK_HEIGHT+0.05)*idx
    # result = dual_rin.get_dual_motion_plan(pose_ur5, pose_ur10e, planner_name, attempts=20)
    # if result != 1:
    #     dual_rin.get_logger().info("Task failed, shutting down")
    #     return
    # time.sleep(DELAY_VAL)

    # release grippers
    dual_rin.arm_0_actuate_gripper(position=dual_rin.arm_0_.GRIPPER_GRAB_POSITION, max_effort=dual_rin.arm_0_.GRIPPER_DEFAULT_EFFORT)
    dual_rin.arm_1_actuate_gripper(position=dual_rin.arm_1_.GRIPPER_GRAB_POSITION, max_effort=dual_rin.arm_1_.GRIPPER_DEFAULT_EFFORT)
    time.sleep(DELAY_VAL_GRIPPER)

        # cleanup
    dual_rin.get_logger().info("End of task, shutting down...")

    dual_rin.get_logger().info("\n\n\n#####\nend of demo\n#####\n\n\n")

def test_cell_strength(dual_rin, args=None):
    # init
    planner_name = "BFMTkConfigDefault"
    arm_0_init_state, arm_1_init_state = JointState(), JointState()
    arm_0_init_state.name = ["arm_0_shoulder_pan_joint", "arm_0_shoulder_lift_joint", "arm_0_elbow_joint", "arm_0_wrist_1_joint", "arm_0_wrist_3_joint", "arm_0_wrist_2_joint"]
    arm_0_init_state.position = [-0.17683917680849248, -1.0201872030841272, 1.8018288612365723, -3.9219163099872034, -1.5177033583270472, -0.4128316084491175]
    arm_0_init_state.effort = [-0.271259069442749, -3.176645517349243, -1.0155401229858398, -0.022875618189573288, -0.007625205907970667, -0.12047825008630753]
    arm_1_init_state.name = ["shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", "shoulder_pan_joint"]
    arm_1_init_state.position = [-2.004134794274801, -2.111377000808716, -0.5071099561503907, 3.1376986503601074, -1.4807093779193323, -0.4828580061541956]
    arm_1_init_state.effort = [5.639028072357178, 2.8079535961151123, 0.4931488633155823, 0.2813922166824341, -0.05031103268265724, 0.16805781424045563]
    arm_0_pullback_state, arm_1_pullback_state = JointState(), JointState()
    arm_0_pullback_state.name = ["arm_0_shoulder_pan_joint", "arm_0_shoulder_lift_joint", "arm_0_elbow_joint", "arm_0_wrist_1_joint", "arm_0_wrist_3_joint", "arm_0_wrist_2_joint"]
    arm_0_pullback_state.position = [-0.2861822287188929, -1.0326255003558558, 1.8068695068359375, -3.9241607824908655, -1.517799202595846, -0.224100414906637]
    arm_0_pullback_state.effort = [0.07622155547142029, -3.474806308746338, -0.8832733631134033, -0.10065271705389023, -0.05795156583189964, 0.03812602907419205]
    arm_1_pullback_state.name = ["shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint", "wrist_3_joint", "shoulder_pan_joint"]
    arm_1_pullback_state.position = [-2.003726144830221, -2.111328125, -0.5070274633220215, 3.1378581523895264, -1.4807575384723108, -0.5172579924212855]
    arm_1_pullback_state.effort = [5.383959770202637, 2.7399115562438965, 0.5046893358230591, 0.2663553059101105, -0.1104990690946579, -0.5560330152511597]

    # go to hardcoded init joint states
    dual_rin.joints_to_motion(arm_0_init_state, arm_1_init_state, planner_name)

    # close grippers
    dual_rin.arm_0_actuate_gripper(position=dual_rin.arm_0_.GRIPPER_GRAB_POSITION, max_effort=dual_rin.arm_0_.GRIPPER_DEFAULT_EFFORT)
    dual_rin.arm_1_actuate_gripper(position=dual_rin.arm_1_.GRIPPER_GRAB_POSITION, max_effort=dual_rin.arm_1_.GRIPPER_DEFAULT_EFFORT)

    # go to hardcoded pullback joint states
    dual_rin.joints_to_motion(arm_0_pullback_state, arm_1_pullback_state, planner_name)

    # release grippers
    dual_rin.arm_0_actuate_gripper(position=dual_rin.arm_0_.GRIPPER_OPEN_POSITION, max_effort=dual_rin.arm_0_.GRIPPER_DEFAULT_EFFORT)
    dual_rin.arm_1_actuate_gripper(position=dual_rin.arm_1_.GRIPPER_OPEN_POSITION, max_effort=dual_rin.arm_1_.GRIPPER_DEFAULT_EFFORT)

def get_arm_poses(dual_rin: DualRobotInterfaceNode, args=None):
    """debug test function that prints out the end effector pose for each arm"""
    for _ in range(10):
        arm0_fk = dual_rin.arm_0_.get_fk()
        arm1_fk = dual_rin.arm_1_.get_fk()
        if arm0_fk is not None and arm1_fk is not None:
            break
    dual_rin.get_logger().info(
        f"\nArm 0 EE pose: {arm0_fk}\nArm 1 EE pose: {arm1_fk}\n\n"
    )
    time.sleep(5)

def test_triad_primitives(dual_rin, args=None):
    # init
    planner_name = "BFMTkConfigDefault"

    # get target pose
    # pose_target, t, pose_frame = dual_rin.arm_0_vision_identify(ids_ur5[idx])
    # pose_ur5 = dual_rin.arm_0_transform_to_pose(pose_target, t, pose_frame)
    # pose_target, t, pose_frame = dual_rin.arm_1_vision_identify(ids_ur10e[idx])
    # pose_ur10e = dual_rin.arm_1_transform_to_pose(pose_target, t, pose_frame)


def test_wedge_primitives(dual_rin, args=None):
    # init
    planner_name = "BFMTkConfigDefault"


def destroy_nodes(*nodes):
    for node in nodes:
        node.destroy_node()


if __name__ == "__main__":
    # asyncio.run(main_async())
    # exit()

    rclpy.init()
    rin_ur5 = RobotInterfaceNode("ur5", "compute_ik", "compute_fk", "arm_0_base_link", "arm_0_tool0")
    rin_ur10e = RobotInterfaceNode("ur10e", "compute_ik", "compute_fk", "base_link", "tool0")   
    dual_rin = DualRobotInterfaceNode(move_group_name="both", arm_0_node=rin_ur5, arm_1_node=rin_ur10e,
        do_fake_publish=False    # use this for running the real robot an external pipeline
        # do_fake_publish=True   # use this for running fake publishers with an external pipeline
        
        # routine functions synchronized with multithreading
        # do_fake_publish=True, sync_routine=cell_side_grip_test
        # do_fake_publish=False, sync_routine=test_cell_strength
        # do_fake_publish=False, sync_routine=dm_pnp_demo
        # do_fake_publish=True, sync_routine=dm_pnp_demo
        # do_fake_publish=False, sync_routine=gripper_test
        # do_fake_publish=False, sync_routine=get_arm_poses
    )   


    # Use MultiThreadedExecutor to handle multiple callbacks concurrently
    executor = MultiThreadedExecutor()
    executor.add_node(dual_rin)

    try:
        # Run the executor to process callbacks
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if dual_rin.function_ is not None:
            dual_rin.sync_thread.join()
        destroy_nodes(dual_rin.arm_0_, dual_rin.arm_1_, dual_rin)
        rclpy.shutdown()
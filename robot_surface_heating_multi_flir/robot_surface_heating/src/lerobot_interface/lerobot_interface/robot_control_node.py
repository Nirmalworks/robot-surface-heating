import rclpy
from rclpy.node import Node, Publisher
from rclpy.action import ActionClient
from control_msgs.action import FollowJointTrajectory 
from control_msgs.msg import JointTolerance 

#### Control Dependencies ####
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from shape_msgs.msg import SolidPrimitive
from moveit_msgs.msg import (
    MoveItErrorCodes, RobotState,
    MotionPlanRequest, PlanningOptions,
    MotionSequenceItem, Constraints,
    PositionConstraint, OrientationConstraint,
    BoundingVolume, MotionSequenceRequest,
    MotionSequenceResponse, RobotTrajectory
)
from moveit_msgs.action import (
    MoveGroupSequence,
    ExecuteTrajectory
)
from moveit_msgs.srv import (
    GetPositionIK,
    GetPositionFK,
    GetMotionSequence,
    GetCartesianPath
)
######## ########

#### Gripper Dependencies #### 
from control_msgs.action import GripperCommand
from control_msgs.msg import GripperCommand as Gc_Msg
######## ########

#### General ROS Dependencies ####
from geometry_msgs.msg import Pose, Quaternion
######## ########

#### LeRobot dependencies ####
import json
import logging
import time
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

import sys
######## ########

#### Camera dependencies ####
from cv_bridge import CvBridge
import cv2
from sensor_msgs.msg import Image, CameraInfo
######## ########

#### wait_for_message() dependencies ####
from typing import Union
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec
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

class ArmConfig:
    """
    Configuration details for arm operation.
    """
    def __init__(self, arm_ns, param_list):
        if len(param_list) == 7:
            self.namespace = arm_ns
            self.arm_joint_state_topic_ = param_list[0]
            self.arm_joint_state_names_ = param_list[1]
            self.gripper_joint_state_topic_ = param_list[2]
            self.gripper_joint_state_names_ = param_list[3]
            self.gripper_service_ = param_list[4]
            self.traj_control_topic_ = param_list[5]
            self.traj_follow_topic_ = param_list[6]
            self.arm_base_frame_ = ''
            self.arm_eef_frame_ = ''
            self.arm_eef_link_ = ''
        elif len(param_list) == 10:
            self.namespace = arm_ns
            self.arm_joint_state_topic_ = param_list[0]
            self.arm_joint_state_names_ = param_list[1]
            self.arm_base_frame_ = param_list[2]
            self.arm_eef_frame_ = param_list[3]
            self.arm_eef_link_ = param_list[4]
            self.gripper_joint_state_topic_ = param_list[5]
            self.gripper_joint_state_names_ = param_list[6]
            self.gripper_service_ = param_list[7]
            self.traj_control_topic_ = param_list[8]
            self.traj_follow_topic_ = param_list[9]
        else:
            raise ValueError("Invalid quantity of parameters for ArmConfig constructor.")

    def __repr__(self):
        return f"\n{self.namespace}/\n\tArm Joint State Topic: {self.arm_joint_state_topic_}\
\n\tArm Joint Names: {self.arm_joint_state_names_}\n\tGripper Joint State Topic: {self.gripper_joint_state_topic_}\
\n\tGripper Joint Names: {self.gripper_joint_state_names_}\n\tGripper Service Topic: {self.gripper_service_}\
\n\tJoint Trajectory Controller Topic: {self.traj_control_topic_}\n\tFollow Joint Trajectory Topic: {self.traj_follow_topic_}\
\n\tBase Frame: {self.arm_base_frame_}\n\tEEF Frame: {self.arm_eef_frame_}\n\tEEF Link: {self.arm_eef_link_}"

class CameraConfig:
    """
    Configuration details for camera operation.
    """ 
    def __init__(self, cam_ns, param_list):
        self.namespace = cam_ns
        self.color_topic_ = param_list[0]
    
    def __repr__(self):
        return f"\n{self.namespace}/\n\tCamera Color Topic: {self.color_topic_}"

class RobotControlNode(Node):
    """
    ROS2 node interfacing with select LeRobot control_robot functions.
    """

    def __init__(self):
        """
        This constructor is intended for use with variants of the LeRobot control_robot script in
        the LeRobot package, which involves a local Hydra config file that contains the parameters
        stored in lerobot_interface_params config file found locally here.

        This constructor variant involves a setup where the UR5 and UR10e are simultaneously
        the leaders and followers.
        """
        super().__init__('robot_control_node')

        self.cartesian_path_client = self.create_client(GetCartesianPath, '/compute_cartesian_path')

        self.joint_state_topic = "/joint_states"
        self.base_frame_ = "base_link"
        self.eef_frame_ = "tool0"

        # fk and ik clients
        self.ik_client_ = self.create_client(GetPositionIK, "compute_ik")
        if not self.ik_client_.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("IK service not available.")
            exit(1)
        self.fk_client_ = self.create_client(GetPositionFK, "compute_fk")
        if not self.fk_client_.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("FK service not available.")
            exit(1)

        # planning and trajectory execution clients
        self._plan_action_client = ActionClient(self, MoveGroupSequence, '/sequence_move_group')
        self._plan_service_client = self.create_client(GetMotionSequence, '/plan_sequence_path')
        if not self._plan_service_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("MotionSequence plan service not available.")
            exit(1)
        self._execute_action_client = ActionClient(self, ExecuteTrajectory, '/execute_trajectory')

        self.traj_publisher = self.create_publisher(JointTrajectory, '/scaled_joint_trajectory_controller/joint_trajectory', 10)
        self.traj_client_ = ActionClient(self, FollowJointTrajectory, '/scaled_joint_trajectory_controller/follow_joint_trajectory')

        # initialize joint state sub
        self.joint_state_sub = self.create_subscription(JointState,self.joint_state_topic,self.joint_state_callback,10)
        
        self.get_logger().info("UR10e Control Node started!")
    ######## ########

    #### utility functions ####
    def joint_state_callback(self, msg: JointState):
        # self.get_logger().info('Received joint states:')
        # self.get_logger().info(f'Names: {msg.name}')
        # self.get_logger().info(f'Positions: {msg.position}')
        # self.get_logger().info(f'Velocities: {msg.velocity}')
        # self.get_logger().info(f'Efforts: {msg.effort}')
        self.arm_0_joint_states = msg

    def sort_arm_pos_in_name_order(self, name: str, arm_dict: dict[str, ArmConfig], joint_state: JointState) -> list[float]:
        """
        Helper function to ensure the joint states in the received message are in the
        correct order of the stored joint names in the corresponding ArmConfig container.
        """
        out_arr = []
        # print(f"presorted joints: {joint_state}")
        for joint_name in arm_dict[name].arm_joint_state_names_:
            try:
                out_arr.append(joint_state.position[joint_state.name.index(joint_name)])
            except ValueError:
                self.get_logger().info(f"faulty joint state received: {joint_state} for arm {name}")
                exit(1)
        return out_arr
    
    def read_arm_joints(self, name: str, arm_dict: dict[str, ArmConfig]) -> dict[str, list[float]]:
        """
        Helper function that gets the current joint states of a given arm, indicated by
        name and leader/follower designation. Joint states are returned in a dictionary
        for compatibility with the format used by the primary methods this class.
        """
        output_dict = {}

        # get arm joint states
        while True:
            current_joint_state_set, current_joint_state = wait_for_message(
                JointState, self, arm_dict[name].arm_joint_state_topic_, time_to_wait=3.0
            )
            if current_joint_state_set and len(current_joint_state.name) == len(arm_dict[name].arm_joint_state_names_):
                # need to acount for fake gripper joint state publishers causing disruptive traffic
                break
            else:
                self.get_logger().error(f"Capture Observation: Failed to get current joint state of arm {name}")
                return None
        output_dict[name] = self.sort_arm_pos_in_name_order(name, arm_dict, current_joint_state)

        # print(f"output dict pre gripper: {output_dict}")
        # print(f"\t\tarm joints collection time: {current_joint_state.header}")

        # get gripper joint state
        current_joint_state_set, current_joint_state = wait_for_message(
            JointState, self, arm_dict[name].gripper_joint_state_topic_, time_to_wait=3.0
        )
        if current_joint_state_set:
            output_dict[name] += [joint_val for joint_val in current_joint_state.position]
        else:
            self.get_logger().error(f"Capture Observation: Failed to get current joint state of gripper for arm {name}")
            return None
    
        # print(f"output dict post gripper: {output_dict}")
        # print(f"\t\tgripper joints collection time: {current_joint_state.header}")

        return output_dict
    ######## ########

    #### ROS interfacing functions ####
    def get_fk(self) -> Pose | None:
        """Makes an action request to the forward kinematics client to obtain the current
        pose of the specified arm."""

        current_joint_state_set, current_joint_state = wait_for_message(
            JointState, self, self.joint_state_topic, time_to_wait=1.0
        )
        if not current_joint_state_set:
            self.get_logger().error("FK: Failed to get current joint state")
            return None

        current_robot_state = RobotState()
        current_robot_state.joint_state = current_joint_state

        request = GetPositionFK.Request()

        request.header.frame_id = self.base_frame_
        request.header.stamp = self.get_clock().now().to_msg()

        request.fk_link_names.append(self.eef_frame_)
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
    ######## ########

    def send_action_sequence(self, items: list[torch.Tensor], from_idx: int, lan: int) -> None:
        """
        Creates a trajectory of joint states from the list of actions, then sends it to the joint trajectory controllers
        of the arms for splining execution.

        This part attempts to implement trajectory replacement.
        """
        return self.send_action_sequence_new(items, from_idx, lan)

    ######## ########

    #### pose-based operations ####
    def joint_to_pose(self, actions: list[torch.Tensor]) -> list[torch.Tensor]:
        """
        Converts the given action in joint space to cartesian space using the FK client.
        """
        ret_actions = []
        attempts = 10

        for action in actions:
            ret_action = []
            action_idx = 0
            for arm in self.follower_arms:
                # reconstruct action joint values as a JointState object
                repr_joint_state = JointState()
                repr_joint_state.name = self.follower_arms[arm].arm_joint_state_names_
                repr_joint_state.position = action.tolist()[action_idx:action_idx+6]
                repr_joint_state.velocity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                repr_joint_state.effort = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

                # send the joint state to FK solver
                robot_state = RobotState()
                robot_state.joint_state = repr_joint_state

                request = GetPositionFK.Request()

                request.header.frame_id = self.follower_arms[arm].arm_base_frame_
                request.header.stamp = self.get_clock().now().to_msg()

                request.fk_link_names.append(self.follower_arms[arm].arm_eef_frame_)
                request.robot_state = robot_state

                for _ in range(attempts):
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
                    break
                
                # append converted pose to return tensor
                action_pose: Pose = response.pose_stamped[0].pose
                ret_action += [action_pose.position.x, action_pose.position.y, action_pose.position.z,
                        action_pose.orientation.x, action_pose.orientation.y, action_pose.orientation.z,
                        action_pose.orientation.w, action[action_idx+6]]
                action_idx += 7

            ret_actions.append(torch.from_numpy(np.array(ret_action)))
        return ret_actions

    def read_arm_joints_poses(self, name: str, arm_dict: dict[str, ArmConfig]) -> dict[str, list[float]]:
        """
        Helper function that gets the current joint states of a given arm, indicated by
        name and leader/follower designation, and also calculates the current pose of
        the arm's end effector.
        """
        output_dict = {}

        # get arm joint states
        while True:
            current_joint_state_set, current_joint_state = wait_for_message(
                JointState, self, arm_dict[name].arm_joint_state_topic_, time_to_wait=3.0
            )
            if current_joint_state_set and len(current_joint_state.name) == len(arm_dict[name].arm_joint_state_names_):
                # need to acount for fake gripper joint state publishers causing disruptive traffic
                break
            else:
                self.get_logger().error(f"Capture Observation: Failed to get current joint state of arm {name}")
                return None
        self.get_logger().info(f"\t\t\t\t\t|||||||||||||||||follower position read|||||||||||||||||")
        arm_pose = self.get_fk(name, arm_dict)
        output_dict[name] = [arm_pose.position.x, arm_pose.position.y, arm_pose.position.z, arm_pose.orientation.x,
                            arm_pose.orientation.y, arm_pose.orientation.z, arm_pose.orientation.w]
        self.get_logger().info(f"\t\t\t\t\t|||||||||||||||||follower position translated to fk|||||||||||||||||")

        # print(f"output dict pre gripper: {output_dict}")

        # get gripper joint state
        current_joint_state_set, current_joint_state = wait_for_message(
            JointState, self, arm_dict[name].gripper_joint_state_topic_, time_to_wait=3.0
        )
        if current_joint_state_set:
            output_dict[name] += [joint_val for joint_val in current_joint_state.position]
        else:
            self.get_logger().error(f"Capture Observation: Failed to get current joint state of gripper for arm {name}")
            return None
    
        # print(f"output dict post gripper: {output_dict}")

        return output_dict

    def generate_orient_constraint(self, arm: str, x: float, y: float, z: float, w: float, margin: float, z_margin: float = None):
        con = [OrientationConstraint()]
        con[0].orientation = Quaternion()
        con[0].header.frame_id = self.follower_arms[arm].arm_base_frame_
        con[0].weight = 1.0
        con[0].link_name = self.follower_arms[arm].arm_eef_frame_
        con[0].orientation.x = x
        con[0].orientation.y = y
        con[0].orientation.z = z
        con[0].orientation.w = w
        con[0].absolute_x_axis_tolerance = margin
        con[0].absolute_y_axis_tolerance = margin
        con[0].absolute_z_axis_tolerance = z_margin if z_margin else margin

        return con

    def generate_pos_constraint(self, arm: str, xyz: list, xyz_margin: float, ort: list[float], orientation_margin: float) -> Constraints:
        con = Constraints()
        con.position_constraints = [PositionConstraint()]
        con.position_constraints[0].header.frame_id = self.follower_arms[arm].arm_base_frame_
        con.position_constraints[0].link_name = self.follower_arms[arm].arm_eef_frame_
        con.position_constraints[0].weight = 1.0

        con.position_constraints[0].constraint_region = BoundingVolume()
        con.position_constraints[0].constraint_region.primitives = [SolidPrimitive()]
        con.position_constraints[0].constraint_region.primitive_poses = [Pose()]

        con.position_constraints[0].constraint_region.primitives[0].type = SolidPrimitive.SPHERE
        con.position_constraints[0].constraint_region.primitives[0].dimensions = [xyz_margin]

        con.position_constraints[0].constraint_region.primitive_poses[0].position.x = float(xyz[0])
        con.position_constraints[0].constraint_region.primitive_poses[0].position.y = float(xyz[1])
        con.position_constraints[0].constraint_region.primitive_poses[0].position.z = float(xyz[2])

        con.position_constraints[0].constraint_region.primitive_poses[0].orientation.x = float(ort[0])
        con.position_constraints[0].constraint_region.primitive_poses[0].orientation.y = float(ort[1])
        con.position_constraints[0].constraint_region.primitive_poses[0].orientation.z = float(ort[2])
        con.position_constraints[0].constraint_region.primitive_poses[0].orientation.w = float(ort[3])

        con.orientation_constraints = self.generate_orient_constraint(arm, float(ort[0]), float(ort[1]), float(ort[2]), float(ort[3]), orientation_margin)

        return con

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

        # get initial gripper joint states
        grippers_pos: dict[str, list[float]] = {}
        for arm in self.follower_arms:
            grippers_pos[arm] = [self.read_arm_joints(arm, self.follower_arms)[arm][-1]]

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

                # # print(f"goal index issue: {req_item.req.goal_constraints}")
                # req_item.req.goal_constraints[0] = self.generate_pos_constraint(arm, xyz, 0.1, items[item_idx]["action"][arm_off+3:arm_off+7], 0.001)
                # # temp = items[item_idx]["action"][arm_off+7]
                # # print(f"action array: {temp}")
                # # print(grippers_pos)
                grippers_pos[arm].append(float(items[item_idx]["action"][arm_off+7]))

                # move to pose and gripper joint data for next arm
                arm_off += 8

        from std_msgs.msg import Header
        request = GetCartesianPath.Request(
            header=Header(frame_id=self.follower_arms["ur5"].arm_base_frame_, stamp=self.get_clock().now().to_msg()),
            max_step=0.01,
            jump_threshold = 0.0,
            group_name="ur5",
            waypoints=waypoint_list
        )

        plan_future = self.cartesian_path_client.call_async(request)
        rclpy.spin_until_future_complete(self, plan_future)
        response = plan_future.result()

        return response.solution.joint_trajectory, grippers_pos

    def send_action_sequence_new(self, items: list[torch.Tensor], from_idx: int, lan: int) -> None:
        print("here")
        cartesian_path, grippers_pos = self.get_cartesian_trajectory(items, from_idx, lan)
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = cartesian_path
        self.traj_clients_["ur5"].wait_for_server()
        goal_future = self.traj_clients_["ur5"].send_goal_async(goal)
        for arm in grippers_pos:
            # only use final provided action for gripper actuation
            self.actuate_gripper(arm, float(grippers_pos[arm][-1]))
            time.sleep(0.005)   # 5ms delay between actuations
        rclpy.spin_until_future_complete(self, goal_future)
        rclpy.spin_until_future_complete(self, goal_future)

    ######## ########

def main():
    rclpy.init()
    node = RobotControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    node.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
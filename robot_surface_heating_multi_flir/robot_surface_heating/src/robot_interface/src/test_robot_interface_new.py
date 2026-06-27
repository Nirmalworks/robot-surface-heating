#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import numpy as np
from typing import Optional
from geometry_msgs.msg import Pose, Point, Quaternion, PointStamped, PoseStamped
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from thermal_camera_interfaces.msg import Extrema
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor

from moveit_msgs.msg import (
    RobotState,
    RobotTrajectory,
    MoveItErrorCodes,
    Constraints,
    JointConstraint,
    PlanningScene,
    CollisionObject
)
from moveit_msgs.srv import GetPositionIK, GetMotionPlan, GetPositionFK, GetPlanningScene, ApplyPlanningScene
from moveit_msgs.action import ExecuteTrajectory

from typing import Union

from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.qos import QoSProfile
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec
from scipy.spatial.transform import Rotation as R

import tf2_ros
import tf2_geometry_msgs
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup

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

def check_same_pose(pose_1: Pose, pose_2: Pose, pos_threshold=0.001, angle_threshold=np.deg2rad(0.1)) -> bool:
    # Position difference
    pos_diff = np.array([
        pose_1.position.x - pose_2.position.x,
        pose_1.position.y - pose_2.position.y,
        pose_1.position.z - pose_2.position.z
    ])
    pos_dist = np.linalg.norm(pos_diff)

    # Quaternion difference using rotation matrix
    q1 = [pose_1.orientation.x, pose_1.orientation.y, pose_1.orientation.z, pose_1.orientation.w]
    q2 = [pose_2.orientation.x, pose_2.orientation.y, pose_2.orientation.z, pose_2.orientation.w]

    # Convert to rotation matrix and compute angular difference
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    relative_rotation = r1.inv() * r2
    angle_diff = relative_rotation.magnitude()  # Angular distance

    # Print differences for debugging
    print(f"Position Difference: {pos_dist}, Orientation Angle Difference: {np.rad2deg(angle_diff)} degrees")

    # Return True only if both position and orientation thresholds are satisfied
    return pos_dist < pos_threshold and angle_diff < angle_threshold

class RobotInterfaceNode(Node):
    timeout_sec_ = 5.0

    move_group_name_ = "ur10e"
    namespace_ = ""

    joint_state_topic_ = "joint_states"
    plan_srv_name_ = "plan_kinematic_path"
    ik_srv_name_ = "compute_ik"
    fk_srv_name_ = "compute_fk"
    coldest_point_topic_ = "coldest_point"
    execute_action_name_ = "execute_trajectory"
    get_planning_scene_srv_name = "get_planning_scene"
    apply_planning_scene_srv_name = "apply_planning_scene"
    coldest_pose_topic_ = "coldest_pose"

    base_ = "base_link"
    end_effector_ = "tool0"

    def __init__(self, executor: SingleThreadedExecutor) -> None:
        super().__init__("robot_interface_node")
        self.get_logger().info("Starting RobotInterfaceNode constructor...")
        # self.z_offset = 0.05
        self.z_offset = 0.35
        self.task_in_progress = False
        # self.query_lock = threading.Lock()
        self.latest_coldest_point = None
        
        self.home_pose = Pose(
            position=Point(x=0.73945, y=-0.29125, z=0.7),
            orientation=Quaternion(x=0.96314, y=-0.26109, z=-0.00405, w=0.06467),
        )
        self.desired_temperature = 100.0 # Celsius

        # self.coldest_point_subscriber = self.create_subscription(
        #     PointStamped,
        #     "coldest_point",
        #     self.coldest_point_callback,
        #     10
        # )
        self.ctrl_loop_exec = executor
        # self.coldest_pose_cb_group = ReentrantCallbackGroup()
        self.coldest_pose_cb_group = MutuallyExclusiveCallbackGroup()
        # self.control_loop_cb_group = ReentrantCallbackGroup()
        self.control_loop_cb_group = MutuallyExclusiveCallbackGroup()
        self.coldest_pose_subscriber = self.create_subscription(
            # PoseStamped,
            Extrema,
            "coldest_pose",
            # "hottest_pose",
            self.coldest_pose_callback,
            10,
            callback_group=self.coldest_pose_cb_group
        )
        self.debug_pose_pub = self.create_publisher(PoseStamped, '/robot_target_pose', 10)
        self.ik_client_ = self.create_client(GetPositionIK, self.ik_srv_name_, callback_group=self.control_loop_cb_group)
        if not self.ik_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("IK service not available.")
            exit(1)

        self.fk_client_ = self.create_client(GetPositionFK, self.fk_srv_name_, callback_group=self.control_loop_cb_group)
        if not self.fk_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("FK service not available.")
            exit(1)

        self.plan_client_ = self.create_client(GetMotionPlan, self.plan_srv_name_, callback_group=self.control_loop_cb_group)
        if not self.plan_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("Plan service not available.")
            exit(1)

        self.execute_client_ = ActionClient(
            self, ExecuteTrajectory, self.execute_action_name_, callback_group=self.control_loop_cb_group
        )
        if not self.execute_client_.wait_for_server(timeout_sec=self.timeout_sec_):
            self.get_logger().error("Execute action not available.")
            exit(1)

        self.get_logger().info("Action/service clients ready.")

        # self.get_planning_scene_client = self.create_client(GetPlanningScene, self.get_planning_scene_srv_name)
        # if not self.get_planning_scene_client.wait_for_service(timeout_sec=self.timeout_sec_):
        #     self.get_logger().error("Get planning scene service not available.")
        #     exit(1)

        # self.apply_planning_scene_client = self.create_client(ApplyPlanningScene, self.apply_planning_scene_srv_name)
        # if not self.apply_planning_scene_client.wait_for_service(timeout_sec=self.timeout_sec_):
        #     self.get_logger().error("Apply planning scene service not available.")
        #     exit(1)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(
            5, 
            self.heating_control_loop, 
            callback_group=self.control_loop_cb_group
        )

        # self.motion_done_event = threading.Event()
        # self.task_lock = threading.Lock()

    def coldest_pose_callback(self, msg: Extrema):
        """Store latest coldest point from topic"""
        # if not self.context.ok():
        #     return
        # with self.query_lock:
        if self.task_in_progress:
            self.get_logger().info("Skipping callback: task already in progress.")
            return
        if self.latest_coldest_point:
            prev = self.latest_coldest_point.pose.position
            new = msg.pose.position
            if np.linalg.norm([new.x - prev.x, new.y - prev.y, new.z - prev.z]) < 0.01:
                print("POSES OF COLD POINTS SAME")
                return
        self.latest_coldest_point = msg
        self.get_logger().info(f"Received coldest position: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}, z={msg.pose.position.z:.3f}, temp={msg.value:.3f}C")
        # print("callback called")
        # self.heating_control_loop()

    def query_coldest_pose(self) -> Optional[Extrema]:
        """Get the latest coldest pose"""
        # with self.query_lock:
        return self.latest_coldest_point

    def get_tool_offset_pose(self, original_pose: Extrema) -> Pose:
        """
        Gets the robot pose offset along the inverted surface normal,
        flipping the tool 180 deg around X and offsetting along the tool Z.
        """
        quat = [
            original_pose.pose.orientation.x,
            original_pose.pose.orientation.y,
            original_pose.pose.orientation.z,
            original_pose.pose.orientation.w
        ]

        rot_orig = R.from_quat(quat)
        flip_x = R.from_euler('x', 180, degrees=True)
        rot_flipped = rot_orig * flip_x

        # Compute offset vector along inverted surface normal (tool's new Z axis)
        offset_vec = -self.z_offset * rot_flipped.apply([0, 0, 1])

        offset_position = np.array([
            original_pose.pose.position.x,
            original_pose.pose.position.y,
            original_pose.pose.position.z
        ]) + offset_vec

        new_quat = rot_flipped.as_quat()

        self.get_logger().info(f"Original position: {original_pose.pose.position}")
        self.get_logger().info(f"Offset vec: {offset_vec}")
        self.get_logger().info(f"New position: {offset_position}")
        self.get_logger().info(f"New orientation: {new_quat}")

        target_pose = PoseStamped()
        target_pose.header = original_pose.header
        target_pose.pose.position = Point(
            x = offset_position[0], y = offset_position[1], z = offset_position[2]
        )
        target_pose.pose.orientation = Quaternion(
            x = new_quat[0], y = new_quat[1], z = new_quat[2], w = new_quat[3],
        )

        # transform CAD local frame pose to world frame pose
        for _ in range(10):
            try:
                target_pose = self.tf_buffer.transform(target_pose, "world", timeout=rclpy.duration.Duration(seconds=0.1))
                return target_pose.pose
            except Exception as e:
                self.get_logger().warn(f"TF transform failed: {e}")
                continue

        return None

    def heating_control_loop(self):
        """Main control loop - queries and executes heating"""        
        # if self.task_in_progress:
        #     self.get_logger().info("Task already in progress, dropping new request.")
        #     return

        self.get_logger().info("querying coldest point")
        coldest_pose = self.query_coldest_pose()
        if coldest_pose is None:
            print("DID NOT FIND A COLDESTPOINT")
            print("DID NOT FIND A COLDESTPOINT")
            print("DID NOT FIND A COLDESTPOINT")
            return
        
        self.get_logger().info("got coldest point")

        self.task_in_progress = True
        try:
            # return to home pose if coldest point is above threshold
            # if coldest_pose.value > self.desired_temperature:
            #     target_pose = self.home_pose
            # else:
            target_pose = self.get_tool_offset_pose(coldest_pose)
            if target_pose is None:
                print("DID NOT FIND A TARGET")
                print("DID NOT FIND A TARGET")
                print("DID NOT FIND A TARGET")
                return

            self.get_logger().info(f"got coldest point: x={coldest_pose.pose.position.x:.3f}, y={coldest_pose.pose.position.y:.3f}, z={coldest_pose.pose.position.z:.3f}, temp={coldest_pose.value:.3f}C")
            self.get_logger().info(f"\tcoldest point ort: {coldest_pose.pose.orientation}")

            self.get_logger().info(f"Moving to heat point at {target_pose.position.x:.3f}, {target_pose.position.y:.3f}, {target_pose.position.z:.3f}")
            self.get_logger().info(f"\ttarget ort: {target_pose.orientation}")

            debug_pose = PoseStamped()
            # debug_pose.header.frame_id = 'world'
            # debug_pose.header.frame_id = self.end_effector_
            debug_pose.header.frame_id = coldest_pose.header.frame_id
            debug_pose.header.stamp = self.get_clock().now().to_msg()
            debug_pose.pose = target_pose
            self.debug_pose_pub.publish(debug_pose)

            self.execute_trajectory(target_pose)
        finally:
            self.task_in_progress = False
        print(f"task in progress value {self.task_in_progress}")

    # def get_ik(self, target_pose: Pose) -> JointState | None:
    #     request = GetPositionIK.Request()
    #     request.ik_request.group_name = self.move_group_name_
    #     request.ik_request.pose_stamped.header.frame_id = self.base_
    #     request.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
    #     request.ik_request.pose_stamped.pose = target_pose
    #     request.ik_request.avoid_collisions = True

    #     future = self.ik_client_.call_async(request)

    #     while rclpy.ok() and not future.done():
    #         time.sleep(0.01)

    #     if future.result() is None:
    #         self.get_logger().error("Failed to get IK solution")
    #         return None

    #     response = future.result()
    #     if response.error_code.val != MoveItErrorCodes.SUCCESS:
    #         self.get_logger().warn(f"IK failed with error code: {response.error_code.val}")
    #         return None

    #     return response.solution.joint_state

    def get_ik(self, target_pose: Pose) -> JointState | None:
        request = GetPositionIK.Request()

        request.ik_request.group_name = self.move_group_name_
        request.ik_request.pose_stamped.header.frame_id = f"{self.base_}"
        request.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        request.ik_request.pose_stamped.pose = target_pose
        request.ik_request.avoid_collisions = True

        future = self.ik_client_.call_async(request)

        rclpy.spin_until_future_complete(self, future, executor=self.ctrl_loop_exec)
        if future.result() is None:
            self.get_logger().error("Failed to get IK solution")
            return None

        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None

        return response.solution.joint_state

    # def get_fk(self) -> Pose | None:
    #     for _ in range(10):
    #         current_joint_state = self.get_joint_state()
    #         if current_joint_state is not None:
    #             break
    #     if current_joint_state is None:
    #         self.get_logger().error("Failed to get current joint state")
    #         return None

    #     current_robot_state = RobotState()
    #     current_robot_state.joint_state = current_joint_state

    #     request = GetPositionFK.Request()
    #     request.header.frame_id = self.base_
    #     request.header.stamp = self.get_clock().now().to_msg()
    #     request.fk_link_names.append(self.end_effector_)
    #     request.robot_state = current_robot_state

    #     future = self.fk_client_.call_async(request)
    #     while rclpy.ok() and not future.done():
    #         time.sleep(0.01)

    #     if future.result() is None:
    #         self.get_logger().error("FK service returned no result")
    #         return None

    #     response = future.result()
    #     if response.error_code.val != MoveItErrorCodes.SUCCESS:
    #         self.get_logger().error(f"FK failed with error code: {response.error_code.val}")
    #         return None

    #     return response.pose_stamped[0].pose

    def get_fk(self) -> Pose | None:
        self.get_logger().info("start of fk")
        for _ in range(10):
            current_joint_state = self.get_joint_state()
            if current_joint_state is not None:
                break
        if current_joint_state is None:
            self.get_logger().error("Failed to get current joint state")
            return None
        self.get_logger().info("got joint state in fk")

        current_robot_state = RobotState()
        current_robot_state.joint_state = current_joint_state

        request = GetPositionFK.Request()

        request.header.frame_id = f"{self.base_}"
        request.header.stamp = self.get_clock().now().to_msg()

        request.fk_link_names.append(self.end_effector_)
        request.robot_state = current_robot_state

        future = self.fk_client_.call_async(request)
        self.get_logger().info("calling for fk")

        rclpy.spin_until_future_complete(self, future, executor=self.ctrl_loop_exec)
        if future.result() is None:
            self.get_logger().error("Failed to get FK solution")
            return None
        
        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"Failed to get FK solution: {response.error_code.val}"
            )
            return None
        self.get_logger().info("got fk")
        return response.pose_stamped[0].pose

    def sum_of_square_diff(
        self, joint_state_1: JointState, joint_state_2: JointState
    ) -> float:
        return np.sum(
            np.square(np.subtract(joint_state_1.position, joint_state_2.position))
        )

    def get_best_ik(self, target_pose: Pose, attempts: int = 10) -> JointState | None:
        for _ in range(attempts):
            current_joint_state = self.get_joint_state()
            if current_joint_state is not None:
                break
        if current_joint_state is None:
            self.get_logger().error("Failed to get current joint state")
            return None

        best_cost = np.inf
        best_joint_state = None

        for _ in range(attempts):
            joint_state = self.get_ik(target_pose)
            if joint_state is None:
                continue

            cost = self.sum_of_square_diff(current_joint_state, joint_state)
            if cost < best_cost:
                best_cost = cost
                best_joint_state = joint_state

        if not best_joint_state:
            self.get_logger().error("Failed to get IK solution")

        return best_joint_state

    def get_joint_state(self) -> JointState:
        current_joint_state_set, current_joint_state = wait_for_message(
            JointState, self, self.joint_state_topic_, time_to_wait=1.0
        )
        if not current_joint_state_set:
            self.get_logger().error("Failed to get current joint state")
            return None
        
        return current_joint_state

    def get_motion_plan(
        self, target_pose: Pose, linear: bool = True, attempts: int = 10
    ) -> RobotTrajectory | None:
        self.get_logger().info("start of get motion plan")

        current_pose = self.get_fk()
        if current_pose is None:
            self.get_logger().error("Failed to get current pose")
        self.get_logger().info("got fk")

        # if check_same_pose(current_pose, target_pose):
        #     self.get_logger().warn("Same pose")
        #     # return RobotTrajectory()
        #     return None

        current_joint_state_set, current_joint_state = wait_for_message(
            JointState, self, self.joint_state_topic_, time_to_wait=1.0
        )
        if not current_joint_state_set:
            self.get_logger().error("Failed to get current joint state")
            return None

        current_robot_state = RobotState()
        current_robot_state.joint_state.position = current_joint_state.position
        self.get_logger().info("made robot state")

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

        self.get_logger().info("got constraints")

        request = GetMotionPlan.Request()
        request.motion_plan_request.group_name = self.move_group_name_
        request.motion_plan_request.start_state = current_robot_state
        request.motion_plan_request.goal_constraints.append(target_constraint)
        request.motion_plan_request.num_planning_attempts = 10
        request.motion_plan_request.allowed_planning_time = 5.0
        request.motion_plan_request.max_velocity_scaling_factor = 1.0
        request.motion_plan_request.max_acceleration_scaling_factor = 0.2

        self.get_logger().info("finished request")

        if linear:
            request.motion_plan_request.pipeline_id = "pilz_industrial_motion_planner"
            request.motion_plan_request.planner_id = "LIN"
        else:
            request.motion_plan_request.pipeline_id = "ompl"
            request.motion_plan_request.planner_id = "BFMTkConfigDefault"

        for _ in range(attempts):
            plan_future = self.plan_client_.call_async(request)
            rclpy.spin_until_future_complete(self, plan_future, executor=self.ctrl_loop_exec)

            if plan_future.result() is None:
                self.get_logger().error("Failed to get motion plan")

            response = plan_future.result()
            if response.motion_plan_response.error_code.val != MoveItErrorCodes.SUCCESS:
                self.get_logger().error(
                    f"Failed to get motion plan: {response.motion_plan_response.error_code.val}"
                )
            else:
                return response.motion_plan_response.trajectory
            
        return None

    # def get_motion_plan(
    #     self, target_pose: Pose, linear: bool = True, attempts: int = 100
    # ) -> RobotTrajectory | None:
    #     current_pose = self.get_fk()
    #     if current_pose is None:
    #         self.get_logger().error("Failed to get current pose")
    #         return None

    #     if check_same_pose(current_pose, target_pose):
    #         self.get_logger().warn("Target pose is the same as current pose")
    #         return None

    #     for _ in range(10):
    #         current_joint_state = self.get_joint_state()
    #         if current_joint_state is not None:
    #             break
    #     if current_joint_state is None:
    #         self.get_logger().error("Failed to get current joint state")
    #         return None
    #     print("got current joint state")
        
    #     current_robot_state = RobotState()
    #     current_robot_state.joint_state = current_joint_state

    #     target_joint_state = self.get_best_ik(target_pose, attempts=1)
    #     if target_joint_state is None:
    #         self.get_logger().error("Failed to get target joint state")
    #         return None
    #     print("got target joint state")

    #     target_constraint = Constraints()
    #     for i in range(len(target_joint_state.position)):
    #         joint_constraint = JointConstraint()
    #         joint_constraint.joint_name = target_joint_state.name[i]
    #         joint_constraint.position = target_joint_state.position[i]
    #         joint_constraint.tolerance_above = 0.001
    #         joint_constraint.tolerance_below = 0.001
    #         joint_constraint.weight = 1.0
    #         target_constraint.joint_constraints.append(joint_constraint)

    #     request = GetMotionPlan.Request()
    #     request.motion_plan_request.group_name = self.move_group_name_
    #     request.motion_plan_request.start_state = current_robot_state
    #     request.motion_plan_request.goal_constraints.append(target_constraint)
    #     request.motion_plan_request.num_planning_attempts = 10
    #     request.motion_plan_request.allowed_planning_time = 5.0
    #     request.motion_plan_request.max_velocity_scaling_factor = 1.0
    #     request.motion_plan_request.max_acceleration_scaling_factor = 0.2

    #     if linear:
    #         request.motion_plan_request.pipeline_id = "pilz_industrial_motion_planner"
    #         request.motion_plan_request.planner_id = "LIN"
    #     else:
    #         request.motion_plan_request.pipeline_id = "ompl"
    #         request.motion_plan_request.planner_id = "BFMTkConfigDefault"

    #     for attempt in range(attempts):
    #         future = self.plan_client_.call_async(request)
    #         while rclpy.ok() and not future.done():
    #             time.sleep(0.01)

    #         result = future.result()
    #         if result is None:
    #             self.get_logger().warn("Planning service returned None")
    #             continue

    #         if result.motion_plan_response.error_code.val == MoveItErrorCodes.SUCCESS:
    #             return result.motion_plan_response.trajectory
    #         else:
    #             self.get_logger().warn(f"Planning failed: {result.motion_plan_response.error_code.val}")

    #     self.get_logger().error("Failed to get valid motion plan after retries")
    #     return None

    def get_motion_execute_client(self) -> ActionClient:
        return self.execute_client_

    def get_planning_scene(self) -> PlanningScene | None:
        request = GetPlanningScene.Request()
        request.components.components = 0

        future = self.get_planning_scene_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, executor=self.ctrl_loop_exec)

        if future.result() is None:
            self.get_logger().error("Failed to get planning scene")

        response = future.result()
        return response.scene

    def update_obstacle(self, obstacle_id: str, obstacle_pose: Pose, obstacle_size: np.ndarray, obstacle_type: int, remove: bool = False) -> bool:
        current_scene = self.get_planning_scene()
        if current_scene is None:
            self.get_logger().error("Failed to get current planning scene")
            return False

        current_collision_objects = current_scene.world.collision_objects

        if remove:
            for collision_object in current_collision_objects:
                if collision_object.id == obstacle_id:
                    collision_object.operation = CollisionObject.REMOVE
                    break
        else:
            collision_object = CollisionObject()
            collision_object.id = obstacle_id
            collision_object.header.frame_id = "/" + self.base_
            collision_object.primitives.append(SolidPrimitive(type=obstacle_type, dimensions=obstacle_size))
            collision_object.primitive_poses.append(obstacle_pose)
            collision_object.operation = CollisionObject.ADD

            current_collision_objects.append(collision_object)

        apply_scene_request = ApplyPlanningScene.Request()
        apply_scene_request.scene = current_scene

        future = self.apply_planning_scene_client.call_async(apply_scene_request)
        rclpy.spin_until_future_complete(self, future, executor=self.ctrl_loop_exec)

        if future.result() is None:
            self.get_logger().error("Failed to update planning scene")
            return False
        
        return future.result().success

    def execute_trajectory(self, target_pose: Pose) -> None:
        print("getting motion plan")
        traj = self.get_motion_plan(target_pose)
        print("successfully got motion plan")
        # self.get_logger().info(f"traj {traj}")
        time.sleep(3)
        traj = None
        if traj:
            client = self.get_motion_execute_client()
            goal = ExecuteTrajectory.Goal()
            goal.trajectory = traj

            future = client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future, executor=self.ctrl_loop_exec)
            
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("Failed to execute trajectory")
            else:
                self.get_logger().info("Trajectory accepted")

                result_future = goal_handle.get_result_async()

                # self.get_logger().info(f"Trajectory points {traj.joint_trajectory.points}")
                expect_duration = traj.joint_trajectory.points[-1].time_from_start
                expect_time = time.time() + 2 * expect_duration.sec
                while not result_future.done() and time.time() < expect_time:
                    time.sleep(0.01)

                self.get_logger().info("Trajectory executed")
                self.get_logger().info("Current pose: " + str(self.get_fk()))
                time.sleep(7)

    # def execute_trajectory(self, target_pose: Pose) -> None:
    #     self.get_logger().info("Planning motion...")
    #     traj = self.get_motion_plan(target_pose, attempts=5)
    #     if not traj:
    #         self.get_logger().warn("No trajectory to execute")
    #         return

    #     self.get_logger().info("Got trajectory. Sending goal...")
    #     goal = ExecuteTrajectory.Goal()
    #     goal.trajectory = traj

    #     future = self.execute_client_.send_goal_async(goal)
    #     while rclpy.ok() and not future.done():
    #         time.sleep(0.01)

    #     self.get_logger().info("Goal sent. Waiting for result handle...")
    #     goal_handle = future.result()
    #     if not goal_handle.accepted:
    #         self.get_logger().error("ExecuteTrajectory goal was not accepted")
    #         return

    #     self.get_logger().info("Trajectory goal accepted")
    #     result_future = goal_handle.get_result_async()

    #     # Wait for execution with timeout safety
    #     expected_duration = traj.joint_trajectory.points[-1].time_from_start
    #     timeout_time = time.time() + expected_duration.sec * 2.0

    #     while not result_future.done() and time.time() < timeout_time:
    #         time.sleep(0.01)

    #     if not result_future.done():
    #         self.get_logger().error("Execution timed out")
    #         return

    #     result = result_future.result()
    #     self.get_logger().info("Trajectory executed")

def main(args=None):
    rclpy.init(args=args)
    ctrl_loop_executor = SingleThreadedExecutor()
    robot_interface_node = RobotInterfaceNode(ctrl_loop_executor)
    robot_interface_node.get_logger().info("Robot interface node created, starting executor")

    # pickup = Pose(
    #     position=Point(x=0.014174701645970345, y=-0.31069329380989075, z=0.1215794323682785),
    #     orientation=Quaternion(x=0.6235913634300232, y=0.6527717113494873, z=-0.2971552908420563, w=0.31100109219551086),
    # )
    # point1 = Pose(
    #     position=Point(x=0.015751153230667114, y=-0.3453364074230194, z=0.6156614422798157),
    #     orientation=Quaternion(x=0.5871766209602356, y=0.614645779132843, z=-0.3638959527015686, w=0.38080689311027527),
    # )
    # point2 = Pose(
    #     position=Point(x=0.32712218165397644,y=0.06661370396614075, z=0.5955476760864258),
    #     orientation=Quaternion(x=-0.09255168586969376, y=0.9168617725372314, z=0.03891941159963608, w=0.38637280464172363),
    # )

    home_pose = Pose(
        # position=Point(x=0.32399776577949524, 
        #     y=0.06254103779792786, 
        #     z=0.4997061789035797),
        position=Point(x=0.73945,
            y=-0.29125, 
            z=0.7),
        orientation=Quaternion(x=0.96314,
            y=-0.26109,
            z=-0.00405,
            w=0.06467),
    )

    robot_interface_node.execute_trajectory(robot_interface_node.home_pose)
    robot_interface_node.task_in_progress = False
    robot_interface_node.get_logger().info("Finished rest to home pose.")

    # executor = MultiThreadedExecutor(num_threads=4)
    # executor.add_node(robot_interface_node)

    # time.sleep(10)
    # robot_interface_node.execute_trajectory(point1)
    # robot_interface_node.execute_trajectory(point2)
    # robot_interface_node.execute_trajectory(home_pose)

    # robot_interface_node.heating_control_loop()

    try:
        rclpy.spin(robot_interface_node)
        # executor.spin()
    except KeyboardInterrupt:
        pass
    robot_interface_node.destroy_node()
    rclpy.shutdown()

    # executor = MultiThreadedExecutor()
    # executor.add_node(robot_interface_node)
    
    # # Start the sync thread only after executor is spinning
    # sync_thread = threading.Thread(target=robot_interface_node.continuous_heating_loop, daemon=True)
    # sync_thread.start()

    # try:
    #     # Run the executor to process callbacks
    #     executor.spin()
    # except KeyboardInterrupt:
    #     pass
    # finally:
    #     # robot_interface_node.sync_thread.join()
    #     robot_interface_node.destroy_node()
    #     rclpy.shutdown()

# def main(args=None):
#     rclpy.init(args=args)
#     robot_interface_node = RobotInterfaceNode()
#     executor = MultiThreadedExecutor()
#     executor.add_node(robot_interface_node)

#     # Start executor thread
#     executor_thread = threading.Thread(target=executor.spin, daemon=True)
#     executor_thread.start()

#     # Move to home pose
#     robot_interface_node.get_logger().info("Moving to home pose...")
#     robot_interface_node.execute_trajectory(robot_interface_node.home_pose)
#     robot_interface_node.task_in_progress = False
#     robot_interface_node.get_logger().info("Reached home pose.")

#     # Start heating loop
#     stop_event = threading.Event()
#     heating_thread = threading.Thread(
#         target=robot_interface_node.continuous_heating_loop,
#         args=(stop_event,),
#         daemon=True
#     )
#     robot_interface_node.get_logger().info("Starting heating loop")
#     heating_thread.start()

#     try:
#         executor_thread.join()
#     except KeyboardInterrupt:
#         robot_interface_node.get_logger().info("Shutting down...")

#         # signal the heating thread to stop
#         stop_event.set()
#         heating_thread.join()

#         executor.shutdown()  # stop the executor
#     finally:
#         robot_interface_node.destroy_node()
#         rclpy.shutdown()

if __name__ == "__main__":
    main()

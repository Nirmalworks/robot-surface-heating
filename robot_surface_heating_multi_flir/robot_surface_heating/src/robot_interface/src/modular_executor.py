#!/usr/bin/env python3
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import numpy as np
import math
from typing import Optional, List
from builtin_interfaces.msg import Duration, Time
from std_msgs.msg import Int32, Float64, Header
from geometry_msgs.msg import Pose, Point, Quaternion, PointStamped, PoseStamped, PoseArray
from sensor_msgs.msg import JointState
from shape_msgs.msg import SolidPrimitive
from visualization_msgs.msg import Marker, MarkerArray
from thermal_camera_interfaces.msg import Extrema

from moveit_msgs.msg import (
    RobotState,
    RobotTrajectory,
    MoveItErrorCodes,
    Constraints,
    JointConstraint,
    PlanningScene,
    CollisionObject
)
from moveit_msgs.srv import GetPositionIK, GetMotionPlan, GetPositionFK, GetPlanningScene, ApplyPlanningScene, GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory

from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from action_msgs.msg import GoalStatusArray

from typing import Union

from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec
from scipy.spatial.transform import Rotation as R, Slerp

import tf2_ros
import tf2_geometry_msgs
import tf_transformations
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from copy import deepcopy
import threading
import queue
import time

import sys
from thermal_camera import common_motionplan_utilities

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

def compute_path_length(poses: list[Pose]) -> float:
    """Compute total linear path length of a list of poses."""
    total_length = 0.0
    for i in range(len(poses) - 1):
        p1 = np.array([poses[i].position.x, poses[i].position.y, poses[i].position.z])
        p2 = np.array([poses[i + 1].position.x, poses[i + 1].position.y, poses[i + 1].position.z])
        total_length += np.linalg.norm(p2 - p1)
    return total_length


def time_to_float(time: Time,) -> float:
    whole = float(time.sec)
    dec = float(time.nanosec) * 1e-9
    return whole + dec 

def time_plus_float(time_obj: Time, float_value: Time) -> Time:
    """Increase or decrease the timestamp of a time object by a specified float amount"""

    int_part = math.floor(float_value)
    dec_part = float_value - int_part
    
    s = time_obj.sec + int_part
    ns = time_obj.nanosec + int(round(dec_part * 1e9))

    if ns >= 1e9:
        s += 1
        ns -= int(1e9)
    elif ns < 0:
        s -= 1
        ns += int(1e9)
    
    result = Time()
    result.sec = s
    result.nanosec = ns
    return result

def duration_to_float(duration_obj: Duration) -> float:
    int_part = float(duration_obj.sec)
    dec_part = float(duration_obj.nanosec) * 1e-9

    return int_part + dec_part


def retime_trajectory(
    traj: JointTrajectory,
    base_speed: float = 0.05,
    *,
    dt_min: float = 0.002,          # minimum spacing between consecutive timestamps [s]
    eps: float = 1e-9,              # tiny number to avoid divides by ~0
    speed_max: float = 0.65,         # cap like your original
    zero_final: bool = True,        # if False, keep computed terminal v/a (better for blending)
    computation_time: float = 0.0,  # seconds to trim from the start
    startup_pad: float = 0.3        # simple buffer so first kept point isn't due "now"
) -> None:
    """
    Assigns time_from_start based on segment length and a speed knob, then
    trims away points with time_from_start < computation_time, re-zeros the
    remaining times, and finally shifts the whole timeline by a small
    startup_pad to avoid a sprint to the first point. Velocities and
    accelerations are computed on the final timeline.

    Modifies traj in place. Returns None.
    """
    # ----------------- early exits -----------------
    if traj is None or not traj.points:
        return
    if len(traj.points) == 1:
        traj.points[0].time_from_start.sec = 0
        traj.points[0].time_from_start.nanosec = 0
        if zero_final:
            n = len(traj.points[0].positions) if traj.points[0].positions else 0
            traj.points[0].velocities = [0.0] * n
            traj.points[0].accelerations = [0.0] * n
        if hasattr(traj, "header") and hasattr(traj.header, "stamp"):
            traj.header.stamp = Time(sec=0, nanosec=0)
        return

    pts = traj.points

    # ----------------- helpers -----------------
    def _sec_to_dur(sec_f: float) -> Duration:
        # robust float->Duration conversion with normalization
        if sec_f < 0.0:
            s = math.floor(sec_f)
            ns = int((sec_f - s) * 1e9)
        else:
            s = int(math.floor(sec_f))
            ns = int(round((sec_f - s) * 1e9))
            if ns >= 1_000_000_000:
                s += 1
                ns -= 1_000_000_000
        return Duration(sec=s, nanosec=ns)

    # ----------------- validate & collect positions -----------------
    positions = []
    n_joints = None
    for p in pts:
        if not p.positions:
            return
        if n_joints is None:
            n_joints = len(p.positions)
        elif len(p.positions) != n_joints:
            return
        positions.append(np.asarray(p.positions, dtype=float))
    q = np.stack(positions, axis=0)  # (N, J)
    N = q.shape[0]

    # ----------------- base timing from path length -----------------
    seg_dist = np.linalg.norm(q[1:] - q[:-1], axis=1)    # (N-1,)
    seg_dist = np.maximum(seg_dist, eps)
    speed = float(np.clip(base_speed, eps, speed_max))
    seg_dt = np.maximum(seg_dist / speed, dt_min)        # (N-1,)

    t = np.zeros(N, dtype=float)
    for i in range(1, N):
        t[i] = max(t[i-1] + dt_min, t[i-1] + float(seg_dt[i-1]))

    # ----------------- TRIM & RE-ZERO -----------------
    if computation_time > 0.0:
        kept_idx = [i for i in range(N) if t[i] >= computation_time]
        if not kept_idx:
            traj.points = []  # nothing feasible left
            return
        t0 = computation_time
        t = t[kept_idx] - t0
        q = q[kept_idx]
        pts = [traj.points[i] for i in kept_idx]
        N = len(pts)
        for i in range(N):
            pts[i].time_from_start = _sec_to_dur(float(t[i]))
        traj.points = pts

        # ----------------- SIMPLE STARTUP PADDING -----------------
        if startup_pad > 0.0 and N >= 1:
            t = t + float(startup_pad)
            # enforce strictly increasing times with dt_min
            for i in range(1, N):
                t[i] = max(t[i], t[i-1] + dt_min)
            for i in range(N):
                traj.points[i].time_from_start = _sec_to_dur(float(t[i]))
    else:
        # ensure the first timestamp is exactly zero
        t[0] = 0.0
        for i in range(N):
            traj.points[i].time_from_start = _sec_to_dur(float(t[i]))                       

    # ----------------- velocities on FINAL timeline -----------------
    v = np.zeros_like(q)
    a = np.zeros_like(q)

    if N >= 2:
        dt01 = max(float(t[1] - t[0]), eps)
        v[0] = (q[1] - q[0]) / dt01
        dtN = max(float(t[N-1] - t[N-2]), eps)
        v[N-1] = (q[N-1] - q[N-2]) / dtN

    for i in range(1, N-1):
        denom = max(float(t[i+1] - t[i-1]), eps)
        v[i] = (q[i+1] - q[i-1]) / denom

    if N >= 2:
        a[0] = (v[1] - v[0]) / max(float(t[1] - t[0]), eps)
        a[N-1] = (v[N-1] - v[N-2]) / max(float(t[N-1] - t[N-2]), eps)

    for i in range(1, N-1):
        denom = max(float(t[i+1] - t[i-1]), eps)
        a[i] = (v[i+1] - v[i-1]) / denom

    if zero_final and N >= 1:
        v[N-1] = 0.0
        a[N-1] = 0.0

    for i, p in enumerate(traj.points):
        p.velocities = v[i].tolist()
        p.accelerations = a[i].tolist()

    # ----------------- execute ASAP -----------------
    if hasattr(traj, "header") and hasattr(traj.header, "stamp"):
        traj.header.stamp = Time(sec=0, nanosec=0)



def stretch_trajectory_time(traj, factor: float):
    """
    Slow down (factor>1) or speed up (factor<1) the whole trajectory timing.
    Scales time_from_start, and adjusts velocities/accelerations consistently.
    """
    if factor <= 0.0 or not traj.points:
        return
    for pt in traj.points:
        t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
        t *= factor
        sec, nsec = divmod(t, 1.0)
        pt.time_from_start.sec = int(sec)
        pt.time_from_start.nanosec = int(nsec * 1e9)
        if pt.velocities:
            pt.velocities = [v / factor for v in pt.velocities]
        if pt.accelerations:
            pt.accelerations = [a / (factor * factor) for a in pt.accelerations]


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
    hottest_pose_topic_ = "hottest_pose"

    base_ = "base_link"
    end_effector_ = "heat_gun"
    # end_effector_ = "butane_torch"

    def __init__(self) -> None:
        super().__init__("robot_interface_node")
        self.get_logger().info("Starting RobotInterfaceNode constructor...")

        self._heating_loop_event = threading.Event()
        self._heating_thread_enable = True
        self._data_lock = threading.Lock()

        # self.z_offset = 0.05
        # self.z_offset = 0.1
        self.z_offset = 0.0
        self.z_safety_offset = 0.35
        self.overheat_mode = False
        self.task_in_progress = False
        # self.query_lock = threading.Lock()
        self.latest_pose_msg = None
        self.latest_coldest_pose = None
        self.latest_hottest_pose = None
        self.latest_hottest_temp = None
        self.timer_callback_group = MutuallyExclusiveCallbackGroup()
        self.action_callback_group = ReentrantCallbackGroup()
        # self.timer_callback_group = ReentrantCallbackGroup()
        self._control_loop_done = False
        self.mutli_executor = None
        self.loop_timer = None
        self.homed_init = True
        self.last_pose = None
        self.anchor_pose = None
        self.latest_traj = None
        self.latest_velocity_scaling = None
        self.execution_in_progress = False
        self.run_once = False
        self.safety_run_once = False
        self.latest_traj_version = 0
        self.sent_traj_version = -1
        self.motion_plan_updated = True
        self.executor_updated = True

        self.hottest_cache = {
            "hottest_temp": {"value": float("nan"), "updated": False},
            "coldest_pose": {"msg": None, "updated": False},
            "policy": {"msg": None, "updated": False},
            "traj": {"in_progress": None}
            }
        self.hottest_cache_lock = threading.Lock()

        # start the dedicated monitor thread
        self.hottest_thread, self.hottest_exec, self.hottest_node = start_subscriber_thread(
            self.hottest_cache, self.hottest_cache_lock, logger=self.get_logger()
        )



        self.base_frame = "world"     
        self.move_group_name = "ur10e"
        # self.eef_frame = 'tool0'
        # self.eef_frame = 'butane_torch'
        self.eef_frame = 'heat_gun'
        self.cartesian_path_service_name = '/compute_cartesian_path'
        self.trajectory_action_name = '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        self.joint_state_topic_ = '/joint_states'
        self.cartesian_path_service_name = '/compute_cartesian_path'
        self.trajectory_action_name = '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        self.joint_state_topic_ = '/joint_states'
        self.policy_topic = '/policy_poses'
        
        # self.home_pose = Pose(
        #     position=Point(x=0.73945, y=-0.29125, z=0.7),
        #     orientation=Quaternion(x=0.96314, y=-0.26109, z=-0.00405, w=0.06467),
        #     # orientation=Quaternion(x=0.7151438676916091, y=-0.697694509569741, z=0.037979733553843105, w=0.018685814365548985),
        # )
        self.home_pose_msg = Extrema()
        self.home_pose_msg.poses = [Pose(
            position=Point(x=0.71527, y=-0.33871, z=0.46647),
            # orientation=Quaternion(x=0.71525, y=-0.69799, z=-0.025027, w=0.024673),)]
            orientation=Quaternion(x=0.4052, y=0.913, z=0.0, w=0.0),)]
        # self.home_pose_offset = common_motionplan_utilities.get_tool_offset_pose(
        #     self,
        #     self.home_pose,
        #     self.z_safety_offset,
        #     self.child_frame,
        #     self.tf_buffer,
        #     transform=self.t
        # )
        # self.home_pose = Pose(
        #     position=Point(x=0.73945, y=-0.29125, z=0.5),
        #     orientation=Quaternion(x=1.0, y=0., z=-pi., w=0.),
        # )
        self.desired_temperature = 70.0 # Celsius, = 140 Fahrenheit
        self.threshold_temperature = 60.0 # Celsius, = 158F
        # self.threshold_temperature = 50.0 # Celsius, = 158F

        

        self.policy_callback_group = MutuallyExclusiveCallbackGroup()
         
        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.history = HistoryPolicy.KEEP_LAST

        # # Greedy policy subscriber
        # self.policy_subscriber = self.create_subscription(
        #     Extrema,
        #     self.policy_topic, # currently only zigzag
        #     # "coldest_pose",
        #     self.policy_callback,
        #     qos,
        #     # 10,
        #     callback_group=self.policy_callback_group
        # )

        # self.hottest_temp_callback_group = MutuallyExclusiveCallbackGroup()
        # self.hottest_temp_callback_group = ReentrantCallbackGroup()
        # self.hottest_temp_subscriber = self.create_subscription(
        #     # Extrema,
        #     Float64,
        #     "/hottest_temp",
        #     self.hottest_temp_callback,
        #     10,
        #     callback_group=self.hottest_temp_callback_group
        # )
        
        self.traj_pub = self.create_publisher(Int32, "/traj_reader", 10)

        self.target_pose_pub = self.create_publisher(PoseStamped, "/target_pose_viz", 10)

        self.debug_pose_pub = self.create_publisher(PoseStamped, '/robot_target_pose', 10)

        self.debug_anchor_pose = self.create_publisher(PoseStamped, '/anchor_pose', 10)

        self.debug_last_pose = self.create_publisher(PoseStamped, "/last_pose", 10)

        # self.debug_waypoints = self.create_publisher(PoseArray, '/debug_cartesian_waypoints', 10)
        self.debug_waypoints = self.create_publisher(MarkerArray, '/debug_cartesian_waypoints', 10)

        self.debug_hottest_temp = self.create_publisher(Float64, "/debug_hottest", 10)

        self.debug_policy_poses = self.create_publisher(PoseArray, "/debug_policy_poses", 10)

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

        # Service client to compute Cartesian path
        self.cartesian_path_client = self.create_client(GetCartesianPath, self.cartesian_path_service_name)
        if not self.cartesian_path_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f"Service {self.cartesian_path_service_name} not available.")
            raise RuntimeError("MoveIt Cartesian path service not available")

        # Action client to execute trajectory
        self.trajectory_client = ActionClient(self, FollowJointTrajectory, self.trajectory_action_name, callback_group=self.action_callback_group)
        if not self.trajectory_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f"Action server {self.trajectory_action_name} not available.")
            raise RuntimeError("Trajectory action server not available")

        self.get_logger().info("Action/service clients ready.")

        # self.create_timer(1.0, lambda: self.get_logger().debug("executor heartbeat"))

        # Initialize transforms
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.child_frame: str = None
        self.t = None

        
        # Start motion planning thread
        self.control_thread = threading.Thread(
            target=self.motion_planning_thread,
            daemon=True
        )
        self.control_thread.start()
        self.get_logger().info("Planning thread started.")

        # Start executor thread
        self.executor_thread = threading.Thread(
            target=self.trajectory_executor_thread,
            daemon=True
        )
        self.executor_thread.start()
        self.get_logger().info("Executor thread started.")

        self.get_logger().info("Subscribing to /heater_cartesian_path")
        self.heater_path_sub = self.create_subscription(
            PoseArray,
            "/heater_cartesian_path",
            self.heater_cartesian_path_callback,
            10,
            callback_group=self.timer_callback_group,
        )


    def construct_fk_msg(self) -> PoseStamped:
        """Gets a pose in world frame for visualizing the robot's
        targeted pose"""
        debug_pose = PoseStamped()
        debug_pose.header.frame_id = 'world'
        # debug_pose.header.frame_id = self.end_effector_
        debug_pose.header.stamp = self.get_clock().now().to_msg()
        raw_fk_pose = self.get_fk()
        offset_in_frame = common_motionplan_utilities.rotate_vec_by_quat(
            v=np.array([0., 0., self.z_offset]),
            q=np.array([raw_fk_pose.orientation.x, raw_fk_pose.orientation.y, raw_fk_pose.orientation.z, raw_fk_pose.orientation.w])
        )
        debug_pose.pose = raw_fk_pose
        debug_pose.pose.position = Point(
            x=debug_pose.pose.position.x+offset_in_frame[0],
            y=debug_pose.pose.position.y+offset_in_frame[1],
            z=debug_pose.pose.position.z+offset_in_frame[2]
        )
        return debug_pose

    # Multi threaded
    def policy_callback(self, msg: Extrema):
        # self.get_logger().info("CALLBACK: New pose msg received")
        # with self._data_lock:
        self.latest_pose_msg = msg
        self.motion_plan_updated = False
        # self.publish_pose_array(self.latest_pose_msg.poses, self.debug_waypoints)
        # self.get_logger().info(f"CALLBACK: motion_plan_updated={self.motion_plan_updated}")

            # self.get_logger().info("DEBUG: coldest_pose_callback stored data")
        # self.get_logger().info(f"Received coldest position: x={msg.pose.position.x:.3f}, y={msg.pose.position.y:.3f}, z={msg.pose.position.z:.3f}, temp={msg.value:.3f}C")

    # Multi threaded
    def hottest_temp_callback(self, msg):
        self.get_logger().info("DEBUG: hottest callback triggered")
        with self._data_lock:
            self.latest_hottest_temp = msg.data    
            # hottest_msg = Float64()
            # hottest_msg.data = self.latest_hottest_temp
            # self.debug_hottest_temp.publish(hottest_msg)
            # self.get_logger().info(f"Hottest temp: {self.latest_hottest_temp}")


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
    

    def heater_cartesian_path_callback(self, msg: PoseArray):
        if len(msg.poses) == 0:
            self.get_logger().warn("Received empty heater cartesian path.")
            return

        self.get_logger().info(
            f"Received heater path with {len(msg.poses)} poses. Computing cartesian trajectory."
        )

        self.debug_policy_poses.publish(msg)
        self.compute_cartesian_trajectory(list(msg.poses))

    
    def motion_planning_thread(self):
        self.get_logger().info("PLANNER: Waiting for policy poses...")

        while rclpy.ok():
            # self.get_logger().info("DEBUG: planner loop")
            # self.get_logger().info(f"motion_plan_updated={self.motion_plan_updated}")
            # test_flag = self.hottest_cache["traj"]["in_progress"]
            # self.get_logger().info(f"in_progress={test_flag}")
            # self.latest_hottest_temp = self.hottest_cache["hottest_temp"]["value"]
            # hottest_msg = Float64()
            # hottest_msg.data = self.latest_hottest_temp
            # self.debug_hottest_temp.publish(hottest_msg)
            if self.overheat_mode:
                if self.safety_run_once:
                    # self.get_logger().info("Waiting...")
                    continue 
                    
                self.home_pose_msg.header.stamp = self.get_clock().now().to_msg()
                # self.latest_pose_msg = self.home_pose_msg
                self.latest_pose_msg = self.latest_coldest_pose
                self.get_logger().info("Assigned coldest pose")
                self.motion_plan_updated = False
                self._heating_thread_enable = True
                self.safety_run_once = True
            # Read incoming pose msg
            elif self.hottest_cache["policy"]["msg"] is not None:
                with self.hottest_cache_lock:
                    policy_pose_msg = Extrema()
                    # self.latest_pose_msg = policy_pose_msg
                    policy_pose_msg = self.hottest_cache["policy"]["msg"]
                    
                    # self.get_logger().info(str(self.hottest_cache["policy"]["updated"]))
                    if self.hottest_cache["policy"]["updated"]:
                        self.latest_pose_msg = policy_pose_msg
                        self.motion_plan_updated = False
                        self.hottest_cache["policy"]["updated"] = False
                        self.get_logger().info("Received new pose msg")
                        self.publish_pose_array(policy_pose_msg.poses, self.debug_policy_poses)            

                        recv_t = time.perf_counter()
                        self.latest_policy_recv_time = recv_t
                        
                        self.get_logger().info(
                            f"[TIMING executor_recv] policy_received t={recv_t:.6f}"
                        )

            # if self._heating_thread_enable:
            if not self.motion_plan_updated and self._heating_thread_enable:
                self._heating_thread_enable = False
                self.heating_control_loop()
                self.motion_plan_updated = True

            # time.sleep(0.1) # debounce time

        self.get_logger().info("Exitted planning loop.")


    def heating_control_loop(self):
        """Main control loop - queries and executes heating"""

        # DEBUG 
        control_loop_start = time.perf_counter()
        self.get_logger().warn('DEBUG: start heating_control_loop')
        
        if self.task_in_progress:
            self.get_logger().info("Task already in progress, dropping new request.")
            self._heating_thread_enable = True
            # self._control_loop_done = True
            return

        with self._data_lock:
            # Get most recent coldest pose
            pose_msg = self.latest_pose_msg
            if pose_msg is None:
                self.get_logger().warn("Missing pose message")
                self._heating_thread_enable = True
                # self._control_loop_done = True
                # self.set_timer()
                return

        # For debug visualization in Rviz only
        # with self.hottest_cache_lock:
        #     policy_pose_msg = Extrema()
        #     policy_pose_msg.poses = self.hottest_cache["policy"]["msg"].poses
        #     if self.hottest_cache["policy"]["updated"]:
        #         self.hottest_cache["policy"]["updated"] = False
        #         self.get_logger().info("Received new pose msg")
        #         self.publish_pose_array(policy_pose_msg.poses, self.debug_policy_poses)


        try:
            if self.child_frame is None:
                self.child_frame = pose_msg.header.frame_id
                self.t = common_motionplan_utilities.get_transform_frame(
                    self,
                    self.tf_buffer,
                    pose_msg.header.frame_id,
                    'world'
                )

                # self.home_pose_offset = common_motionplan_utilities.get_tool_offset_pose(
                #     self,
                #     self.home_pose,
                #     self.z_safety_offset,
                #     self.child_frame,
                #     self.tf_buffer,
                #     transform=self.t
                # )
                # pose_msg.poses = [self.home_pose_offset]


            self.compute_cartesian_trajectory(pose_msg.poses) # trajectory planner call
        finally:
            self.task_in_progress = False
        
        self._heating_thread_enable = True

        # DEBUG
        control_loop_end = time.perf_counter()
        self.get_logger().warn(f"DEBUG: end heating_control_loop [{control_loop_end - control_loop_start:.6f}s]")  


    def trajectory_executor_thread(self):
        self.get_logger().info("EXECUTOR: Waiting for motion plan...")

        while rclpy.ok():
            # self.get_logger().info("DEBUG: executor loop")
            # self.get_logger().info(f"executor_upated={self.executor_updated}")
            # if (self.latest_traj is not None and 
            #     self.latest_velocity_scaling is not None and
            #     not self.executor_updated):
            self.latest_hottest_temp = self.hottest_cache["hottest_temp"]["value"]
            hottest_msg = Float64()
            hottest_msg.data = self.latest_hottest_temp
            self.debug_hottest_temp.publish(hottest_msg)

            self.latest_coldest_pose = self.hottest_cache["coldest_pose"]["msg"]

            if self.latest_hottest_temp >= self.threshold_temperature:
                if not self.overheat_mode:
                    self.get_logger().warn("Overheat mode activated!")
                    self.latest_overheat_time = time.perf_counter()
                self.overheat_mode = True
            elif self.overheat_mode and time.perf_counter() - self.latest_overheat_time > 3.0:
                self.get_logger().info("Overheat mode deactivated")
                self.overheat_mode = False
                self.safety_run_once = False

            if not self.executor_updated:

                # # Publish a message for debugging
                # with self.hottest_cache_lock:
                #     hottest_msg = Float64()
                #     hottest_msg.data = self.hottest_cache["value"]
            
                #     if self.hottest_cache.get("updated", False):
                #         self.hottest_cache["updated"] = False
                #         # self.get_logger().info(f"Hottest temp: {hottest_msg.data}")
                #         self.debug_hottest_temp.publish(hottest_msg)

                # Only call if a trajectory is not in progress
                ver = self.latest_traj_version
                # if not self.execution_in_progress:
                self.execute_cartesian_trajectory(self.latest_traj, plan_version=ver)

            # else:
            #     self.get_logger().info("Waiting for latest trajectory...")
            # time.sleep(1)

        self.get_logger().info("Trajectory executor ended.")
 

    def publish_pose_stamped(self, pose, publisher):
        pose_stamped = PoseStamped()
        pose_stamped.header.stamp = self.get_clock().now().to_msg()
        pose_stamped.header.frame_id = "world"   
        pose_stamped.pose = pose

        publisher.publish(pose_stamped)

    def publish_pose_array(self, poses, publisher):
        pose_array = PoseArray()
        pose_array.header.frame_id = "world"  
        pose_array.poses = poses

        publisher.publish(pose_array)

    def publish_marker_array(self, poses, publisher):
        marker_array = MarkerArray()
        header = Header()
        header.frame_id = "world"   
        header.stamp = self.get_clock().now().to_msg()

        for i, pose in enumerate(poses):
            marker = Marker()
            marker.header = header
            marker.id = i
            marker.type = Marker.SPHERE   
            marker.action = Marker.ADD
            marker.pose = pose

            # marker appearance
            marker.scale.x = 0.005
            marker.scale.y = 0.005
            marker.scale.z = 0.005
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0

            # lifetime (0 means forever)
            # marker.lifetime = Duration(sec=0)

            marker_array.markers.append(marker)

        publisher.publish(marker_array)
    
        

    # OLD
    def get_ik(self, target_pose: Pose) -> JointState | None:
        request = GetPositionIK.Request()

        request.ik_request.group_name = self.move_group_name_
        request.ik_request.pose_stamped.header.frame_id = f"{self.base_}"
        request.ik_request.pose_stamped.header.stamp = self.get_clock().now().to_msg()
        request.ik_request.pose_stamped.pose = target_pose
        request.ik_request.avoid_collisions = True

        future = self.ik_client_.call_async(request)

        # self.get_logger().info("DEBUG: before spin")
        rclpy.spin_until_future_complete(self, future)
        # self.get_logger().info("DEBUG: after spin")
        if future.result() is None:
            self.get_logger().error("Failed to get IK solution")
            return None

        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            return None

        return response.solution.joint_state


    def get_fk(self) -> Pose | None:
        # self.get_logger().info("start of fk")
        for _ in range(10):
            current_joint_state = self.get_joint_state()
            if current_joint_state is not None:
                break
        if current_joint_state is None:
            self.get_logger().error("Failed to get current joint state")
            return None
        # self.get_logger().info("got joint state in fk")

        current_robot_state = RobotState()
        current_robot_state.joint_state = current_joint_state

        request = GetPositionFK.Request()

        request.header.frame_id = f"{self.base_}"
        request.header.stamp = self.get_clock().now().to_msg()

        request.fk_link_names.append(self.end_effector_)
        request.robot_state = current_robot_state

        future = self.fk_client_.call_async(request)
        # self.get_logger().info("calling for fk")

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
        self.get_logger().info("got fk")
        return response.pose_stamped[0].pose

    def sum_of_square_diff(
        self, joint_state_1: JointState, joint_state_2: JointState
    ) -> float:
        return np.sum(
            np.square(np.subtract(joint_state_1.position, joint_state_2.position))
        )

    # OLD
    def get_best_ik(self, target_pose: Pose, attempts: int = 10) -> JointState | None:
        for _ in range(attempts):
            current_joint_state = self.get_joint_state()
            if current_joint_state is not None:
                break
        if current_joint_state is None:
            self.get_logger().info().error("Failed to get current joint state")
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

    # OLD
    def get_motion_plan(
        self, target_pose: Pose, linear: bool = True, attempts: int = 10
    ) -> RobotTrajectory | None:
        self.get_logger().info("start of get motion plan")

        current_pose = self.get_fk()
        if current_pose is None:
            self.get_logger().error("Failed to get current pose")

        if check_same_pose(current_pose, target_pose):
            self.get_logger().warn("Same pose")
            # return RobotTrajectory()
            return None

        current_joint_state_set, current_joint_state = wait_for_message(
            JointState, self, self.joint_state_topic_, time_to_wait=1.0
        )
        if not current_joint_state_set:
            self.get_logger().error("Failed to get current joint state")
            return None 

        current_robot_state = RobotState()
        current_robot_state.joint_state.position = current_joint_state.position
        self.get_logger().info("made robot state")

        target_joint_state = self.get_best_ik(target_pose, attempts=5)
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
        # request.motion_plan_request.max_acceleration_scaling_factor = 0.5
        request.motion_plan_request.max_acceleration_scaling_factor = 0.3
        # request.motion_plan_request.max_acceleration_scaling_factor = 0.8

        self.get_logger().info("finished motion plan request")

        if linear:
            request.motion_plan_request.pipeline_id = "pilz_industrial_motion_planner"
            request.motion_plan_request.planner_id = "LIN"
        else:
            request.motion_plan_request.pipeline_id = "ompl"
            request.motion_plan_request.planner_id = "BFMTkConfigDefault"

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
                return response.motion_plan_response.trajectory
            
        return None


    def get_motion_execute_client(self) -> ActionClient:
        return self.execute_client_

    def get_planning_scene(self) -> PlanningScene | None:
        request = GetPlanningScene.Request()
        request.components.components = 0

        future = self.get_planning_scene_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error("Failed to get planning scene")

        response = future.result()
        return response.scene

    # OLD
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
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error("Failed to update planning scene")
            return False
        
        return future.result().success

    # OLD
    def execute_trajectory(self, target_pose: Pose) -> None:
        traj = self.get_motion_plan(target_pose)
        # self.get_logger().info(f"traj {traj}")
        if traj:
            client = self.get_motion_execute_client()
            goal = ExecuteTrajectory.Goal()
            goal.trajectory = traj

            future = client.send_goal_async(goal)
            rclpy.spin_until_future_complete(self, future)
            
            goal_handle = future.result()
            if not goal_handle.accepted:
                self.get_logger().error("Failed to execute trajectory")
            else:
                self.get_logger().info("Trajectory accepted")

                result_future = goal_handle.get_result_async()

                # self.get_logger().info(f"Trajectory points {traj.joint_trajectory.points}")
                expect_duration = traj.joint_trajectory.points[-1].time_from_start
                expect_time = time.time() + 2 * expect_duration.sec
                # while not result_future.done() and time.time() < expect_time:
                #     time.sleep(0.01)

                # self.get_logger().info("Trajectory executed")

                result_future = goal_handle.get_result_async()
                rclpy.spin_until_future_complete(self, result_future, timeout_sec=2 * expect_duration.sec)

                if result_future.done():
                    self.get_logger().info("Trajectory executed")
                else:
                    self.get_logger().warn("Trajectory execution timed out")

                # self.get_logger().info("Current pose: " + str(self.get_fk()))
                # time.sleep(0.1)
                
                # self.get_logger().info("before debug_pose")
                # debug_pose = self.construct_fk_msg()
                # self.debug_pose_pub.publish(debug_pose)
                # self.get_logger().info("after debug_pose")

    # OLD
    def wait_future(self, future, timeout_sec: float | None = None, poll_sec: float = 0.01) -> bool:
        """Wait for a rclpy future to finish without spinning from this thread."""
        start = time.time()
        while rclpy.ok() and not future.done():
            time.sleep(poll_sec)
            if timeout_sec is not None and (time.time() - start) > timeout_sec:
                return False
        return future.done()


    def interpolate_poses(self, poses: list[Pose], steps_per_segment: int = 10) -> list[Pose]:
            """
            Linearly interpolate positions and SLERP orientations between consecutive poses.
            
            Args:
                poses: List of geometry_msgs.msg.Pose (at least 2)
                steps_per_segment: Number of interpolation steps between each pair

            Returns:
                List of interpolated Pose waypoints
            """
            interpolated_poses = []

            for i in range(len(poses) - 1):
                start = poses[i]
                end = poses[i + 1]

                # Interpolate position
                start_pos = np.array([start.position.x, start.position.y, start.position.z])
                end_pos = np.array([end.position.x, end.position.y, end.position.z])

                # Interpolate orientation using SLERP
                start_quat = [start.orientation.x, start.orientation.y, start.orientation.z, start.orientation.w]
                end_quat = [end.orientation.x, end.orientation.y, end.orientation.z, end.orientation.w]

                key_rots = R.from_quat([start_quat, end_quat])
                key_times = [0, 1]
                if np.dot(start_quat, end_quat) < 0.0:
                    end_quat = [-q for q in end_quat]
                slerp = Slerp(key_times, key_rots)
                interp_times = np.linspace(0, 1, steps_per_segment)
                interp_rots = slerp(interp_times).as_quat()  # shape: (N, 4)

                for t, quat in zip(interp_times, interp_rots):
                    interp_pos = (1 - t) * start_pos + t * end_pos

                    p = Pose()
                    p.position.x, p.position.y, p.position.z = interp_pos
                    p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat

                    interpolated_poses.append(p)

                    # # Instaneous delay to allow for callback polling
                    # if t % 20 == 0:
                    #     time.sleep(0.001)

            return interpolated_poses



    def compute_cartesian_trajectory(self, poses):
        """
        Given a list of poses, compute cartesian trajectory and store it
        """

        t0 = time.perf_counter()

        self.t = common_motionplan_utilities.get_transform_frame(
            self,
            self.tf_buffer,
            self.child_frame,
            'world'
        )

        # Assign first pose of motion plan
        # if self.latest_traj is None:
        #     # On first iteration, start at robot's current pose
        #     first_pose = self.get_fk()
        # elif self.anchor_pose is not None:
        #     # On future iterations, start where the current trajectory ends
        #     first_pose = self.anchor_pose
        start_time = time.perf_counter()
        self.get_logger().info("start time")
        first_pose = self.get_fk()
        self.get_logger().info("DEBUG: Caclulated FK")

        t1 = time.perf_counter()

        poses_offset = [first_pose]

        # Transform each pose to world frame and add z_offset
        if self.overheat_mode:
            z_offset = self.z_safety_offset
        else:
            z_offset = self.z_offset
        # else:
        for pose in poses:
            p_off = common_motionplan_utilities.get_tool_offset_pose(
                self,
                pose,
                # self.z_offset,
                z_offset,
                self.child_frame,
                self.tf_buffer,
                transform=self.t
            )

            poses_offset.append(p_off)        


        path_length = compute_path_length(poses_offset)

        t2 = time.perf_counter()

        # Adaptive interpolation
        base_steps = 5
        scaling_factor = 20
        # steps_per_segment = int(np.clip(base_steps + scaling_factor * path_length, 5, 50))
        steps_per_segment = 3

        # min_speed = 0.01
        min_speed = 0.5
        # max_speed = 0.5
        max_speed = 0.8
        # max_speed = 1.3
        velocity_scaling = min_speed + (max_speed - min_speed) * min(path_length / 1.0, 1.0)
        self.latest_velocity_scaling = velocity_scaling

        self.get_logger().info(f"PLANNER: Path length: {path_length:.3f} m → steps_per_segment={steps_per_segment}, speed={velocity_scaling:.3f}")

        waypoints = self.interpolate_poses(poses_offset, steps_per_segment=steps_per_segment)

        # Publish waypoints to Rviz for debugging
        self.publish_marker_array(waypoints, self.debug_waypoints)

        # Create Cartesian path request
        request = GetCartesianPath.Request()
        request.header.frame_id = self.base_frame
        request.header.stamp = self.get_clock().now().to_msg()
        request.group_name = self.move_group_name
        request.waypoints = waypoints
        request.max_step = 0.02
        request.jump_threshold = 0.0

        # after interpolation / marker publishing / request setup
        t3 = time.perf_counter()

        future = self.cartesian_path_client.call_async(request)

        rclpy.spin_until_future_complete(self, future)
        t4 = time.perf_counter()
        # future = self.cartesian_path_client.call_async(request)
        # if not self.spin_until_complete_or_interrupt(future):
        #     return

        if future.result() is None or future.result().error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error("PLANNER: Cartesian path planning failed.")
            return

        traj = future.result().solution.joint_trajectory
        elapsed_time = time.perf_counter() - start_time
        self.get_logger().info(f"PLANNER: Computed {len(traj.points)} trajectory points in {elapsed_time:.3f} s")

        # after retiming / final trajectory assignment
        t5 = time.perf_counter()

        recv_to_plan = float("nan")
        if self.latest_policy_recv_time is not None:
            recv_to_plan = t0 - self.latest_policy_recv_time

        # Adaptive velocity retiming
        # retime_trajectory(traj, speed=velocity_scaling)
        
        # Adjust trajectory time stamp to account for computation time
        # computation_time = 0.75
        # current_time = self.get_clock().now().to_msg()
        # print("current_time:\t", current_time)
        # start_time = current_time

        # if self.execution_in_progress:
        # if self.latest_traj_version == 2: # band aid fix
        if self.hottest_cache["traj"]["in_progress"]:
            # computation_time = 1.3
            computation_time = 0.5
            # computation_time = elapsed_time 
            # start_time = time_plus_float(time_obj=current_time, float_value=computation_time*-1.0)
        else:
            computation_time = 0
            # start_time = current_time
        # start_time = time_plus_float(time_obj=current_time, float_value=computation_time*-1.0)
        # print("start_time:\t", start_time)
        # traj.header.stamp = start_time 

        # self.get_logger().info("DEBUG: start retiming")
        retime_trajectory(traj, base_speed=velocity_scaling, computation_time=computation_time)


        # if not self.motion_plan_updated:
        self.latest_traj = traj
        self.latest_traj_version += 1
        self.get_logger().info(f"PLANNER: Trajectory stored. version={self.latest_traj_version}")
        self.executor_updated = False
        # self.motion_plan_updated = True # block motion planning until new waypoints arrive

        self.get_logger().info(
            f"[TIMING executor_plan] recv_to_plan={recv_to_plan:.3f}s "
            f"fk={t1-t0:.3f}s "
            f"waypoints={t2-t1:.3f}s "
            f"prep_request={t3-t2:.3f}s "
            f"cartesian_service={t4-t3:.3f}s "
            f"retime_finalize={t5-t4:.3f}s "
            f"total={t5-t0:.3f}s"
        )


    def execute_cartesian_trajectory(self, traj, plan_version=None):
        """
        Given a cartesian path, execute it (thread-safe for multithreaded executors).
        Blocks the calling thread until the action result is available, then returns
        the result future (same as your original return type).
        """
        
        self.executor_updated = True
        ver = self.latest_traj_version

        # if self.execution_in_progress:
        #     # self.get_logger().warn("EXECUTOR: Trajectory already in progress...")
        #     self.get_logger().warn("EXECUTOR: Replacing trajectory...")
        #     # return
        # else:
        #     self.execution_in_progress = True

        if self.hottest_cache["traj"]["in_progress"]:
            # self.get_logger().info("EXECUTOR: Replcaing trajectory...")
            self.get_logger().warn("EXECUTOR: Trajectory already in progress...")
            return
        # if self.hottest_cache["traj"]["in_progress"]:
        #     self.get_logger().warn("EXECUTOR: Replacing trajectory in progress...")
        # else:
        #     self.execution_in_progress = True
            

        self.get_logger().info(f"EXECUTOR: Starting new trajectory version {ver} with {len(traj.points)} points...")

        traj_goal = FollowJointTrajectory.Goal()
        traj_goal.trajectory = traj
        # traj_goal.trajectory = blended_trajectory

        def _on_result_ready(fut):
            # Action finished
            self.get_logger().info(f"EXECUTOR: Trajectory version {ver} execution complete.")
            # time.sleep(self.latest_velocity_scaling)
            # time.sleep(0.05)
            # self.run_once = True
            # self.executor_updated = True
            self.execution_in_progress = False

        def _on_goal_response(send_fut):
            goal_handle = send_fut.result()
            self.get_logger().info(f"_on_goal_response, version={ver}")
            if not goal_handle or not goal_handle.accepted:
                self.get_logger().error("EXECUTOR: Trajectory goal rejected")
                self.execution_in_progress = False
                return
      
            get_result_future = goal_handle.get_result_async()
            get_result_future.add_done_callback(_on_result_ready)
        
        # self.get_logger().info("DEBUG: before send_goal_future")
        send_goal_future = self.trajectory_client.send_goal_async(traj_goal)
        send_goal_future.add_done_callback(_on_goal_response)

        # time.sleep(0.1)
        # self.get_logger().info("DEBUG: after add done")



class SubscriberNode(Node):
    def __init__(self, shared_value_ref: dict, shared_lock: threading.Lock):
        super().__init__("hottest_temp_monitor")
        self._shared_value_ref = shared_value_ref
        self._shared_lock = shared_lock

        self.debug_hottest_temp = self.create_publisher(Float64, "/debug_hottest", 10)
        
        self.hottest_temp_sub = self.create_subscription(
            Float64, 
            "/hottest_temp", 
            self.hottest_callback, 
            10
        )

        self.coldest_sub = self.create_subscription(
            Extrema,
            "/coldest_pose",
            self.coldest_pose_callback,
            10
        )

        self.policy_sub = self.create_subscription(
            Extrema,
            # "/policy_poses",
            # "/greedy_policy",
            "/heater_policy_poses",
            self.policy_callback,
            10
        )

        self.traj_status_sub = self.create_subscription(
            GoalStatusArray,
            "/scaled_joint_trajectory_controller/follow_joint_trajectory/_action/status",
            self.traj_status_callback,
            10
        )

    def hottest_callback(self, msg: Float64):
        # ultra-fast, no blocking work here
        with self._shared_lock:
            self._shared_value_ref["hottest_temp"]["value"] = msg.data
            self._shared_value_ref["hottest_temp"]["updated"] = True
            # hottest_msg = Float64()
            # hottest_msg.data = self._shared_value_ref["value"]
            # # self.get_logger().info(f"Hottest temp: {hottest_msg.data}")
            # self.debug_hottest_temp.publish(hottest_msg)

    def coldest_pose_callback(self, msg: Extrema):
        with self._shared_lock:
            # self.get_logger().info("policy callback")
            self._shared_value_ref["coldest_pose"]["msg"] = msg
            self._shared_value_ref["coldest_pose"]["updated"] = True

    def policy_callback(self, msg: Extrema):
        with self._shared_lock:
            # self.get_logger().info("policy callback")
            self._shared_value_ref["policy"]["msg"] = msg
            self._shared_value_ref["policy"]["updated"] = True

            recv_t = time.perf_counter()
            self.latest_policy_recv_time = recv_t
            self.get_logger().info(
                f"[TIMING executor_recv] policy_received t={recv_t:.6f} poses={len(msg.poses)}"
            )

    def traj_status_callback(self, msg: GoalStatusArray):
        with self._shared_lock:
            # active_uuid = self._shared_value_ref["traj"].get("active_uuid", "")
            in_progress = False
            self.get_logger().info(f"status={msg.status_list[-1].status}")

            if msg.status_list[-1].status == 2:
                in_progress = True
                self.get_logger().info("In progress...")
            # SUCCEEDED, CANCELED, or ABORTED
            elif msg.status_list[-1].status >= 4:
                self.get_logger().info("Trajectory completed!")
                in_progress = False            

            # Update the shared flag (True = executing, False = not)
            self._shared_value_ref["traj"]["in_progress"] = in_progress
        

def start_subscriber_thread(shared_value_ref: dict, shared_lock: threading.Lock, *, logger) -> tuple[threading.Thread, SingleThreadedExecutor, SubscriberNode]:
    """
    Creates a tiny node + its own executor, spins in a dedicated thread.
    Returns (thread, executor, node) so you can stop/clean them later.
    """
    monitor_node = SubscriberNode(shared_value_ref, shared_lock)
    executor = SingleThreadedExecutor()
    executor.add_node(monitor_node)

    def _spin():
        logger.info("Subscriber monitor thread started.")
        try:
            executor.spin()
        finally:
            logger.info("Subscriber monitor thread exiting.")

    thread = threading.Thread(target=_spin, name="subscriber-monitor", daemon=True)
    thread.start()
    return thread, executor, monitor_node

def stop_subscriber_thread(thread: threading.Thread, executor: SingleThreadedExecutor, node: SubscriberNode, *, logger):
    try:
        executor.shutdown()          # stops spin()
    except Exception as e:
        logger.warn(f"Monitor executor shutdown: {e}")
    try:
        node.destroy_node()
    except Exception as e:
        logger.warn(f"Monitor node destroy: {e}")
    if thread.is_alive():
        thread.join(timeout=1.5)



def main(args=None):
    rclpy.init(args=args)
    robot_interface_node = RobotInterfaceNode()

    executor = MultiThreadedExecutor()
    # executor = SingleThreadedExecutor()
    executor.add_node(robot_interface_node)

    robot_interface_node.multi_executor = executor

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        # robot_interface_node.stop_executor_thread()
        robot_interface_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
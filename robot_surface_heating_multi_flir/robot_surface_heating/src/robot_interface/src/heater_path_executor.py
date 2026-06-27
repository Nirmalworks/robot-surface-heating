#!/usr/bin/env python3

import math
import threading
from collections import deque
from copy import deepcopy

import numpy as np
import rclpy
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from scipy.spatial.transform import Rotation as R, Slerp

from action_msgs.msg import GoalStatus
from builtin_interfaces.msg import Duration, Time
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseArray, PoseStamped
from moveit_msgs.msg import Constraints, MoveItErrorCodes, PositionIKRequest, RobotState
from moveit_msgs.srv import GetCartesianPath, GetPositionFK, GetPositionIK
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import Marker, MarkerArray

from thermal_camera import common_motionplan_utilities


def pose_to_arrays(pose: Pose) -> tuple[np.ndarray, np.ndarray]:
    p = np.array([pose.position.x, pose.position.y, pose.position.z], dtype=np.float64)
    q = np.array(
        [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
        dtype=np.float64,
    )
    q_norm = np.linalg.norm(q)
    if q_norm < 1e-12:
        q = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    else:
        q = q / q_norm
    return p, q


def arrays_to_pose(position_xyz: np.ndarray, quat_xyzw: np.ndarray) -> Pose:
    pose = Pose()
    pose.position.x = float(position_xyz[0])
    pose.position.y = float(position_xyz[1])
    pose.position.z = float(position_xyz[2])
    pose.orientation.x = float(quat_xyzw[0])
    pose.orientation.y = float(quat_xyzw[1])
    pose.orientation.z = float(quat_xyzw[2])
    pose.orientation.w = float(quat_xyzw[3])
    return pose


def duration_to_float(duration_msg: Duration) -> float:
    return float(duration_msg.sec) + 1.0e-9 * float(duration_msg.nanosec)


def float_to_duration(seconds: float) -> Duration:
    seconds = max(0.0, float(seconds))
    sec = int(math.floor(seconds))
    nanosec = int(round((seconds - sec) * 1.0e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    msg = Duration()
    msg.sec = sec
    msg.nanosec = nanosec
    return msg


def retime_trajectory(
    traj: JointTrajectory,
    base_speed: float = 0.15,
    *,
    dt_min: float = 0.01,
    eps: float = 1e-9,
    speed_max: float = 0.75,
    zero_final: bool = True,
) -> None:
    if traj is None or not traj.points:
        return

    if len(traj.points) == 1:
        traj.points[0].time_from_start = float_to_duration(0.0)
        if zero_final:
            n = len(traj.points[0].positions)
            traj.points[0].velocities = [0.0] * n
            traj.points[0].accelerations = [0.0] * n
        return

    q = np.stack([np.asarray(p.positions, dtype=float) for p in traj.points], axis=0)
    n_pts = q.shape[0]

    seg_dist = np.linalg.norm(q[1:] - q[:-1], axis=1)
    seg_dist = np.maximum(seg_dist, eps)

    speed = float(np.clip(base_speed, eps, speed_max))
    seg_dt = np.maximum(seg_dist / speed, dt_min)

    t = np.zeros((n_pts,), dtype=float)
    for i in range(1, n_pts):
        t[i] = t[i - 1] + float(seg_dt[i - 1])

    v = np.zeros_like(q)
    a = np.zeros_like(q)

    if n_pts >= 2:
        dt0 = max(t[1] - t[0], eps)
        v[0] = (q[1] - q[0]) / dt0
        dtn = max(t[-1] - t[-2], eps)
        v[-1] = (q[-1] - q[-2]) / dtn

    for i in range(1, n_pts - 1):
        dt = max(t[i + 1] - t[i - 1], eps)
        v[i] = (q[i + 1] - q[i - 1]) / dt

    if n_pts >= 2:
        a[0] = (v[1] - v[0]) / max(t[1] - t[0], eps)
        a[-1] = (v[-1] - v[-2]) / max(t[-1] - t[-2], eps)

    for i in range(1, n_pts - 1):
        dt = max(t[i + 1] - t[i - 1], eps)
        a[i] = (v[i + 1] - v[i - 1]) / dt

    if zero_final:
        v[-1] = 0.0
        a[-1] = 0.0

    for i, pt in enumerate(traj.points):
        pt.time_from_start = float_to_duration(float(t[i]))
        pt.velocities = v[i].tolist()
        pt.accelerations = a[i].tolist()


def interpolate_pose_segment(
    pose_a: Pose,
    pose_b: Pose,
    *,
    max_translation_step_m: float,
    max_rotation_step_rad: float,
) -> list[Pose]:
    pa, qa = pose_to_arrays(pose_a)
    pb, qb = pose_to_arrays(pose_b)

    dp = np.linalg.norm(pb - pa)
    dq = (R.from_quat(qa).inv() * R.from_quat(qb)).magnitude()

    n_lin = int(math.ceil(dp / max(max_translation_step_m, 1e-9)))
    n_ang = int(math.ceil(dq / max(max_rotation_step_rad, 1e-9)))
    n_seg = max(1, n_lin, n_ang)

    key_times = np.array([0.0, 1.0], dtype=np.float64)
    key_rots = R.from_quat(np.stack([qa, qb], axis=0))
    slerp = Slerp(key_times, key_rots)

    out = []
    for i in range(1, n_seg + 1):
        u = float(i) / float(n_seg)
        p = (1.0 - u) * pa + u * pb
        q = slerp([u]).as_quat()[0]
        out.append(arrays_to_pose(p, q))

    return out


def densify_pose_sequence(
    poses: list[Pose],
    *,
    max_translation_step_m: float,
    max_rotation_step_rad: float,
) -> list[Pose]:
    if len(poses) <= 1:
        return poses

    dense = [deepcopy(poses[0])]
    for i in range(len(poses) - 1):
        dense.extend(
            interpolate_pose_segment(
                poses[i],
                poses[i + 1],
                max_translation_step_m=max_translation_step_m,
                max_rotation_step_rad=max_rotation_step_rad,
            )
        )
    return dense


def make_pose_marker_array(poses: list[Pose], frame_id: str) -> MarkerArray:
    marker_array = MarkerArray()

    for i, pose in enumerate(poses):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = Time(sec=0, nanosec=0)
        marker.ns = "heater_cartesian_path"
        marker.id = i
        marker.type = Marker.ARROW
        marker.action = Marker.ADD
        marker.pose = pose
        marker.scale.x = 0.04
        marker.scale.y = 0.008
        marker.scale.z = 0.008
        marker.color.r = 0.1
        marker.color.g = 0.9
        marker.color.b = 0.3
        marker.color.a = 0.9
        marker_array.markers.append(marker)

    return marker_array


class HeaterPathExecutor(Node):
    def __init__(self) -> None:
        super().__init__("heater_path_executor")

        self.declare_parameter("path_topic", "/heater_cartesian_path")
        self.declare_parameter("move_group_name", "ur10e")
        self.declare_parameter("base_frame", "world")
        self.declare_parameter("eef_link", "heat_gun")
        self.declare_parameter("joint_state_topic", "/joint_states")
        self.declare_parameter("cartesian_path_service", "/compute_cartesian_path")
        self.declare_parameter("trajectory_action", "/scaled_joint_trajectory_controller/follow_joint_trajectory")
        self.declare_parameter("fk_service", "compute_fk")
        self.declare_parameter("ik_service", "compute_ik")
        self.declare_parameter("max_cartesian_step_m", 0.004)
        self.declare_parameter("max_rotation_step_deg", 4.0)
        self.declare_parameter("jump_threshold", 0.0)
        self.declare_parameter("cartesian_fraction_min", 0.95)
        self.declare_parameter("base_speed", 0.15)
        self.declare_parameter("preview_only", False)
        self.declare_parameter("apply_tool_offset", False)
        self.declare_parameter("tool_offset_m", 0.0)
        self.declare_parameter("ik_check_endpoints", True)
        self.declare_parameter("prepend_current_fk", True)

        self.path_topic = self.get_parameter("path_topic").value
        self.move_group_name = self.get_parameter("move_group_name").value
        self.base_frame = self.get_parameter("base_frame").value
        self.eef_link = self.get_parameter("eef_link").value
        self.joint_state_topic = self.get_parameter("joint_state_topic").value
        self.cartesian_path_service_name = self.get_parameter("cartesian_path_service").value
        self.trajectory_action_name = self.get_parameter("trajectory_action").value
        self.fk_service_name = self.get_parameter("fk_service").value
        self.ik_service_name = self.get_parameter("ik_service").value
        self.max_cartesian_step_m = float(self.get_parameter("max_cartesian_step_m").value)
        self.max_rotation_step_rad = math.radians(float(self.get_parameter("max_rotation_step_deg").value))
        self.jump_threshold = float(self.get_parameter("jump_threshold").value)
        self.cartesian_fraction_min = float(self.get_parameter("cartesian_fraction_min").value)
        self.base_speed = float(self.get_parameter("base_speed").value)
        self.preview_only = bool(self.get_parameter("preview_only").value)
        self.apply_tool_offset = bool(self.get_parameter("apply_tool_offset").value)
        self.tool_offset_m = float(self.get_parameter("tool_offset_m").value)
        self.ik_check_endpoints = bool(self.get_parameter("ik_check_endpoints").value)
        self.prepend_current_fk = bool(self.get_parameter("prepend_current_fk").value)

        qos = QoSProfile(depth=10)
        qos.reliability = ReliabilityPolicy.RELIABLE
        qos.history = HistoryPolicy.KEEP_LAST

        self.sub_group = MutuallyExclusiveCallbackGroup()
        self.service_group = ReentrantCallbackGroup()
        self.action_group = ReentrantCallbackGroup()

        self.path_sub = self.create_subscription(
            PoseArray,
            self.path_topic,
            self.path_callback,
            qos,
            callback_group=self.sub_group,
        )

        self.preview_pub = self.create_publisher(PoseArray, "/heater_path_cartesian_preview", 10)
        self.marker_pub = self.create_publisher(MarkerArray, "/heater_path_cartesian_markers", 10)

        self.fk_client = self.create_client(
            GetPositionFK,
            self.fk_service_name,
            callback_group=self.service_group,
        )
        self.ik_client = self.create_client(
            GetPositionIK,
            self.ik_service_name,
            callback_group=self.service_group,
        )
        self.cartesian_path_client = self.create_client(
            GetCartesianPath,
            self.cartesian_path_service_name,
            callback_group=self.service_group,
        )

        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            self.trajectory_action_name,
            callback_group=self.action_group,
        )

        self.tf_buffer = common_motionplan_utilities.tf2_ros.Buffer() if hasattr(common_motionplan_utilities, "tf2_ros") else None
        self.tf_listener = None
        if self.tf_buffer is not None:
            import tf2_ros
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.path_queue: deque[PoseArray] = deque(maxlen=1)
        self.queue_lock = threading.Lock()
        self.execution_lock = threading.Lock()
        self.executing = False

        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()

        self.wait_for_dependencies()

        self.get_logger().info(f"Listening for cartesian heater paths on {self.path_topic}")

    def wait_for_dependencies(self) -> None:
        if not self.fk_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"FK service '{self.fk_service_name}' not available")
        if not self.ik_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"IK service '{self.ik_service_name}' not available")
        if not self.cartesian_path_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"Cartesian path service '{self.cartesian_path_service_name}' not available")
        if not self.trajectory_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f"Trajectory action '{self.trajectory_action_name}' not available")

    def path_callback(self, msg: PoseArray) -> None:
        if len(msg.poses) == 0:
            self.get_logger().warn("Received empty /heater_path_cartesian PoseArray")
            return

        with self.queue_lock:
            self.path_queue.clear()
            self.path_queue.append(msg)

        self.get_logger().info(f"Queued heater path with {len(msg.poses)} poses")

    def worker_loop(self) -> None:
        while rclpy.ok():
            msg = None
            with self.queue_lock:
                if self.path_queue:
                    msg = self.path_queue.popleft()

            if msg is None:
                continue

            with self.execution_lock:
                self.executing = True
                try:
                    self.process_pose_array(msg)
                except Exception as exc:
                    self.get_logger().error(f"Execution failed: {exc}")
                finally:
                    self.executing = False

    def process_pose_array(self, msg: PoseArray) -> None:
        input_poses = [deepcopy(p) for p in msg.poses]

        if self.apply_tool_offset:
            input_poses = [self.apply_offset_to_pose(p, msg.header.frame_id) for p in input_poses]
            input_poses = [p for p in input_poses if p is not None]
            if not input_poses:
                self.get_logger().error("All poses were invalid after tool offset application")
                return

        if self.ik_check_endpoints:
            if not self.check_endpoint_ik(input_poses[0], "first"):
                return
            if not self.check_endpoint_ik(input_poses[-1], "last"):
                return

        poses_for_plan = input_poses

        if self.prepend_current_fk:
            fk_pose = common_motionplan_utilities.get_fk(
                node=self,
                base_frame=self.base_frame,
                eef_link_name=self.eef_link,
                joint_state_topic=self.joint_state_topic,
                fk_client=self.fk_client,
                verbose=False,
            )
            if fk_pose is None:
                self.get_logger().error("Failed to get current FK pose")
                return
            poses_for_plan = [fk_pose] + poses_for_plan

        dense_poses = densify_pose_sequence(
            poses_for_plan,
            max_translation_step_m=self.max_cartesian_step_m,
            max_rotation_step_rad=self.max_rotation_step_rad,
        )

        preview_msg = PoseArray()
        preview_msg.header = msg.header
        preview_msg.header.frame_id = self.base_frame
        preview_msg.poses = [deepcopy(p) for p in dense_poses]
        self.preview_pub.publish(preview_msg)
        self.marker_pub.publish(make_pose_marker_array(dense_poses, self.base_frame))

        if self.preview_only:
            self.get_logger().info("Preview-only mode enabled. Skipping execution.")
            return

        trajectory = self.compute_cartesian_trajectory(dense_poses)
        if trajectory is None:
            return

        retime_trajectory(trajectory, base_speed=self.base_speed)
        self.send_trajectory(trajectory)

    def apply_offset_to_pose(self, pose: Pose, frame_id: str) -> Pose | None:
        if self.tf_buffer is None:
            self.get_logger().error("TF buffer not available for tool offset application")
            return None

        return common_motionplan_utilities.get_tool_offset_pose(
            node=self,
            original_pose=pose,
            z_offset=self.tool_offset_m,
            frame_id=frame_id,
            tf_buffer=self.tf_buffer,
            verbose=False,
        )

    def check_endpoint_ik(self, pose: Pose, label: str) -> bool:
        joint_state = common_motionplan_utilities.get_best_ik(
            node=self,
            target_pose=pose,
            joint_state_topic=self.joint_state_topic,
            move_group_name=self.move_group_name,
            base_frame=self.base_frame,
            ik_client=self.ik_client,
            attempts=6,
            verbose=False,
        )
        if joint_state is None:
            self.get_logger().error(f"IK failed for {label} waypoint")
            return False
        return True

    def current_robot_state(self) -> RobotState | None:
        joint_state = common_motionplan_utilities.get_joint_state(self, self.joint_state_topic)
        if joint_state is None:
            return None
        state = RobotState()
        state.joint_state = joint_state
        return state

    def compute_cartesian_trajectory(self, poses: list[Pose]) -> JointTrajectory | None:
        if len(poses) < 2:
            self.get_logger().error("Need at least 2 poses to compute a cartesian trajectory")
            return None

        robot_state = self.current_robot_state()
        if robot_state is None:
            self.get_logger().error("Failed to read current robot state")
            return None

        req = GetCartesianPath.Request()
        req.header.frame_id = self.base_frame
        req.header.stamp = self.get_clock().now().to_msg()
        req.start_state = robot_state
        req.group_name = self.move_group_name
        req.link_name = self.eef_link
        req.waypoints = [deepcopy(p) for p in poses]
        req.max_step = float(self.max_cartesian_step_m)
        req.jump_threshold = float(self.jump_threshold)
        req.avoid_collisions = True

        future = self.cartesian_path_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        if future.result() is None:
            self.get_logger().error("Cartesian path service call failed")
            return None

        resp = future.result()

        if resp.fraction < self.cartesian_fraction_min:
            self.get_logger().error(
                f"Cartesian path fraction too low: {resp.fraction:.3f} < {self.cartesian_fraction_min:.3f}"
            )
            return None

        if not resp.solution.joint_trajectory.points:
            self.get_logger().error("Cartesian planner returned an empty trajectory")
            return None

        self.get_logger().info(
            f"Computed cartesian path with fraction={resp.fraction:.3f}, "
            f"points={len(resp.solution.joint_trajectory.points)}"
        )
        return resp.solution.joint_trajectory

    def send_trajectory(self, joint_trajectory: JointTrajectory) -> None:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = joint_trajectory

        send_future = self.trajectory_client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, send_future)

        goal_handle = send_future.result()
        if goal_handle is None:
            self.get_logger().error("Failed to send trajectory goal")
            return

        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal was rejected")
            return

        self.get_logger().info("Trajectory goal accepted")

        result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, result_future)

        result = result_future.result()
        if result is None:
            self.get_logger().error("No result returned from trajectory action")
            return

        status = result.status
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Trajectory execution succeeded")
        else:
            self.get_logger().error(f"Trajectory execution finished with status={status}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HeaterPathExecutor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
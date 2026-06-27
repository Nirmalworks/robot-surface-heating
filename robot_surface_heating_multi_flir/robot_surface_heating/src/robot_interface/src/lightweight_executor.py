#!/usr/bin/env python3

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.spatial.transform import Rotation as R

import rclpy
from action_msgs.msg import GoalStatusArray
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, PoseArray
from moveit_msgs.msg import DisplayTrajectory, MoveItErrorCodes, RobotState, RobotTrajectory
from moveit_msgs.srv import GetCartesianPath
from rclpy.action import ActionClient
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup, ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64, Int32, String
from thermal_camera_interfaces.msg import Extrema
import tf2_ros
from trajectory_msgs.msg import JointTrajectory
from visualization_msgs.msg import Marker, MarkerArray


POSE_TOPIC = "/heater_policy_poses"
BUSY_TOPIC = "/lightweight_executor_busy"
EVENT_TOPIC = "/lightweight_executor_event"
EST_DURATION_TOPIC = "/lightweight_executor_est_duration_s"
DISPLAY_TOPIC = "/display_planned_path"
TRAJ_DEBUG_TOPIC = "/lightweight_executor/joint_trajectory"
INPUT_POSE_DEBUG_TOPIC = "/lightweight_executor/debug_input_poses"
TRANSFORMED_POSE_DEBUG_TOPIC = "/lightweight_executor/debug_transformed_poses"
MARKER_TOPIC = "/lightweight_executor/debug_waypoints"
JOINT_STATE_TOPIC = "/joint_states"
STATUS_TOPIC = "/scaled_joint_trajectory_controller/follow_joint_trajectory/_action/status"
CARTESIAN_SERVICE = "/compute_cartesian_path"
TRAJ_ACTION = "/scaled_joint_trajectory_controller/follow_joint_trajectory"

# New lightweight execution status topics for replanner integration
EXEC_ACTIVE_TOPIC = "/lightweight_executor/exec_active"
EXEC_ELAPSED_TOPIC = "/lightweight_executor/exec_elapsed_s"
EXEC_REMAINING_TOPIC = "/lightweight_executor/exec_remaining_s"
EXEC_PROGRESS_TOPIC = "/lightweight_executor/exec_progress"
EXEC_ACTIVE_JOB_TOPIC = "/lightweight_executor/active_job_id"
EXEC_QUEUED_JOB_TOPIC = "/lightweight_executor/queued_job_id"
EXEC_COMMANDED_DURATION_TOPIC = "/lightweight_executor/commanded_duration_s"

DEFAULT_PLANNING_FRAME = "base_link"
DEFAULT_MOVE_GROUP = "ur10e"
DEFAULT_TOOL_LINK = "heat_gun"
DEFAULT_MAX_STEP_M = 0.02
DEFAULT_JUMP_THRESHOLD = 0.0
DEFAULT_POSE_QUEUE_DEPTH = 10
DEFAULT_NOMINAL_SPEED_MPS = 0.1
DEFAULT_MIN_DURATION_S = 0.25
DEFAULT_START_BRIDGE_DIST_M = 0.0
DEFAULT_STATUS_TIMER_S = 0.25
DEFAULT_AVOID_COLLISIONS = True
DEFAULT_TOOL_CORR_ROLL_DEG = 180.0
DEFAULT_TOOL_CORR_PITCH_DEG = 0.0
DEFAULT_TOOL_CORR_YAW_DEG = 0.0

DEFAULT_JOIN_LOOKAHEAD_POINTS = 5
DEFAULT_JOIN_SKIP_DIST_M = 0.015
DEFAULT_JOIN_USE_CLOSEST_IN_LOOKAHEAD = True


@dataclass
class PathJob:
    sequence_id: int
    receive_perf: float
    receive_ros: float
    pose_count: int
    path_length_m: float
    est_duration_s: float
    input_frame: str
    transformed_poses: list[Pose]
    first_pose: Pose
    last_pose: Pose


@dataclass
class PlannedJob:
    path_job: PathJob
    planning_started_perf: float
    planning_finished_perf: float
    joint_traj: JointTrajectory
    display_traj: DisplayTrajectory
    plan_fraction: float
    bridge_used: bool
    bridge_distance_m: float


class LightweightExecutor(Node):
    def __init__(self) -> None:
        super().__init__("lightweight_executor")

        self.sequence_counter = 0
        self.last_receive_perf: Optional[float] = None
        self.latest_joint_state: Optional[JointState] = None
        self.latest_tool_pose: Optional[Pose] = None

        self.active_goal_handle = None
        self.active_send_perf: Optional[float] = None
        self.active_start_perf: Optional[float] = None
        self.active_commanded_duration_s: float = 0.0
        self.active_job: Optional[PlannedJob] = None
        self.planning_busy = False
        self.pending_job: Optional[PathJob] = None
        self.queued_job: Optional[PathJob] = None
        self.latest_status_code: Optional[int] = None

        self.policy_callback_group = MutuallyExclusiveCallbackGroup()
        self.client_callback_group = ReentrantCallbackGroup()

        # self.execute_enabled = bool(self.declare_parameter("execute_enabled", False).value)
        self.execute_enabled = bool(self.declare_parameter("execute_enabled", True).value)
        self.use_current_pose_bridge = bool(
            self.declare_parameter("use_current_pose_bridge", self.execute_enabled).value
        )
        self.nominal_speed_mps = float(
            self.declare_parameter("nominal_speed_mps", DEFAULT_NOMINAL_SPEED_MPS).value
        )
        self.max_step_m = float(self.declare_parameter("max_step_m", DEFAULT_MAX_STEP_M).value)
        self.jump_threshold = float(
            self.declare_parameter("jump_threshold", DEFAULT_JUMP_THRESHOLD).value
        )
        self.start_bridge_dist_m = float(
            self.declare_parameter("start_bridge_dist_m", DEFAULT_START_BRIDGE_DIST_M).value
        )
        self.planning_frame = str(
            self.declare_parameter("planning_frame", DEFAULT_PLANNING_FRAME).value
        )
        self.move_group = str(self.declare_parameter("move_group", DEFAULT_MOVE_GROUP).value)
        self.tool_link = str(self.declare_parameter("tool_link", DEFAULT_TOOL_LINK).value)
        self.avoid_collisions = bool(
            self.declare_parameter("avoid_collisions", DEFAULT_AVOID_COLLISIONS).value
        )
        self.tool_corr_roll_deg = float(
            self.declare_parameter("tool_corr_roll_deg", DEFAULT_TOOL_CORR_ROLL_DEG).value
        )
        self.tool_corr_pitch_deg = float(
            self.declare_parameter("tool_corr_pitch_deg", DEFAULT_TOOL_CORR_PITCH_DEG).value
        )
        self.tool_corr_yaw_deg = float(
            self.declare_parameter("tool_corr_yaw_deg", DEFAULT_TOOL_CORR_YAW_DEG).value
        )
        self.pose_queue_depth = int(
            self.declare_parameter("pose_queue_depth", DEFAULT_POSE_QUEUE_DEPTH).value
        )
        self.status_timer_s = float(
            self.declare_parameter("status_timer_s", DEFAULT_STATUS_TIMER_S).value
        )

        self.join_lookahead_points = int(
            self.declare_parameter("join_lookahead_points", DEFAULT_JOIN_LOOKAHEAD_POINTS).value
        )
        self.join_skip_dist_m = float(
            self.declare_parameter("join_skip_dist_m", DEFAULT_JOIN_SKIP_DIST_M).value
        )
        self.join_use_closest_in_lookahead = bool(
            self.declare_parameter(
                "join_use_closest_in_lookahead",
                DEFAULT_JOIN_USE_CLOSEST_IN_LOOKAHEAD,
            ).value
        )

        self.path_sub = self.create_subscription(
            Extrema,
            POSE_TOPIC,
            self.path_callback,
            self.pose_queue_depth,
            callback_group=self.policy_callback_group,
        )
        self.joint_state_sub = self.create_subscription(
            JointState,
            JOINT_STATE_TOPIC,
            self.joint_state_callback,
            20,
        )
        self.status_sub = self.create_subscription(
            GoalStatusArray,
            STATUS_TOPIC,
            self.status_callback,
            20,
        )

        self.busy_pub = self.create_publisher(Bool, BUSY_TOPIC, 10)
        self.event_pub = self.create_publisher(String, EVENT_TOPIC, 10)
        self.duration_pub = self.create_publisher(Float64, EST_DURATION_TOPIC, 10)
        self.display_pub = self.create_publisher(DisplayTrajectory, DISPLAY_TOPIC, 10)
        self.traj_pub = self.create_publisher(JointTrajectory, TRAJ_DEBUG_TOPIC, 10)
        self.input_pose_debug_pub = self.create_publisher(PoseArray, INPUT_POSE_DEBUG_TOPIC, 10)
        self.transformed_pose_debug_pub = self.create_publisher(PoseArray, TRANSFORMED_POSE_DEBUG_TOPIC, 10)
        self.marker_pub = self.create_publisher(MarkerArray, MARKER_TOPIC, 10)

        # New status publishers
        self.exec_active_pub = self.create_publisher(Bool, EXEC_ACTIVE_TOPIC, 10)
        self.exec_elapsed_pub = self.create_publisher(Float64, EXEC_ELAPSED_TOPIC, 10)
        self.exec_remaining_pub = self.create_publisher(Float64, EXEC_REMAINING_TOPIC, 10)
        self.exec_progress_pub = self.create_publisher(Float64, EXEC_PROGRESS_TOPIC, 10)
        self.active_job_pub = self.create_publisher(Int32, EXEC_ACTIVE_JOB_TOPIC, 10)
        self.queued_job_pub = self.create_publisher(Int32, EXEC_QUEUED_JOB_TOPIC, 10)
        self.commanded_duration_pub = self.create_publisher(Float64, EXEC_COMMANDED_DURATION_TOPIC, 10)

        self.cartesian_client = self.create_client(
            GetCartesianPath,
            CARTESIAN_SERVICE,
            callback_group=self.client_callback_group,
        )
        if not self.cartesian_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError(f"Service {CARTESIAN_SERVICE} not available")

        self.trajectory_client = ActionClient(
            self,
            FollowJointTrajectory,
            TRAJ_ACTION,
            callback_group=self.client_callback_group,
        )
        if not self.trajectory_client.wait_for_server(timeout_sec=5.0):
            raise RuntimeError(f"Action server {TRAJ_ACTION} not available")

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.status_timer = self.create_timer(self.status_timer_s, self.publish_status)

        self.get_logger().info(f"Listening on {POSE_TOPIC}")
        self.get_logger().info(f"execute_enabled={self.execute_enabled}")
        self.get_logger().info(
            f"planning_frame={self.planning_frame} tool_link={self.tool_link} move_group={self.move_group}"
        )
        self.get_logger().info(
            f"tool_corr_rpy_deg=({self.tool_corr_roll_deg:.1f}, {self.tool_corr_pitch_deg:.1f}, {self.tool_corr_yaw_deg:.1f})"
        )
        self.get_logger().info(
            "This node transforms incoming heater poses into the planning frame, computes a MoveIt cartesian path for the heat_gun link, retimes it, and optionally sends it to the controller."
        )
        self.get_logger().info(
            "Added execution status publishers for replanner integration without changing execution behavior."
        )

    def joint_state_callback(self, msg: JointState) -> None:
        self.latest_joint_state = msg

    def status_callback(self, msg: GoalStatusArray) -> None:
        if msg.status_list:
            self.latest_status_code = int(msg.status_list[-1].status)

    def path_callback(self, msg: Extrema) -> None:
        t_recv_perf = time.perf_counter()
        t_recv_ros = self.get_clock().now().nanoseconds * 1e-9

        input_poses = list(msg.poses)
        input_frame = msg.header.frame_id.strip() if msg.header.frame_id else self.planning_frame
        if len(input_poses) < 2:
            self.get_logger().warn("Received path with fewer than 2 poses. Ignoring.")
            return

        transformed_poses = self.transform_pose_list(input_poses, input_frame, self.planning_frame)
        if transformed_poses is None or len(transformed_poses) < 2:
            self.get_logger().warn(
                f"[EXEC drop] could not transform poses from {input_frame} to {self.planning_frame}"
            )
            return

        transformed_poses = enforce_quaternion_continuity(transformed_poses)
        transformed_poses = [self.apply_tool_frame_correction(p) for p in transformed_poses]
        transformed_poses = enforce_quaternion_continuity(transformed_poses)

        path_length_m = compute_pose_path_length(transformed_poses)
        est_duration_s = max(DEFAULT_MIN_DURATION_S, path_length_m / max(self.nominal_speed_mps, 1e-6))
        dt_since_prev = float("nan")
        if self.last_receive_perf is not None:
            dt_since_prev = t_recv_perf - self.last_receive_perf
        self.last_receive_perf = t_recv_perf

        job = PathJob(
            sequence_id=self.sequence_counter,
            receive_perf=t_recv_perf,
            receive_ros=t_recv_ros,
            pose_count=len(transformed_poses),
            path_length_m=path_length_m,
            est_duration_s=est_duration_s,
            input_frame=input_frame,
            transformed_poses=transformed_poses,
            first_pose=transformed_poses[0],
            last_pose=transformed_poses[-1],
        )
        self.sequence_counter += 1

        self.log_path_summary(job, dt_since_prev)
        self.duration_pub.publish(Float64(data=est_duration_s))
        self.publish_pose_array_debug(input_poses, input_frame, self.input_pose_debug_pub)
        self.publish_pose_array_debug(transformed_poses, self.planning_frame, self.transformed_pose_debug_pub)
        self.publish_marker_debug(transformed_poses, self.planning_frame)

        if self.planning_busy:
            self.pending_job = job
            self.get_logger().info(
                f"[EXEC queue] planning_busy=True replacing pending job with id={job.sequence_id}"
            )
            return

        if self.active_goal_handle is None and self.active_job is None:
            self.begin_planning(job)
            return

        self.queued_job = job
        active_id = self.active_job.path_job.sequence_id if self.active_job else -1
        self.get_logger().info(f"[EXEC queue] active_id={active_id} queued_id={job.sequence_id}")
        self.publish_event(f"queued path {job.sequence_id}")

    def begin_planning(self, job: PathJob) -> None:
        if self.latest_joint_state is None:
            self.get_logger().warn(f"[EXEC drop] job={job.sequence_id} no joint state cached yet")
            return

        self.planning_busy = True
        t0 = time.perf_counter()

        # bridge_pose = None
        # bridge_distance_m = 0.0
        # bridge_used = False
        # if self.use_current_pose_bridge:
        #     bridge_pose = self.lookup_current_tool_pose()

        # waypoints = list(job.transformed_poses)
        # if bridge_pose is not None and waypoints:
        #     bridge_distance_m = pose_distance(bridge_pose, waypoints[0])
        #     bridge_used = bridge_distance_m > self.start_bridge_dist_m
        #     if bridge_used:
        #         waypoints = [bridge_pose] + waypoints
        bridge_pose = None
        bridge_distance_m = 0.0
        bridge_used = False
        skipped_prefix_count = 0

        if self.use_current_pose_bridge:
            bridge_pose = self.lookup_current_tool_pose()

        waypoints = list(job.transformed_poses)

        if bridge_pose is not None and waypoints:
            waypoints, skipped_prefix_count = trim_waypoints_for_smoother_entry(
                bridge_pose,
                waypoints,
                lookahead_points=self.join_lookahead_points,
                min_skip_dist_m=self.join_skip_dist_m,
                use_closest_in_lookahead=self.join_use_closest_in_lookahead,
            )

            bridge_distance_m = pose_distance(bridge_pose, waypoints[0])
            bridge_used = bridge_distance_m > self.start_bridge_dist_m

            if bridge_used:
                waypoints = [bridge_pose] + waypoints

        self.get_logger().info(
            f"[EXEC join] job={job.sequence_id} skipped_prefix_count={skipped_prefix_count} "
            f"bridge_used={bridge_used} bridge_dist={bridge_distance_m:.3f}m "
            f"remaining_waypoints={len(waypoints)}"
        )

        request = GetCartesianPath.Request()
        request.header.frame_id = self.planning_frame
        request.header.stamp = self.get_clock().now().to_msg()
        request.group_name = self.move_group
        request.link_name = self.tool_link
        request.max_step = self.max_step_m
        request.jump_threshold = self.jump_threshold
        request.waypoints = waypoints
        request.avoid_collisions = self.avoid_collisions
        request.start_state = RobotState(joint_state=self.latest_joint_state)

        future = self.cartesian_client.call_async(request)
        future.add_done_callback(
            lambda fut, job=job, t0=t0, bridge_used=bridge_used, bridge_distance_m=bridge_distance_m: self._on_cartesian_done(
                fut,
                job,
                t0,
                bridge_used,
                bridge_distance_m,
            )
        )

    def _on_cartesian_done(
        self,
        future,
        job: PathJob,
        t0: float,
        bridge_used: bool,
        bridge_distance_m: float,
    ) -> None:
        t1 = time.perf_counter()
        self.planning_busy = False

        try:
            result = future.result()
        except Exception as exc:
            self.get_logger().error(f"[EXEC plan] job={job.sequence_id} cartesian service exception: {exc}")
            self.maybe_start_next_after_idle()
            return

        if result is None:
            self.get_logger().error(f"[EXEC plan] job={job.sequence_id} cartesian service returned None")
            self.maybe_start_next_after_idle()
            return
        if result.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"[EXEC plan] job={job.sequence_id} cartesian planning failed error={result.error_code.val} fraction={float(result.fraction):.3f}"
            )
            self.maybe_start_next_after_idle()
            return

        joint_traj = result.solution.joint_trajectory
        retime_joint_trajectory(
            joint_traj,
            base_speed=estimate_joint_speed(job.path_length_m, self.nominal_speed_mps),
        )
        # retime_joint_trajectory(
        #     joint_traj,
        #     cartesian_poses=job.transformed_poses,
        #     nominal_speed_mps=self.nominal_speed_mps,
        # )

        display = DisplayTrajectory()
        display.model_id = ""
        rt = RobotTrajectory()
        rt.joint_trajectory = joint_traj
        display.trajectory = [rt]
        display.trajectory_start = RobotState(
            joint_state=self.latest_joint_state if self.latest_joint_state is not None else JointState()
        )

        planned = PlannedJob(
            path_job=job,
            planning_started_perf=t0,
            planning_finished_perf=t1,
            joint_traj=joint_traj,
            display_traj=display,
            plan_fraction=float(result.fraction),
            bridge_used=bridge_used,
            bridge_distance_m=bridge_distance_m,
        )

        self.display_pub.publish(display)
        self.traj_pub.publish(joint_traj)

        self.get_logger().info(
            "[EXEC plan] "
            f"job={job.sequence_id} frame={job.input_frame}->{self.planning_frame} poses={job.pose_count} "
            f"path_length={job.path_length_m:.3f}m est_duration={job.est_duration_s:.3f}s "
            f"plan_time={t1 - t0:.3f}s fraction={planned.plan_fraction:.3f} "
            f"joint_points={len(joint_traj.points)} bridge_used={planned.bridge_used} "
            f"bridge_dist={planned.bridge_distance_m:.3f}m"
        )

        if self.execute_enabled:
            self.start_execution(planned)
        else:
            self.active_job = planned
            self.get_logger().info(
                f"[EXEC dry_run] job={job.sequence_id} planned and published to {DISPLAY_TOPIC}; not sending to controller"
            )
            self.publish_event(f"planned path {job.sequence_id} (dry run)")
            self.active_job = None
            self.maybe_start_next_after_idle()

    def start_execution(self, planned: PlannedJob) -> None:
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = planned.joint_traj

        self.active_job = planned
        self.active_send_perf = time.perf_counter()
        self.active_start_perf = None
        self.active_commanded_duration_s = joint_trajectory_duration_s(planned.joint_traj)
        self.get_logger().info(
            f"[EXEC send] job={planned.path_job.sequence_id} points={len(planned.joint_traj.points)}"
        )

        send_future = self.trajectory_client.send_goal_async(goal)
        send_future.add_done_callback(self._on_goal_response)

    def _on_goal_response(self, future) -> None:
        try:
            goal_handle = future.result()
        except Exception as exc:
            self.get_logger().error(f"[EXEC send] goal send exception: {exc}")
            self.active_goal_handle = None
            self.active_job = None
            self.active_send_perf = None
            self.active_start_perf = None
            self.active_commanded_duration_s = 0.0
            self.maybe_start_next_after_idle()
            return

        if goal_handle is None or not goal_handle.accepted:
            self.get_logger().error("[EXEC send] goal rejected")
            self.active_goal_handle = None
            self.active_job = None
            self.active_send_perf = None
            self.active_start_perf = None
            self.active_commanded_duration_s = 0.0
            self.maybe_start_next_after_idle()
            return

        self.active_goal_handle = goal_handle
        self.active_start_perf = time.perf_counter()
        delay = self.active_start_perf - (self.active_send_perf or self.active_start_perf)
        job_id = self.active_job.path_job.sequence_id if self.active_job is not None else -1
        self.get_logger().info(f"[EXEC send] goal accepted job={job_id} response_delay={delay:.3f}s")
        self.publish_event(f"started path {job_id}")

        result_future = goal_handle.get_result_async()
        result_future.add_done_callback(self._on_result_ready)

    def _on_result_ready(self, future) -> None:
        delay = time.perf_counter() - (self.active_send_perf or time.perf_counter())
        job_id = self.active_job.path_job.sequence_id if self.active_job is not None else -1
        self.get_logger().info(f"[EXEC done] job={job_id} total_exec_wait={delay:.3f}s")
        self.publish_event(f"completed path {job_id}")
        self.active_goal_handle = None
        self.active_job = None
        self.active_send_perf = None
        self.active_start_perf = None
        self.active_commanded_duration_s = 0.0
        self.maybe_start_next_after_idle()

    def maybe_start_next_after_idle(self) -> None:
        if self.planning_busy:
            return
        next_job = None
        if self.pending_job is not None:
            next_job = self.pending_job
            self.pending_job = None
        elif self.queued_job is not None:
            next_job = self.queued_job
            self.queued_job = None

        if next_job is not None:
            self.begin_planning(next_job)

    def lookup_current_tool_pose(self) -> Optional[Pose]:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.planning_frame,
                self.tool_link,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05),
            )
        except Exception as exc:
            self.get_logger().warn(
                f"[EXEC tf] failed to get {self.planning_frame}->{self.tool_link}: {exc}"
            )
            return None

        p = Pose()
        p.position.x = transform.transform.translation.x
        p.position.y = transform.transform.translation.y
        p.position.z = transform.transform.translation.z
        p.orientation = transform.transform.rotation
        self.latest_tool_pose = p
        return p

    def transform_pose_list(
        self,
        poses: list[Pose],
        source_frame: str,
        target_frame: str,
    ) -> Optional[list[Pose]]:
        if source_frame == target_frame:
            return [copy_pose(p) for p in poses]

        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.10),
            )
        except Exception as exc:
            self.get_logger().warn(
                f"[EXEC tf] failed to get {target_frame}<-{source_frame}: {exc}"
            )
            return None

        T = transform_to_matrix(transform)
        return [transform_pose(T, p) for p in poses]

    def apply_tool_frame_correction(self, pose: Pose) -> Pose:
        if (
            abs(self.tool_corr_roll_deg) < 1e-12
            and abs(self.tool_corr_pitch_deg) < 1e-12
            and abs(self.tool_corr_yaw_deg) < 1e-12
        ):
            return copy_pose(pose)

        q_in = np.array(
            [
                pose.orientation.x,
                pose.orientation.y,
                pose.orientation.z,
                pose.orientation.w,
            ],
            dtype=np.float64,
        )
        r_in = R.from_quat(q_in)
        r_corr = R.from_euler(
            "xyz",
            [self.tool_corr_roll_deg, self.tool_corr_pitch_deg, self.tool_corr_yaw_deg],
            degrees=True,
        )
        q_out = (r_in * r_corr).as_quat()

        out = copy_pose(pose)
        out.orientation.x = float(q_out[0])
        out.orientation.y = float(q_out[1])
        out.orientation.z = float(q_out[2])
        out.orientation.w = float(q_out[3])
        return out

    def publish_pose_array_debug(self, poses: list[Pose], frame_id: str, pub) -> None:
        msg = PoseArray()
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.poses = poses
        pub.publish(msg)

    def publish_marker_debug(self, poses: list[Pose], frame_id: str) -> None:
        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i, pose in enumerate(poses):
            m = Marker()
            m.header.frame_id = frame_id
            m.header.stamp = stamp
            m.ns = "transformed_waypoints"
            m.id = i
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.pose = pose
            m.scale.x = 0.02
            m.scale.y = 0.004
            m.scale.z = 0.004
            m.color.a = 1.0
            m.color.r = 0.2
            m.color.g = 0.8
            m.color.b = 1.0
            markers.markers.append(m)
        self.marker_pub.publish(markers)

    def publish_status(self) -> None:
        busy = self.planning_busy or self.active_goal_handle is not None or self.active_job is not None
        self.busy_pub.publish(Bool(data=busy))

        exec_active = self.active_goal_handle is not None and self.active_job is not None and self.active_start_perf is not None
        elapsed = 0.0
        remaining = 0.0
        progress = 0.0
        active_job_id = -1
        queued_job_id = -1
        commanded_duration = 0.0

        if self.active_job is not None:
            active_job_id = self.active_job.path_job.sequence_id
        if self.queued_job is not None:
            queued_job_id = self.queued_job.sequence_id

        if exec_active:
            elapsed = max(0.0, time.perf_counter() - self.active_start_perf)
            commanded_duration = max(0.0, self.active_commanded_duration_s)
            if commanded_duration > 1e-9:
                remaining = max(0.0, commanded_duration - elapsed)
                progress = min(1.0, max(0.0, elapsed / commanded_duration))
            else:
                remaining = 0.0
                progress = 0.0

        self.exec_active_pub.publish(Bool(data=exec_active))
        self.exec_elapsed_pub.publish(Float64(data=elapsed))
        self.exec_remaining_pub.publish(Float64(data=remaining))
        self.exec_progress_pub.publish(Float64(data=progress))
        self.active_job_pub.publish(Int32(data=active_job_id))
        self.queued_job_pub.publish(Int32(data=queued_job_id))
        self.commanded_duration_pub.publish(Float64(data=commanded_duration))

    def publish_event(self, text: str) -> None:
        self.event_pub.publish(String(data=text))

    def log_path_summary(self, job: PathJob, dt_since_prev: float) -> None:
        first_xyz = pose_xyz(job.first_pose)
        last_xyz = pose_xyz(job.last_pose)
        disp = float(np.linalg.norm(last_xyz - first_xyz))
        straightness = disp / max(job.path_length_m, 1e-9)
        self.get_logger().info(
            "[EXEC recv] "
            f"id={job.sequence_id} ros_t={job.receive_ros:.3f}s frame={job.input_frame}->{self.planning_frame} "
            f"poses={job.pose_count} length={job.path_length_m:.3f}m est_duration={job.est_duration_s:.3f}s "
            f"dt_since_prev={format_float(dt_since_prev)}s "
            f"start=({first_xyz[0]:.3f},{first_xyz[1]:.3f},{first_xyz[2]:.3f}) "
            f"end=({last_xyz[0]:.3f},{last_xyz[1]:.3f},{last_xyz[2]:.3f}) "
            f"disp={disp:.3f}m straightness={straightness:.3f}"
        )


def copy_pose(p: Pose) -> Pose:
    out = Pose()
    out.position.x = p.position.x
    out.position.y = p.position.y
    out.position.z = p.position.z
    out.orientation.x = p.orientation.x
    out.orientation.y = p.orientation.y
    out.orientation.z = p.orientation.z
    out.orientation.w = p.orientation.w
    return out


def pose_xyz(p: Pose) -> np.ndarray:
    return np.array([p.position.x, p.position.y, p.position.z], dtype=float)


def pose_distance(p0: Pose, p1: Pose) -> float:
    return float(np.linalg.norm(pose_xyz(p1) - pose_xyz(p0)))

def trim_waypoints_for_smoother_entry(
    current_pose: Pose,
    waypoints: list[Pose],
    *,
    lookahead_points: int,
    min_skip_dist_m: float,
    use_closest_in_lookahead: bool,
) -> tuple[list[Pose], int]:
    """
    Optionally discard the first few waypoints if they are effectively behind the
    current robot pose and a later waypoint within the lookahead window is closer.

    Returns:
        trimmed_waypoints, skipped_count
    """
    if current_pose is None or waypoints is None or len(waypoints) < 2:
        return waypoints, 0

    max_idx = min(max(1, lookahead_points), len(waypoints) - 1)

    if not use_closest_in_lookahead:
        if len(waypoints) > max_idx:
            d0 = pose_distance(current_pose, waypoints[0])
            dk = pose_distance(current_pose, waypoints[max_idx])
            if dk + min_skip_dist_m < d0:
                return waypoints[max_idx:], max_idx
        return waypoints, 0

    dists = [pose_distance(current_pose, waypoints[i]) for i in range(max_idx + 1)]
    best_idx = int(np.argmin(dists))
    d0 = dists[0]
    dbest = dists[best_idx]

    if best_idx > 0 and (dbest + min_skip_dist_m < d0):
        return waypoints[best_idx:], best_idx

    return waypoints, 0


def compute_pose_path_length(poses: list[Pose]) -> float:
    total = 0.0
    for i in range(len(poses) - 1):
        total += pose_distance(poses[i], poses[i + 1])
    return total


def joint_trajectory_duration_s(traj: JointTrajectory) -> float:
    if traj is None or not traj.points:
        return 0.0
    last = traj.points[-1].time_from_start
    return float(last.sec) + float(last.nanosec) * 1e-9


def estimate_joint_speed(path_length_m: float, nominal_speed_mps: float) -> float:
    min_speed = 0.5
    max_speed = 0.8
    return float(min_speed + (max_speed - min_speed) * min(path_length_m / 1.0, 1.0))


def format_float(x: float) -> str:
    if not math.isfinite(x):
        return "nan"
    return f"{x:.3f}"


def transform_to_matrix(transform_stamped) -> np.ndarray:
    t = transform_stamped.transform.translation
    q = transform_stamped.transform.rotation
    T = np.eye(4, dtype=float)
    T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = np.array([t.x, t.y, t.z], dtype=float)
    return T


def transform_pose(T: np.ndarray, pose: Pose) -> Pose:
    p = np.array([pose.position.x, pose.position.y, pose.position.z, 1.0], dtype=float)
    p_out = T @ p
    R_in = R.from_quat([pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]).as_matrix()
    R_out = T[:3, :3] @ R_in
    q_out = R.from_matrix(R_out).as_quat()

    out = Pose()
    out.position.x = float(p_out[0])
    out.position.y = float(p_out[1])
    out.position.z = float(p_out[2])
    out.orientation.x = float(q_out[0])
    out.orientation.y = float(q_out[1])
    out.orientation.z = float(q_out[2])
    out.orientation.w = float(q_out[3])
    return out


def enforce_quaternion_continuity(poses: list[Pose]) -> list[Pose]:
    if not poses:
        return []
    out = [copy_pose(poses[0])]
    prev_q = np.array(
        [
            out[0].orientation.x,
            out[0].orientation.y,
            out[0].orientation.z,
            out[0].orientation.w,
        ],
        dtype=float,
    )
    for p in poses[1:]:
        q = np.array([p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w], dtype=float)
        if float(np.dot(prev_q, q)) < 0.0:
            q = -q
        cp = copy_pose(p)
        cp.orientation.x = float(q[0])
        cp.orientation.y = float(q[1])
        cp.orientation.z = float(q[2])
        cp.orientation.w = float(q[3])
        out.append(cp)
        prev_q = q
    return out


def retime_joint_trajectory(
    traj: JointTrajectory,
    *,
    base_speed: float = 1.0,
    speed_max: float = 3.0,
    dt_min: float = 0.03,
    eps: float = 1e-9,
    # zero_final: bool = True,
    zero_final: bool = False,
) -> None:
    if traj is None or not traj.points:
        return
    if len(traj.points) == 1:
        traj.points[0].time_from_start = Duration(sec=0, nanosec=0)
        return

    positions = []
    n_joints = None
    for pt in traj.points:
        if not pt.positions:
            return
        if n_joints is None:
            n_joints = len(pt.positions)
        elif len(pt.positions) != n_joints:
            return
        positions.append(np.asarray(pt.positions, dtype=float))
    q = np.stack(positions, axis=0)
    n = q.shape[0]

    seg_dist = np.linalg.norm(q[1:] - q[:-1], axis=1)
    seg_dist = np.maximum(seg_dist, eps)
    speed = float(np.clip(base_speed, eps, speed_max))
    seg_dt = np.maximum(seg_dist / speed, dt_min)

    t = np.zeros(n, dtype=float)
    for i in range(1, n):
        t[i] = max(t[i - 1] + dt_min, t[i - 1] + float(seg_dt[i - 1]))

    v = np.zeros_like(q)
    a = np.zeros_like(q)

    if n >= 2:
        dt01 = max(float(t[1] - t[0]), eps)
        v[0] = (q[1] - q[0]) / dt01
        dtn = max(float(t[-1] - t[-2]), eps)
        v[-1] = (q[-1] - q[-2]) / dtn
    for i in range(1, n - 1):
        denom = max(float(t[i + 1] - t[i - 1]), eps)
        v[i] = (q[i + 1] - q[i - 1]) / denom

    if n >= 2:
        a[0] = (v[1] - v[0]) / max(float(t[1] - t[0]), eps)
        a[-1] = (v[-1] - v[-2]) / max(float(t[-1] - t[-2]), eps)
    for i in range(1, n - 1):
        denom = max(float(t[i + 1] - t[i - 1]), eps)
        a[i] = (v[i + 1] - v[i - 1]) / denom

    if zero_final:
        v[-1] = 0.0
        a[-1] = 0.0

    for i, pt in enumerate(traj.points):
        sec = int(math.floor(t[i]))
        nanosec = int(round((t[i] - sec) * 1e9))
        if nanosec >= 1_000_000_000:
            sec += 1
            nanosec -= 1_000_000_000
        pt.time_from_start = Duration(sec=sec, nanosec=nanosec)
        pt.velocities = v[i].tolist()
        pt.accelerations = a[i].tolist()
# def retime_joint_trajectory(
#     traj: JointTrajectory,
#     *,
#     cartesian_poses: list[Pose],
#     nominal_speed_mps: float,
#     dt_min: float = 0.03,
#     eps: float = 1e-9,
# ) -> None:
#     """
#     Retime the joint trajectory using Cartesian arc length from the input
#     EEF pose path.

#     Important behavior:
#     - timing is driven by Cartesian distance
#     - velocities/accelerations are cleared so the controller is not given
#       an explicit ramp-up / ramp-down profile from this node
#     """

#     if traj is None or not traj.points:
#         return

#     if nominal_speed_mps <= 0.0:
#         nominal_speed_mps = 1e-6

#     n_traj = len(traj.points)

#     if n_traj == 1:
#         traj.points[0].time_from_start = Duration(sec=0, nanosec=0)
#         traj.points[0].velocities = []
#         traj.points[0].accelerations = []
#         return

#     if cartesian_poses is None or len(cartesian_poses) < 2:
#         return

#     cart_xyz = np.array(
#         [[p.position.x, p.position.y, p.position.z] for p in cartesian_poses],
#         dtype=float,
#     )

#     if cart_xyz.shape[0] < 2:
#         return

#     # --------------------------------------------------------
#     # Cartesian cumulative arc length
#     # --------------------------------------------------------
#     cart_seg = np.linalg.norm(cart_xyz[1:] - cart_xyz[:-1], axis=1)
#     cart_seg = np.maximum(cart_seg, 0.0)

#     cart_s = np.zeros(cart_xyz.shape[0], dtype=float)
#     if len(cart_seg) > 0:
#         cart_s[1:] = np.cumsum(cart_seg)

#     total_length_m = float(cart_s[-1])

#     # --------------------------------------------------------
#     # Desired time along Cartesian path
#     # --------------------------------------------------------
#     if total_length_m < eps:
#         t = np.arange(n_traj, dtype=float) * dt_min
#     else:
#         cart_t = cart_s / max(nominal_speed_mps, eps)

#         cart_u = cart_s / max(total_length_m, eps)
#         traj_u = np.linspace(0.0, 1.0, n_traj, dtype=float)

#         t = np.interp(traj_u, cart_u, cart_t)

#         # enforce strictly increasing timestamps
#         for i in range(1, n_traj):
#             if t[i] <= t[i - 1] + dt_min:
#                 t[i] = t[i - 1] + dt_min

#     # --------------------------------------------------------
#     # Write back only time_from_start
#     # Clear velocities/accelerations to avoid explicit ramp profile
#     # --------------------------------------------------------
#     for i, pt in enumerate(traj.points):
#         sec = int(math.floor(t[i]))
#         nanosec = int(round((t[i] - sec) * 1e9))
#         if nanosec >= 1_000_000_000:
#             sec += 1
#             nanosec -= 1_000_000_000

#         pt.time_from_start = Duration(sec=sec, nanosec=nanosec)
#         pt.velocities = []
#         pt.accelerations = []



def main(args=None) -> None:
    rclpy.init(args=args)
    node = LightweightExecutor()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

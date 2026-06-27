#!/usr/bin/env python3

import math
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from std_msgs.msg import Bool, Float64, String
from thermal_camera_interfaces.msg import Extrema


POSE_TOPIC = "/heater_policy_poses"
BUSY_TOPIC = "/toy_executor_busy"
EST_DURATION_TOPIC = "/toy_executor_est_duration_s"
EVENT_TOPIC = "/toy_executor_event"

TIMER_HZ = 20.0
NOMINAL_SPEED_MPS = 0.08
MIN_DURATION_S = 0.25
REPLACE_MARGIN_S = 0.75
REPORT_PROGRESS_EVERY_S = 0.5


@dataclass
class VirtualPath:
    sequence_id: int
    receive_perf: float
    receive_ros: float
    pose_count: int
    path_length_m: float
    est_duration_s: float
    first_pose: Pose
    last_pose: Pose
    msg: Extrema


class ToyExecutorStub(Node):
    def __init__(self) -> None:
        super().__init__("toy_executor_stub")

        self.active_path: Optional[VirtualPath] = None
        self.queued_path: Optional[VirtualPath] = None
        self.active_start_perf: Optional[float] = None
        self.last_progress_print_perf: float = -1e9
        self.last_receive_perf: Optional[float] = None
        self.sequence_counter: int = 0

        self.path_sub = self.create_subscription(
            Extrema,
            POSE_TOPIC,
            self.path_callback,
            10,
        )

        self.busy_pub = self.create_publisher(Bool, BUSY_TOPIC, 10)
        self.duration_pub = self.create_publisher(Float64, EST_DURATION_TOPIC, 10)
        self.event_pub = self.create_publisher(String, EVENT_TOPIC, 10)

        self.timer = self.create_timer(1.0 / TIMER_HZ, self.timer_callback)
        self.status_timer = self.create_timer(0.25, self.publish_status)

        self.get_logger().info(f"Listening on {POSE_TOPIC}")
        self.get_logger().info("This node does not move the robot. It only simulates execution and prints timing diagnostics.")

    def path_callback(self, msg: Extrema) -> None:
        t_recv_perf = time.perf_counter()
        t_recv_ros = self.get_clock().now().nanoseconds * 1e-9

        poses = list(msg.poses)
        pose_count = len(poses)
        if pose_count < 2:
            self.get_logger().warn("Received path with fewer than 2 poses. Ignoring.")
            return

        path_length_m = self.compute_path_length(poses)
        est_duration_s = max(MIN_DURATION_S, path_length_m / max(NOMINAL_SPEED_MPS, 1e-6))

        path = VirtualPath(
            sequence_id=self.sequence_counter,
            receive_perf=t_recv_perf,
            receive_ros=t_recv_ros,
            pose_count=pose_count,
            path_length_m=path_length_m,
            est_duration_s=est_duration_s,
            first_pose=poses[0],
            last_pose=poses[-1],
            msg=msg,
        )
        self.sequence_counter += 1

        dt_since_prev = float("nan")
        if self.last_receive_perf is not None:
            dt_since_prev = t_recv_perf - self.last_receive_perf
        self.last_receive_perf = t_recv_perf

        self.log_path_summary(path, dt_since_prev)
        self.duration_pub.publish(Float64(data=est_duration_s))

        if self.active_path is None:
            self.start_path(path)
            return

        remaining_s = self.remaining_active_time(t_recv_perf)
        self.queued_path = path

        replace_readiness = remaining_s - path.est_duration_s
        relation = "late"
        if remaining_s > REPLACE_MARGIN_S:
            relation = "early"
        elif remaining_s > 0.0:
            relation = "near_handoff"

        self.get_logger().info(
            "[TOY queue] "
            f"active_id={self.active_path.sequence_id} queued_id={path.sequence_id} "
            f"remaining_active={remaining_s:.3f}s relation={relation} "
            f"remaining_minus_new={replace_readiness:.3f}s"
        )
        self.publish_event(
            f"queued path {path.sequence_id} while active {self.active_path.sequence_id} still running; remaining={remaining_s:.3f}s"
        )

    def timer_callback(self) -> None:
        now_perf = time.perf_counter()

        if self.active_path is None or self.active_start_perf is None:
            return

        elapsed = now_perf - self.active_start_perf
        remaining = max(0.0, self.active_path.est_duration_s - elapsed)

        if now_perf - self.last_progress_print_perf >= REPORT_PROGRESS_EVERY_S:
            progress = min(1.0, elapsed / max(self.active_path.est_duration_s, 1e-6))
            self.get_logger().info(
                "[TOY progress] "
                f"active_id={self.active_path.sequence_id} elapsed={elapsed:.3f}s "
                f"remaining={remaining:.3f}s progress={progress:.1%}"
            )
            self.last_progress_print_perf = now_perf

        if elapsed + 1e-9 < self.active_path.est_duration_s:
            return

        finished = self.active_path
        self.get_logger().info(
            "[TOY complete] "
            f"id={finished.sequence_id} exec_time={elapsed:.3f}s "
            f"expected={finished.est_duration_s:.3f}s"
        )
        self.publish_event(f"completed path {finished.sequence_id}")

        self.active_path = None
        self.active_start_perf = None

        if self.queued_path is not None:
            queued = self.queued_path
            self.queued_path = None
            self.start_path(queued)

    def start_path(self, path: VirtualPath) -> None:
        t_start_perf = time.perf_counter()
        lag_from_receive = t_start_perf - path.receive_perf

        self.active_path = path
        self.active_start_perf = t_start_perf
        self.last_progress_print_perf = -1e9

        self.get_logger().info(
            "[TOY start] "
            f"id={path.sequence_id} lag_from_receive={lag_from_receive:.3f}s "
            f"poses={path.pose_count} path_length={path.path_length_m:.3f}m "
            f"est_duration={path.est_duration_s:.3f}s"
        )
        self.publish_event(f"started path {path.sequence_id}")

    def publish_status(self) -> None:
        self.busy_pub.publish(Bool(data=self.active_path is not None))

    def remaining_active_time(self, now_perf: float) -> float:
        if self.active_path is None or self.active_start_perf is None:
            return 0.0
        elapsed = now_perf - self.active_start_perf
        return max(0.0, self.active_path.est_duration_s - elapsed)

    def log_path_summary(self, path: VirtualPath, dt_since_prev: float) -> None:
        first_xyz = self.pose_xyz(path.first_pose)
        last_xyz = self.pose_xyz(path.last_pose)
        disp = np.linalg.norm(last_xyz - first_xyz)
        straightness = disp / max(path.path_length_m, 1e-9)

        self.get_logger().info(
            "[TOY recv] "
            f"id={path.sequence_id} ros_t={path.receive_ros:.3f}s "
            f"poses={path.pose_count} length={path.path_length_m:.3f}m "
            f"est_duration={path.est_duration_s:.3f}s dt_since_prev={self.format_float(dt_since_prev)}s "
            f"start=({first_xyz[0]:.3f},{first_xyz[1]:.3f},{first_xyz[2]:.3f}) "
            f"end=({last_xyz[0]:.3f},{last_xyz[1]:.3f},{last_xyz[2]:.3f}) "
            f"disp={disp:.3f}m straightness={straightness:.3f}"
        )

    def publish_event(self, text: str) -> None:
        self.event_pub.publish(String(data=text))

    @staticmethod
    def pose_xyz(p: Pose) -> np.ndarray:
        return np.array([p.position.x, p.position.y, p.position.z], dtype=float)

    @staticmethod
    def compute_path_length(poses: list[Pose]) -> float:
        total = 0.0
        for i in range(len(poses) - 1):
            p0 = ToyExecutorStub.pose_xyz(poses[i])
            p1 = ToyExecutorStub.pose_xyz(poses[i + 1])
            total += float(np.linalg.norm(p1 - p0))
        return total

    @staticmethod
    def format_float(x: float) -> str:
        if not math.isfinite(x):
            return "nan"
        return f"{x:.3f}"


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ToyExecutorStub()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

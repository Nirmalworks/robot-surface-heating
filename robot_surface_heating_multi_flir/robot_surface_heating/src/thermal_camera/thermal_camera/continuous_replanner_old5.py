#!/usr/bin/env python3
# pyright: reportInvalidTypeForm=false

import time
from dataclasses import dataclass

import numpy as np
import rclpy
import tf2_ros
import warp as wp

from geometry_msgs.msg import Point, Vector3
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float64, Header, Int32
from tf2_ros import TransformException
from thermal_camera_interfaces.msg import ProjectedHeaterPath

import thermal_camera.adhesive_single as bb
import thermal_camera.optimize_single_real as single_plan


THERMAL_TOPIC = single_plan.THERMAL_TOPIC
THERMAL_SCALE = single_plan.THERMAL_SCALE
MIN_POINTS_FOR_SNAPSHOT = single_plan.MIN_POINTS_FOR_SNAPSHOT
EEF_FRAME = single_plan.EEF_FRAME
V_MPS = single_plan.V_MPS

PUBLISH_TOPIC = "/heater_path_projected"
MANUAL_TRIGGER_TOPIC = "/heater_replan_trigger"
PLANNER_STATUS_TOPIC = "/heater_replan_busy"

EXEC_ACTIVE_TOPIC = "/lightweight_executor/exec_active"
EXEC_ELAPSED_TOPIC = "/lightweight_executor/exec_elapsed_s"
EXEC_REMAINING_TOPIC = "/lightweight_executor/exec_remaining_s"
EXEC_PROGRESS_TOPIC = "/lightweight_executor/exec_progress"
EXEC_ACTIVE_JOB_TOPIC = "/lightweight_executor/active_job_id"
EXEC_QUEUED_JOB_TOPIC = "/lightweight_executor/queued_job_id"
EXEC_COMMANDED_DURATION_TOPIC = "/lightweight_executor/commanded_duration_s"

MAIN_LOOP_HZ = 20.0
SNAPSHOT_STALE_S = 0.75
MIN_REPLAN_INTERVAL_S = 0.20
REPLAN_TIME_MARGIN_S = 1.00
MIN_PROGRESS_FRACTION = 0.25
MIN_PROGRESS_TIME_S = 0.30
AUTO_REPLAN = True
ALLOW_MANUAL_TRIGGER = True

MIN_SNAPSHOT_ADVANCE_S = 0.02
DUPLICATE_START_DIST_M = 0.01
DUPLICATE_END_DIST_M = 0.02
DUPLICATE_LEN_DIFF_M = 0.02

EXEC_STATUS_STALE_S = 1.0
USE_ACTIVE_PATH_ENDPOINT_START = True

START_TRIM_MAX_PREFIX_M = 0.08
START_PREPEND_MAX_DIST_M = 0.06
START_PREPEND_MIN_DIST_M = 0.003

# New acknowledgement-based gating.
# After publishing, do not allow another publish until executor state shows that
# a new path has actually been consumed: either active_job_id changes or queued_job_id changes.
WAIT_FOR_EXECUTOR_ACK = True
EXECUTOR_ACK_TIMEOUT_S = 3.0


@dataclass
class SnapshotData:
    points_xyz: np.ndarray
    temps_C: np.ndarray
    frame_id: str
    stamp_sec: float


@dataclass
class ActivePathState:
    msg: ProjectedHeaterPath
    publish_time_sec: float
    estimated_duration_sec: float
    source_snapshot_stamp_sec: float
    path_length_m: float
    start_xy: np.ndarray
    end_xy: np.ndarray
    center: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    axis_n: np.ndarray
    x_min: float
    y_min: float
    cell_size_m: float


@dataclass
class ExecutorState:
    exec_active: bool = False
    elapsed_s: float = 0.0
    remaining_s: float = 0.0
    progress: float = 0.0
    active_job_id: int = -1
    queued_job_id: int = -1
    commanded_duration_s: float = 0.0
    last_update_sec: float = -1e9


class ContinuousReplanner(Node):
    def __init__(self):
        super().__init__("continuous_replanner")

        self.latest_snapshot = None
        self.active_path = None
        self.exec_state = ExecutorState()

        self.replan_requested = False
        self.planning_busy = False
        self.last_plan_start_sec = -1e9
        self.last_plan_finish_sec = -1e9
        self.last_publish_snapshot_stamp_sec = -1e9

        self.wp_ready = False
        self.wp_device = None

        # Publish acknowledgement gate
        self.awaiting_executor_ack = False
        self.ack_publish_time_sec = -1e9
        self.ack_active_job_id_at_publish = -1
        self.ack_queued_job_id_at_publish = -1

        self.sub = self.create_subscription(PointCloud2, THERMAL_TOPIC, self.pointcloud_callback, 10)
        self.path_pub = self.create_publisher(ProjectedHeaterPath, PUBLISH_TOPIC, 10)
        self.busy_pub = self.create_publisher(Bool, PLANNER_STATUS_TOPIC, 10)

        if ALLOW_MANUAL_TRIGGER:
            self.trigger_sub = self.create_subscription(Bool, MANUAL_TRIGGER_TOPIC, self.trigger_callback, 10)
        else:
            self.trigger_sub = None

        self.create_subscription(Bool, EXEC_ACTIVE_TOPIC, self.exec_active_callback, 10)
        self.create_subscription(Float64, EXEC_ELAPSED_TOPIC, self.exec_elapsed_callback, 10)
        self.create_subscription(Float64, EXEC_REMAINING_TOPIC, self.exec_remaining_callback, 10)
        self.create_subscription(Float64, EXEC_PROGRESS_TOPIC, self.exec_progress_callback, 10)
        self.create_subscription(Int32, EXEC_ACTIVE_JOB_TOPIC, self.exec_active_job_callback, 10)
        self.create_subscription(Int32, EXEC_QUEUED_JOB_TOPIC, self.exec_queued_job_callback, 10)
        self.create_subscription(Float64, EXEC_COMMANDED_DURATION_TOPIC, self.exec_commanded_duration_callback, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(1.0 / MAIN_LOOP_HZ, self.main_loop)
        self.status_timer = self.create_timer(0.1, self.publish_busy_status)

        self.get_logger().info(
            "continuous_replanner using executor-acknowledgement publish gating"
        )

    def pointcloud_callback(self, msg: PointCloud2):
        pts = []
        temps = []

        for p in point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z", "thermal"),
            skip_nans=True,
        ):
            x, y, z, t = p
            if not np.isfinite(t):
                continue
            pts.append([x, y, z])
            temps.append(float(t) / THERMAL_SCALE)

        if len(pts) < MIN_POINTS_FOR_SNAPSHOT:
            return

        msg_stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        if msg_stamp_sec <= 0.0:
            msg_stamp_sec = self.get_clock().now().nanoseconds * 1e-9

        self.latest_snapshot = SnapshotData(
            points_xyz=np.asarray(pts, dtype=np.float32),
            temps_C=np.asarray(temps, dtype=np.float32),
            frame_id=msg.header.frame_id,
            stamp_sec=msg_stamp_sec,
        )

    def trigger_callback(self, msg: Bool):
        if msg.data:
            self.replan_requested = True

    def _touch_exec_state(self):
        self.exec_state.last_update_sec = self.get_clock().now().nanoseconds * 1e-9

    def _check_executor_ack(self):
        if not self.awaiting_executor_ack:
            return

        active_changed = self.exec_state.active_job_id != self.ack_active_job_id_at_publish
        queued_changed = self.exec_state.queued_job_id != self.ack_queued_job_id_at_publish

        if active_changed or queued_changed:
            self.awaiting_executor_ack = False
            self.get_logger().info(
                f"[ACK executor] active_job_id {self.ack_active_job_id_at_publish}->{self.exec_state.active_job_id}, "
                f"queued_job_id {self.ack_queued_job_id_at_publish}->{self.exec_state.queued_job_id}"
            )

    def exec_active_callback(self, msg: Bool):
        self.exec_state.exec_active = bool(msg.data)
        if not self.exec_state.exec_active:
            self.awaiting_executor_ack = False
        self._touch_exec_state()
        self._check_executor_ack()

    def exec_elapsed_callback(self, msg: Float64):
        self.exec_state.elapsed_s = float(msg.data)
        self._touch_exec_state()

    def exec_remaining_callback(self, msg: Float64):
        self.exec_state.remaining_s = float(msg.data)
        self._touch_exec_state()

    def exec_progress_callback(self, msg: Float64):
        self.exec_state.progress = float(msg.data)
        self._touch_exec_state()

    def exec_active_job_callback(self, msg: Int32):
        self.exec_state.active_job_id = int(msg.data)
        self._touch_exec_state()
        self._check_executor_ack()

    def exec_queued_job_callback(self, msg: Int32):
        self.exec_state.queued_job_id = int(msg.data)
        self._touch_exec_state()
        self._check_executor_ack()

    def exec_commanded_duration_callback(self, msg: Float64):
        self.exec_state.commanded_duration_s = float(msg.data)
        self._touch_exec_state()

    def main_loop(self):
        if self.planning_busy:
            return

        snapshot = self.latest_snapshot
        if snapshot is None:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if not self.should_plan(now_sec, snapshot):
            return

        self.planning_busy = True
        self.last_plan_start_sec = now_sec
        t_loop = time.perf_counter()

        try:
            snapshot_age = now_sec - snapshot.stamp_sec

            t_plan0 = time.perf_counter()
            plan_msg, est_duration, plan_meta, active_state, start_source = self.compute_plan_from_snapshot(snapshot)
            t_plan1 = time.perf_counter()

            if self.is_duplicate_candidate(plan_meta):
                self.last_plan_finish_sec = self.get_clock().now().nanoseconds * 1e-9
                self.replan_requested = False
                self.get_logger().info(
                    f"[SKIP duplicate] snapshot_age={snapshot_age:.3f}s "
                    f"est_path_duration={est_duration:.3f}s "
                    f"start_delta={plan_meta['start_delta_m']:.4f}m "
                    f"end_delta={plan_meta['end_delta_m']:.4f}m "
                    f"len_delta={plan_meta['len_delta_m']:.4f}m"
                )
                self.get_logger().info(
                    f"[TIMING replanner] snapshot_age={snapshot_age:.3f}s "
                    f"plan={t_plan1 - t_plan0:.3f}s "
                    f"publish=0.000s "
                    f"total_loop={time.perf_counter() - t_loop:.3f}s "
                    f"est_path_duration={est_duration:.3f}s"
                )
                return

            t_pub0 = time.perf_counter()
            self.path_pub.publish(plan_msg)
            publish_time_sec = self.get_clock().now().nanoseconds * 1e-9
            t_pub1 = time.perf_counter()

            active_state.publish_time_sec = publish_time_sec
            active_state.estimated_duration_sec = est_duration
            active_state.source_snapshot_stamp_sec = snapshot.stamp_sec
            self.active_path = active_state

            if WAIT_FOR_EXECUTOR_ACK:
                self.awaiting_executor_ack = True
                self.ack_publish_time_sec = publish_time_sec
                self.ack_active_job_id_at_publish = self.exec_state.active_job_id
                self.ack_queued_job_id_at_publish = self.exec_state.queued_job_id
                self.get_logger().info(
                    f"[WAIT ack] active_job_id={self.ack_active_job_id_at_publish} "
                    f"queued_job_id={self.ack_queued_job_id_at_publish}"
                )

            self.last_plan_finish_sec = publish_time_sec
            self.last_publish_snapshot_stamp_sec = snapshot.stamp_sec
            self.replan_requested = False

            self.get_logger().info(
                f"Published replanned path with {len(plan_msg.path_proj)} points, "
                f"est_duration={est_duration:.2f}s, "
                f"best_branch={plan_msg.best_branch_index}, cost={plan_msg.final_cost:.6f}"
            )
            self.get_logger().info(
                f"[TIMING replanner] snapshot_age={snapshot_age:.3f}s "
                f"plan={t_plan1 - t_plan0:.3f}s "
                f"publish={t_pub1 - t_pub0:.3f}s "
                f"total_loop={t_pub1 - t_loop:.3f}s "
                f"est_path_duration={est_duration:.3f}s"
            )

        except Exception as exc:
            self.last_plan_finish_sec = self.get_clock().now().nanoseconds * 1e-9
            self.get_logger().error(f"Planning failed: {exc}")
        finally:
            self.planning_busy = False

    def should_plan(self, now_sec: float, snapshot: SnapshotData) -> bool:
        if WAIT_FOR_EXECUTOR_ACK and self.awaiting_executor_ack:
            if now_sec - self.ack_publish_time_sec <= EXECUTOR_ACK_TIMEOUT_S:
                self.get_logger().info("[SKIP gated] waiting for executor acknowledgement")
                return False
            self.get_logger().warn("[ACK timeout] releasing publish gate")
            self.awaiting_executor_ack = False

        if now_sec - self.last_plan_start_sec < MIN_REPLAN_INTERVAL_S:
            return False

        if not self.snapshot_is_fresh(snapshot, now_sec):
            return False

        if not self.snapshot_is_new_enough(snapshot):
            return False

        if self.replan_requested:
            return True

        if self.active_path is None:
            return True

        if not AUTO_REPLAN:
            return False

        if self.executor_status_is_fresh(now_sec):
            if not self.exec_state.exec_active:
                return True

            armed = (
                self.exec_state.elapsed_s >= MIN_PROGRESS_TIME_S
                or self.exec_state.progress >= MIN_PROGRESS_FRACTION
                or self.exec_state.remaining_s <= REPLAN_TIME_MARGIN_S
            )
            if not armed:
                return False

            return self.exec_state.remaining_s <= REPLAN_TIME_MARGIN_S

        elapsed_sec = max(0.0, now_sec - self.active_path.publish_time_sec)
        progress_fraction = 0.0
        if self.active_path.estimated_duration_sec > 1e-6:
            progress_fraction = elapsed_sec / self.active_path.estimated_duration_sec

        remaining_sec = self.active_path.estimated_duration_sec - elapsed_sec
        armed = (
            elapsed_sec >= MIN_PROGRESS_TIME_S
            or progress_fraction >= MIN_PROGRESS_FRACTION
            or remaining_sec <= REPLAN_TIME_MARGIN_S
        )
        if not armed:
            return False

        return remaining_sec <= REPLAN_TIME_MARGIN_S or elapsed_sec >= self.active_path.estimated_duration_sec

    def executor_status_is_fresh(self, now_sec: float) -> bool:
        return (now_sec - self.exec_state.last_update_sec) <= EXEC_STATUS_STALE_S

    def snapshot_is_fresh(self, snapshot: SnapshotData, now_sec: float) -> bool:
        return (now_sec - snapshot.stamp_sec) <= SNAPSHOT_STALE_S

    def snapshot_is_new_enough(self, snapshot: SnapshotData) -> bool:
        return (snapshot.stamp_sec - self.last_publish_snapshot_stamp_sec) >= MIN_SNAPSHOT_ADVANCE_S

    def is_duplicate_candidate(self, plan_meta: dict) -> bool:
        if self.active_path is None:
            plan_meta["start_delta_m"] = float("nan")
            plan_meta["end_delta_m"] = float("nan")
            plan_meta["len_delta_m"] = float("nan")
            return False

        start_delta = float(np.linalg.norm(plan_meta["start_xy"] - self.active_path.start_xy))
        end_delta = float(np.linalg.norm(plan_meta["end_xy"] - self.active_path.end_xy))
        len_delta = abs(plan_meta["path_length_m"] - self.active_path.path_length_m)

        plan_meta["start_delta_m"] = start_delta
        plan_meta["end_delta_m"] = end_delta
        plan_meta["len_delta_m"] = len_delta

        return (
            start_delta <= DUPLICATE_START_DIST_M
            and end_delta <= DUPLICATE_END_DIST_M
            and len_delta <= DUPLICATE_LEN_DIFF_M
        )

    def ensure_warp(self):
        if self.wp_ready:
            return
        wp.init()
        self.wp_device = wp.get_preferred_device()
        self.wp_ready = True
        self.get_logger().info(f"Warp ready on device: {self.wp_device}")

    def get_current_tool_position(self, target_frame: str, child_frame: str, timeout_sec: float = 0.20):
        deadline = time.perf_counter() + timeout_sec
        while time.perf_counter() < deadline and rclpy.ok():
            try:
                if self.tf_buffer.can_transform(target_frame, child_frame, rclpy.time.Time()):
                    t = self.tf_buffer.lookup_transform(target_frame, child_frame, rclpy.time.Time())
                    return np.array(
                        [t.transform.translation.x, t.transform.translation.y, t.transform.translation.z],
                        dtype=np.float32,
                    )
            except TransformException:
                pass
            time.sleep(0.01)

        self.get_logger().warn(
            f"Could not transform {child_frame} into {target_frame} within {timeout_sec:.2f}s"
        )
        return None

    def choose_plan_start_world(self, frame_id: str):
        if USE_ACTIVE_PATH_ENDPOINT_START and self.exec_state.exec_active and self.active_path is not None:
            end_xy = np.asarray(self.active_path.end_xy, dtype=np.float32)
            end_world = (
                self.active_path.center
                + end_xy[0] * self.active_path.axis_u
                + end_xy[1] * self.active_path.axis_v
            )
            return "active_path_end", end_world, end_xy

        current_tool_xyz = self.get_current_tool_position(frame_id, EEF_FRAME)
        if current_tool_xyz is not None:
            return "current_tool", current_tool_xyz, None

        return "none", None, None

    def compute_plan_from_snapshot(self, snapshot: SnapshotData):
        t0 = time.perf_counter()
        self.ensure_warp()

        points_xyz = snapshot.points_xyz.copy()
        temps_C_raw = snapshot.temps_C.copy()
        frame_id = snapshot.frame_id

        proj = single_plan.rasterize_snapshot(
            points_xyz=points_xyz,
            temps_C=temps_C_raw,
            grid_spacing_m=single_plan.GRID_SPACING_M,
        )

        T0_C_raw = proj["T0_C_raw"]
        T0_C = proj["T0_C_filled"]
        valid_mask = proj["valid_mask"]
        Nx = proj["Nx"]
        Ny = proj["Ny"]
        L = proj["L"]
        x_min = proj["x_min"]
        y_min = proj["y_min"]

        center = np.asarray(proj["center"], dtype=np.float32)
        axis_u = np.asarray(proj["axis_u"], dtype=np.float32)
        axis_v = np.asarray(proj["axis_v"], dtype=np.float32)
        axis_n = np.asarray(proj["axis_n"], dtype=np.float32)

        t1 = time.perf_counter()

        start_source, start_world_xyz, fixed_start_proj_xy = self.choose_plan_start_world(frame_id)

        if start_world_xyz is not None:
            P0 = single_plan.choose_start_cell_from_robot(
                tool_xyz=start_world_xyz,
                center=center,
                axis_u=axis_u,
                axis_v=axis_v,
                x_min=x_min,
                y_min=y_min,
                L=L,
                valid_mask=valid_mask,
            )

            if fixed_start_proj_xy is None:
                start_u, start_v = single_plan.project_world_point_to_plane(
                    start_world_xyz, center, axis_u, axis_v
                )
                start_proj_xy = np.array([start_u, start_v], dtype=np.float32)
            else:
                start_proj_xy = fixed_start_proj_xy.astype(np.float32)

            start_cell_xy = np.array(
                [(start_proj_xy[0] - x_min) / L, (start_proj_xy[1] - y_min) / L],
                dtype=np.float32,
            )
        else:
            if single_plan.USE_FALLBACK_START:
                P0 = single_plan.choose_start_cell(valid_mask)
                start_proj_xy = None
                start_cell_xy = None
                start_source = "fallback"
                self.get_logger().warn(
                    f"Using fallback projected start because no future/current start could be resolved for {EEF_FRAME}."
                )
            else:
                raise RuntimeError(
                    f"Could not get start state for frame '{EEF_FRAME}' in '{frame_id}'."
                )

        allowed_mask = valid_mask.copy()

        T_valid = T0_C_raw[allowed_mask.astype(bool)]
        T_valid = T_valid[np.isfinite(T_valid)]
        if T_valid.size == 0:
            raise RuntimeError("No valid projected temperatures available.")

        T0_K = (T0_C + 273.15).astype(np.float32)
        T_hot_C = float(np.percentile(T_valid, single_plan.HOT_PERCENTILE))
        T_hot_K = float(T_hot_C + 273.15)
        inv_scale = 1.0 / single_plan.SCALE_K

        T0_dev_2d = wp.array(
            T0_K.reshape(-1), dtype=wp.float32, device=self.wp_device
        ).reshape((Ny, Nx))
        mask_screen_dev = wp.zeros((Ny, Nx), dtype=wp.uint8, device=self.wp_device)

        targets_ixiy, targets_cells, targets_T = bb.pick_k_targets_cold_exclusion(
            T0_C,
            P0,
            L,
            single_plan.R_EXCL_START_M,
            single_plan.R_EXCL_TARGETS_M,
            single_plan.NUM_BRANCHES,
            allowed_mask=allowed_mask,
        )

        if len(targets_cells) == 0:
            raise RuntimeError("No feasible targets found under exclusion constraints.")

        self.get_logger().info(
            f"Planning from P0=({P0[0]:.1f}, {P0[1]:.1f}) source={start_source} with {len(targets_cells)} branches"
        )
        for k, ((ix, iy), p, tC) in enumerate(zip(targets_ixiy, targets_cells, targets_T)):
            dxm = (p[0] - P0[0]) * L
            dym = (p[1] - P0[1]) * L
            dist_m = float(np.sqrt(dxm * dxm + dym * dym))
            self.get_logger().info(
                f"  branch {k + 1}: (ix,iy)=({ix},{iy}) T={tC:.2f}C dist={dist_m:.3f}m"
            )

        t2 = time.perf_counter()

        t_opt_start = time.perf_counter()
        ctrl_opt_b, spline_opt_b, J_final_b, J_hist_b, _W = bb.optimize_to_targets_batched(
            T0_C=T0_C,
            T0_dev_2d=T0_dev_2d,
            mask_screen_dev=mask_screen_dev,
            device=self.wp_device,
            P0=P0,
            P4_list=targets_cells,
            sigma_m=float(single_plan.SIGMA_M),
            h_peak=float(single_plan.H_PEAK),
            L=float(L),
            dt=float(single_plan.DT),
            T_total=float(single_plan.T_TOTAL),
            v_mps=float(single_plan.V_MPS),
            T_hot_K=float(T_hot_K),
            inv_scale=float(inv_scale),
            w_heat=float(single_plan.W_HEAT),
            w_curv=float(single_plan.W_CURV),
            w_screen=float(single_plan.W_SCREEN),
            iters=int(single_plan.ITERS),
            lr=float(single_plan.LR),
        )
        opt_elapsed_sec = time.perf_counter() - t_opt_start

        branch_results = []
        for k in range(len(targets_cells)):
            branch_results.append(
                {
                    "k": k,
                    "P4": targets_cells[k],
                    "ctrl_opt": ctrl_opt_b[k],
                    "J_final": float(J_final_b[k]),
                }
            )

        best_idx = int(np.argmax([b["J_final"] for b in branch_results]))
        best = branch_results[best_idx]

        best_spline_dense_cells = single_plan.evaluate_spline_dense(
            best["ctrl_opt"], degree=3, num=50
        )
        best_spline_dense_proj = single_plan.cells_to_proj_points(
            best_spline_dense_cells, x_min, y_min, L
        )
        ctrl_proj = single_plan.cells_to_proj_points(best["ctrl_opt"], x_min, y_min, L)
        target_proj = single_plan.cells_to_proj_points(best["P4"][None, :], x_min, y_min, L)[0]
        start_proj = single_plan.cells_to_proj_points(P0[None, :], x_min, y_min, L)[0]

        align_info = None
        if start_proj_xy is not None and start_cell_xy is not None:
            best_spline_dense_proj, best_spline_dense_cells, align_info = self.align_path_start_to_future_start(
                best_spline_dense_proj,
                best_spline_dense_cells,
                start_proj_xy,
                start_cell_xy,
            )
            start_proj = best_spline_dense_proj[0].copy()

        t3 = time.perf_counter()

        msg = ProjectedHeaterPath()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        msg.x_min = float(x_min)
        msg.y_min = float(y_min)
        msg.cell_size_m = float(L)
        msg.nx = int(Nx)
        msg.ny = int(Ny)

        msg.center = self.make_point_xyz(*center)
        msg.axis_u = self.make_vector3_xyz(*axis_u)
        msg.axis_v = self.make_vector3_xyz(*axis_v)
        msg.axis_n = self.make_vector3_xyz(*axis_n)

        msg.start_proj = self.make_point_xyz(start_proj[0], start_proj[1], 0.0)
        msg.target_proj = self.make_point_xyz(target_proj[0], target_proj[1], 0.0)

        msg.control_proj = [self.make_point_xyz(p[0], p[1], 0.0) for p in ctrl_proj]
        msg.path_proj = [self.make_point_xyz(p[0], p[1], 0.0) for p in best_spline_dense_proj]
        msg.path_cells = [self.make_point_xyz(p[0], p[1], 0.0) for p in best_spline_dense_cells]
        msg.final_cost = float(best["J_final"])
        msg.best_branch_index = int(best_idx)

        estimated_duration_sec = self.estimate_duration_from_path(best_spline_dense_proj)
        self.get_logger().info(
            f"Optimization finished in {opt_elapsed_sec:.3f}s, selected branch {best_idx + 1}, "
            f"est_path_duration={estimated_duration_sec:.2f}s"
        )
        if align_info is not None:
            self.get_logger().info(
                f"[START align] source={start_source} "
                f"trimmed_prefix={align_info['trimmed_prefix']} "
                f"trimmed_len={align_info['trimmed_len_m']:.3f}m "
                f"prepended_start={align_info['prepended_start']} "
                f"start_offset_before={align_info['start_offset_before_m']:.3f}m "
                f"start_offset_after={align_info['start_offset_after_m']:.3f}m"
            )

        t4 = time.perf_counter()

        self.get_logger().info(
            f"[TIMING replanner_detail] prep={t1-t0:.3f}s "
            f"start_pose={t2-t1:.3f}s "
            f"optimize={t3-t2:.3f}s "
            f"build_msg={t4-t3:.3f}s "
            f"total={t4-t0:.3f}s"
        )

        plan_meta = {
            "path_length_m": self.path_length_xy(best_spline_dense_proj),
            "start_xy": np.asarray(best_spline_dense_proj[0, :2], dtype=np.float32),
            "end_xy": np.asarray(best_spline_dense_proj[-1, :2], dtype=np.float32),
        }

        active_state = ActivePathState(
            msg=msg,
            publish_time_sec=0.0,
            estimated_duration_sec=0.0,
            source_snapshot_stamp_sec=0.0,
            path_length_m=plan_meta["path_length_m"],
            start_xy=plan_meta["start_xy"],
            end_xy=plan_meta["end_xy"],
            center=center,
            axis_u=axis_u,
            axis_v=axis_v,
            axis_n=axis_n,
            x_min=float(x_min),
            y_min=float(y_min),
            cell_size_m=float(L),
        )

        return msg, estimated_duration_sec, plan_meta, active_state, start_source

    def align_path_start_to_future_start(self, path_proj, path_cells, start_proj_xy, start_cell_xy):
        if path_proj.shape[0] < 2:
            return path_proj, path_cells, {
                "trimmed_prefix": False,
                "trimmed_len_m": 0.0,
                "prepended_start": False,
                "start_offset_before_m": float(np.linalg.norm(path_proj[0, :2] - start_proj_xy[:2])),
                "start_offset_after_m": float(np.linalg.norm(path_proj[0, :2] - start_proj_xy[:2])),
            }

        original_start_offset = float(np.linalg.norm(path_proj[0, :2] - start_proj_xy[:2]))
        dists = np.linalg.norm(path_proj[:, :2] - start_proj_xy[None, :2], axis=1)
        nearest_idx = int(np.argmin(dists))

        prefix_len = 0.0
        if nearest_idx > 0:
            prefix_len = self.path_length_xy(path_proj[: nearest_idx + 1])

        trimmed_prefix = False
        if nearest_idx > 0 and prefix_len <= START_TRIM_MAX_PREFIX_M:
            path_proj = path_proj[nearest_idx:, :]
            path_cells = path_cells[nearest_idx:, :]
            trimmed_prefix = True

        prepended_start = False
        new_start_offset = float(np.linalg.norm(path_proj[0, :2] - start_proj_xy[:2]))
        if START_PREPEND_MIN_DIST_M < new_start_offset <= START_PREPEND_MAX_DIST_M:
            path_proj = np.vstack([start_proj_xy[None, :2], path_proj])
            path_cells = np.vstack([start_cell_xy[None, :2], path_cells])
            prepended_start = True

        final_start_offset = float(np.linalg.norm(path_proj[0, :2] - start_proj_xy[:2]))
        info = {
            "trimmed_prefix": trimmed_prefix,
            "trimmed_len_m": float(prefix_len if trimmed_prefix else 0.0),
            "prepended_start": prepended_start,
            "start_offset_before_m": original_start_offset,
            "start_offset_after_m": final_start_offset,
        }
        return path_proj.astype(np.float32), path_cells.astype(np.float32), info

    def estimate_duration_from_path(self, path_proj):
        total_len_m = self.path_length_xy(path_proj)
        return total_len_m / max(V_MPS, 1e-6)

    @staticmethod
    def path_length_xy(path_proj):
        if path_proj.shape[0] < 2:
            return 0.0
        diffs = np.diff(path_proj[:, :2], axis=0)
        return float(np.sum(np.linalg.norm(diffs, axis=1)))

    def publish_busy_status(self):
        msg = Bool()
        msg.data = bool(self.planning_busy)
        self.busy_pub.publish(msg)

    @staticmethod
    def make_point_xyz(x, y, z=0.0):
        p = Point()
        p.x = float(x)
        p.y = float(y)
        p.z = float(z)
        return p

    @staticmethod
    def make_vector3_xyz(x, y, z):
        v = Vector3()
        v.x = float(x)
        v.y = float(y)
        v.z = float(z)
        return v


def main():
    rclpy.init()
    node = ContinuousReplanner()
    executor = MultiThreadedExecutor()
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

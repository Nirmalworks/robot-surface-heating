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
from std_msgs.msg import Bool, Header
from tf2_ros import TransformException
from thermal_camera_interfaces.msg import ProjectedHeaterPath

import thermal_camera.adhesive_single as bb
import thermal_camera.optimize_single_real as single_plan


# ============================================================
# Replanning settings
# ============================================================

THERMAL_TOPIC = single_plan.THERMAL_TOPIC
THERMAL_SCALE = single_plan.THERMAL_SCALE
MIN_POINTS_FOR_SNAPSHOT = single_plan.MIN_POINTS_FOR_SNAPSHOT
EEF_FRAME = single_plan.EEF_FRAME
V_MPS = single_plan.V_MPS

PUBLISH_TOPIC = "/heater_path_projected"
MANUAL_TRIGGER_TOPIC = "/heater_replan_trigger"
PLANNER_STATUS_TOPIC = "/heater_replan_busy"

MAIN_LOOP_HZ = 20.0
SNAPSHOT_STALE_S = 0.75
MIN_REPLAN_INTERVAL_S = 0.20
REPLAN_TIME_MARGIN_S = 1.00
MIN_PROGRESS_FRACTION = 0.25
MIN_PROGRESS_TIME_S = 0.30
AUTO_REPLAN = True
ALLOW_MANUAL_TRIGGER = True


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


class ContinuousReplanner(Node):
    def __init__(self):
        super().__init__("continuous_replanner")

        self.latest_snapshot: SnapshotData | None = None
        self.active_path: ActivePathState | None = None
        self.replan_requested = False
        self.planning_busy = False
        self.last_plan_start_sec = -1e9
        self.last_plan_finish_sec = -1e9
        self.wp_ready = False
        self.wp_device = None

        self.sub = self.create_subscription(
            PointCloud2,
            THERMAL_TOPIC,
            self.pointcloud_callback,
            10,
        )

        self.path_pub = self.create_publisher(ProjectedHeaterPath, PUBLISH_TOPIC, 10)
        self.busy_pub = self.create_publisher(Bool, PLANNER_STATUS_TOPIC, 10)

        if ALLOW_MANUAL_TRIGGER:
            self.trigger_sub = self.create_subscription(
                Bool,
                MANUAL_TRIGGER_TOPIC,
                self.trigger_callback,
                10,
            )
        else:
            self.trigger_sub = None

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(1.0 / MAIN_LOOP_HZ, self.main_loop)
        self.status_timer = self.create_timer(0.1, self.publish_busy_status)

    # --------------------------------------------------------
    # ROS callbacks
    # --------------------------------------------------------

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

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self.latest_snapshot = SnapshotData(
            points_xyz=np.asarray(pts, dtype=np.float32),
            temps_C=np.asarray(temps, dtype=np.float32),
            frame_id=msg.header.frame_id,
            stamp_sec=now_sec,
        )

    def trigger_callback(self, msg: Bool):
        if msg.data:
            self.replan_requested = True

    # --------------------------------------------------------
    # Main policy
    # --------------------------------------------------------

    def main_loop(self):
        if self.planning_busy:
            return

        if self.latest_snapshot is None:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9

        if not self.should_plan(now_sec):
            return

        self.planning_busy = True
        self.last_plan_start_sec = now_sec

        try:
            t_loop = time.perf_counter()
            now_sec = self.get_clock().now().nanoseconds * 1e-9

            snapshot = self.latest_snapshot
            snapshot_age = float("nan")
            if snapshot is not None:
                snapshot_age = now_sec - snapshot.stamp_sec

            plan_msg, estimated_duration_sec = self.compute_plan_from_latest_snapshot()
            self.path_pub.publish(plan_msg)
            self.active_path = ActivePathState(
                msg=plan_msg,
                publish_time_sec=self.get_clock().now().nanoseconds * 1e-9,
                estimated_duration_sec=estimated_duration_sec,
            )
            self.last_plan_finish_sec = self.active_path.publish_time_sec
            self.replan_requested = False

            self.get_logger().info(
                f"Published replanned path with {len(plan_msg.path_proj)} points, "
                f"est_duration={estimated_duration_sec:.2f}s, "
                f"best_branch={plan_msg.best_branch_index}, cost={plan_msg.final_cost:.6f}"
            )

            t_plan0 = time.perf_counter()
            plan_msg, est_duration = self.compute_plan_from_latest_snapshot()
            t_plan1 = time.perf_counter()

            t_pub0 = time.perf_counter()
            self.path_pub.publish(plan_msg)
            t_pub1 = time.perf_counter()

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

    def should_plan(self, now_sec: float) -> bool:
        snapshot = self.latest_snapshot
        if snapshot is None:
            return False

        if now_sec - self.last_plan_start_sec < MIN_REPLAN_INTERVAL_S:
            return False

        if self.active_path is None:
            return self.snapshot_is_fresh(snapshot, now_sec)

        if self.replan_requested and self.snapshot_is_fresh(snapshot, now_sec):
            return True

        if not AUTO_REPLAN:
            return False

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

        if remaining_sec <= REPLAN_TIME_MARGIN_S and self.snapshot_is_fresh(snapshot, now_sec):
            return True

        if elapsed_sec >= self.active_path.estimated_duration_sec and self.snapshot_is_fresh(snapshot, now_sec):
            return True

        return False

    def snapshot_is_fresh(self, snapshot: SnapshotData, now_sec: float) -> bool:
        if now_sec - snapshot.stamp_sec > SNAPSHOT_STALE_S:
            return False

        if self.active_path is None:
            return True

        return snapshot.stamp_sec >= self.active_path.publish_time_sec

    # --------------------------------------------------------
    # Planning
    # --------------------------------------------------------

    def ensure_warp(self):
        if self.wp_ready:
            return
        wp.init()
        self.wp_device = wp.get_preferred_device()
        self.wp_ready = True
        self.get_logger().info(f"Warp ready on device: {self.wp_device}")

    def get_current_tool_position(
        self,
        target_frame: str,
        child_frame: str,
        timeout_sec: float = 0.20,
    ) -> np.ndarray | None:
        deadline = time.perf_counter() + timeout_sec

        while time.perf_counter() < deadline and rclpy.ok():
            try:
                if self.tf_buffer.can_transform(
                    target_frame,
                    child_frame,
                    rclpy.time.Time(),
                ):
                    t = self.tf_buffer.lookup_transform(
                        target_frame,
                        child_frame,
                        rclpy.time.Time(),
                    )
                    return np.array(
                        [
                            t.transform.translation.x,
                            t.transform.translation.y,
                            t.transform.translation.z,
                        ],
                        dtype=np.float32,
                    )
            except TransformException:
                pass

            time.sleep(0.01)

        self.get_logger().warn(
            f"Could not transform {child_frame} into {target_frame} within {timeout_sec:.2f}s"
        )
        return None

    def compute_plan_from_latest_snapshot(self) -> tuple[ProjectedHeaterPath, float]:

        t0 = time.perf_counter()

        self.ensure_warp()

        snapshot = self.latest_snapshot
        if snapshot is None:
            raise RuntimeError("No thermal snapshot available.")

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
        
        # after snapshot extraction / rasterization / prep
        t1 = time.perf_counter()

        tool_xyz = self.get_current_tool_position(frame_id, EEF_FRAME)
        if tool_xyz is not None:
            P0 = single_plan.choose_start_cell_from_robot(
                tool_xyz=tool_xyz,
                center=proj["center"],
                axis_u=proj["axis_u"],
                axis_v=proj["axis_v"],
                x_min=x_min,
                y_min=y_min,
                L=L,
                valid_mask=valid_mask,
            )
        else:
            if single_plan.USE_FALLBACK_START:
                P0 = single_plan.choose_start_cell(valid_mask)
                self.get_logger().warn(
                    f"Using fallback projected start because TF for {EEF_FRAME} was unavailable."
                )
            else:
                raise RuntimeError(
                    f"Could not get current robot pose for frame '{EEF_FRAME}' in '{frame_id}'."
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
            T0_K.reshape(-1),
            dtype=wp.float32,
            device=self.wp_device,
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
            f"Planning from P0=({P0[0]:.1f}, {P0[1]:.1f}) with {len(targets_cells)} branches"
        )
        for k, ((ix, iy), p, tC) in enumerate(zip(targets_ixiy, targets_cells, targets_T)):
            dxm = (p[0] - P0[0]) * L
            dym = (p[1] - P0[1]) * L
            dist_m = float(np.sqrt(dxm * dxm + dym * dym))
            self.get_logger().info(
                f"  branch {k + 1}: (ix,iy)=({ix},{iy}) T={tC:.2f}C dist={dist_m:.3f}m"
            )

        # after get_current_tool_position(...) and start-cell selection
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
                    "spline_opt": spline_opt_b[k],
                    "J_final": float(J_final_b[k]),
                    "J_hist": J_hist_b[k],
                }
            )

        best_idx = int(np.argmax([b["J_final"] for b in branch_results]))
        best = branch_results[best_idx]

        best_spline_dense_cells = single_plan.evaluate_spline_dense(
            best["ctrl_opt"],
            degree=3,
            num=50,
        )
        best_spline_dense_proj = single_plan.cells_to_proj_points(
            best_spline_dense_cells,
            x_min,
            y_min,
            L,
        )
        ctrl_proj = single_plan.cells_to_proj_points(best["ctrl_opt"], x_min, y_min, L)
        target_proj = single_plan.cells_to_proj_points(best["P4"][None, :], x_min, y_min, L)[0]
        start_proj = single_plan.cells_to_proj_points(P0[None, :], x_min, y_min, L)[0]

        # after branch generation + optimization
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

        msg.center = self.make_point_xyz(*proj["center"])
        msg.axis_u = self.make_vector3_xyz(*proj["axis_u"])
        msg.axis_v = self.make_vector3_xyz(*proj["axis_v"])
        msg.axis_n = self.make_vector3_xyz(*proj["axis_n"])

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
        
        # right before return
        t4 = time.perf_counter()

        self.get_logger().info(
            f"[TIMING replanner_detail] prep={t1-t0:.3f}s "
            f"start_pose={t2-t1:.3f}s "
            f"optimize={t3-t2:.3f}s "
            f"build_msg={t4-t3:.3f}s "
            f"total={t4-t0:.3f}s"
        )

        return msg, estimated_duration_sec

    # --------------------------------------------------------
    # Helpers
    # --------------------------------------------------------

    def estimate_duration_from_path(self, path_proj: np.ndarray) -> float:
        if path_proj.shape[0] < 2:
            return 0.0
        diffs = np.diff(path_proj[:, :2], axis=0)
        total_len_m = float(np.sum(np.linalg.norm(diffs, axis=1)))
        return total_len_m / max(V_MPS, 1e-6)

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

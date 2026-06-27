#!/usr/bin/env python3
# pyright: reportInvalidTypeForm=false

import time
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Point, Vector3
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from thermal_camera_interfaces.msg import ProjectedHeaterPath

import tf2_ros
from tf2_ros import TransformException

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


# ============================================================
# User settings
# ============================================================

THERMAL_TOPIC = "/selected_surface_points"
OUTPUT_TOPIC = "/heater_path_projected"

EEF_FRAME = "heat_gun"
THERMAL_SCALE = 1000.0

GRID_SPACING_M = 0.020
MIN_POINTS_FOR_SNAPSHOT = 100

# ====== normal settings ======
# Path generation
PATH_SAMPLE_SPACING_M = 0.008
MIN_PATH_POINTS = 8

# Replan behavior
CONTROL_RATE_HZ = 10.0
REPLAN_TRIGGER_DIST_M = 0.12
TARGET_REACHED_DIST_M = 0.1
PUBLISH_COOLDOWN_S = 0.8
MIN_TARGET_SEPARATION_M = 0.080

# Prevent flip-flopping between a few cells
VISITED_KEEP_COUNT = 1
VISITED_EXCLUSION_M = 0.1

# Optional target hysteresis
TARGET_TEMP_RISE_KEEP_C = 0.0

MIN_NEXT_TARGET_DIST_M = 0.2
# ====== normal settings ======

# ====== local settings ======
# MIN_POINTS_FOR_SNAPSHOT = 10

# PATH_SAMPLE_SPACING_M = 0.006
# MIN_PATH_POINTS = 4

# CONTROL_RATE_HZ = 5.0
# REPLAN_TRIGGER_DIST_M = 0.04
# TARGET_REACHED_DIST_M = 0.025
# PUBLISH_COOLDOWN_S = 1.0

# MIN_TARGET_SEPARATION_M = 0.025
# MIN_NEXT_TARGET_DIST_M = 0.04

# VISITED_KEEP_COUNT = 0
# VISITED_EXCLUSION_M = 0.03

# TARGET_TEMP_RISE_KEEP_C = 0.0
# ====== local settings ======

# Optional fallback if TF is unavailable
USE_FALLBACK_START = True
START_FRAC_X = 0.18
START_FRAC_Y = 0.18


# ============================================================
# Small helpers
# ============================================================

def make_point_xyz(x: float, y: float, z: float) -> Point:
    p = Point()
    p.x = float(x)
    p.y = float(y)
    p.z = float(z)
    return p


def make_vector3_xyz(x: float, y: float, z: float) -> Vector3:
    v = Vector3()
    v.x = float(x)
    v.y = float(y)
    v.z = float(z)
    return v


def build_projection_frame(points_xyz: np.ndarray):
    center = points_xyz.mean(axis=0)
    centered = points_xyz - center[None, :]
    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    axis_u = vt[0].astype(np.float32)
    axis_v = vt[1].astype(np.float32)
    axis_n = vt[2].astype(np.float32)

    return center.astype(np.float32), axis_u, axis_v, axis_n


def project_to_plane(points_xyz: np.ndarray, center, axis_u, axis_v):
    q = points_xyz - center[None, :]
    u = q @ axis_u
    v = q @ axis_v
    return u.astype(np.float32), v.astype(np.float32)


def project_world_point_to_plane(point_xyz: np.ndarray, center, axis_u, axis_v):
    q = point_xyz.astype(np.float32) - center.astype(np.float32)
    u = float(np.dot(q, axis_u))
    v = float(np.dot(q, axis_v))
    return u, v


def nearest_fill_grid(values: np.ndarray, valid_mask: np.ndarray):
    out = values.copy()

    if np.all(valid_mask):
        return out

    valid_iy, valid_ix = np.nonzero(valid_mask)
    bad_iy, bad_ix = np.nonzero(~valid_mask)

    if len(valid_ix) == 0:
        raise RuntimeError("No valid projected grid cells.")

    if cKDTree is not None:
        tree = cKDTree(np.stack([valid_iy, valid_ix], axis=1).astype(np.float32))
        _, nn = tree.query(np.stack([bad_iy, bad_ix], axis=1).astype(np.float32), k=1)
        out[bad_iy, bad_ix] = values[valid_iy[nn], valid_ix[nn]]
        return out

    for iy, ix in zip(bad_iy, bad_ix):
        d2 = (valid_iy - iy) ** 2 + (valid_ix - ix) ** 2
        j = int(np.argmin(d2))
        out[iy, ix] = values[valid_iy[j], valid_ix[j]]

    return out


def rasterize_snapshot(points_xyz: np.ndarray, temps_C: np.ndarray, grid_spacing_m: float):
    center, axis_u, axis_v, axis_n = build_projection_frame(points_xyz)
    u, v = project_to_plane(points_xyz, center, axis_u, axis_v)

    u_min = float(np.min(u))
    u_max = float(np.max(u))
    v_min = float(np.min(v))
    v_max = float(np.max(v))

    Nx = int(np.ceil((u_max - u_min) / grid_spacing_m)) + 1
    Ny = int(np.ceil((v_max - v_min) / grid_spacing_m)) + 1

    if Nx < 3 or Ny < 3:
        raise RuntimeError(f"Projected grid too small: Nx={Nx}, Ny={Ny}")

    ix = np.clip(np.round((u - u_min) / grid_spacing_m).astype(np.int32), 0, Nx - 1)
    iy = np.clip(np.round((v - v_min) / grid_spacing_m).astype(np.int32), 0, Ny - 1)

    temp_sum = np.zeros((Ny, Nx), dtype=np.float32)
    temp_count = np.zeros((Ny, Nx), dtype=np.int32)

    np.add.at(temp_sum, (iy, ix), temps_C)
    np.add.at(temp_count, (iy, ix), 1)

    valid_mask = temp_count > 0

    T0_C_raw = np.full((Ny, Nx), np.nan, dtype=np.float32)
    T0_C_raw[valid_mask] = temp_sum[valid_mask] / temp_count[valid_mask]

    T0_C_filled = np.zeros((Ny, Nx), dtype=np.float32)
    T0_C_filled[valid_mask] = T0_C_raw[valid_mask]
    T0_C_filled = nearest_fill_grid(T0_C_filled, valid_mask)

    x_min = u_min
    x_max = u_min + (Nx - 1) * grid_spacing_m
    y_min = v_min
    y_max = v_min + (Ny - 1) * grid_spacing_m

    return {
        "T0_C_raw": T0_C_raw,
        "T0_C_filled": T0_C_filled,
        "valid_mask": valid_mask.astype(np.uint8),
        "Nx": Nx,
        "Ny": Ny,
        "L": float(grid_spacing_m),
        "x_min": float(x_min),
        "x_max": float(x_max),
        "y_min": float(y_min),
        "y_max": float(y_max),
        "center": center,
        "axis_u": axis_u,
        "axis_v": axis_v,
        "axis_n": axis_n,
    }


def choose_start_cell(valid_mask: np.ndarray) -> np.ndarray:
    Ny, Nx = valid_mask.shape
    x0 = START_FRAC_X * (Nx - 1)
    y0 = START_FRAC_Y * (Ny - 1)

    ys, xs = np.nonzero(valid_mask.astype(bool))
    if len(xs) == 0:
        raise RuntimeError("No valid cells available for start selection.")

    d2 = (xs.astype(np.float32) - x0) ** 2 + (ys.astype(np.float32) - y0) ** 2
    j = int(np.argmin(d2))
    return np.array([float(xs[j]), float(ys[j])], dtype=np.float32)


def snap_cell_to_valid(cell_xy: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(valid_mask.astype(bool))
    if len(xs) == 0:
        raise RuntimeError("No valid cells available.")

    dx = xs.astype(np.float32) - float(cell_xy[0])
    dy = ys.astype(np.float32) - float(cell_xy[1])
    j = int(np.argmin(dx * dx + dy * dy))
    return np.array([float(xs[j]), float(ys[j])], dtype=np.float32)


def choose_start_cell_from_robot(
    tool_xyz: np.ndarray,
    center: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
    x_min: float,
    y_min: float,
    L: float,
    valid_mask: np.ndarray,
) -> np.ndarray:
    u, v = project_world_point_to_plane(tool_xyz, center, axis_u, axis_v)
    cx = (u - x_min) / L
    cy = (v - y_min) / L
    return snap_cell_to_valid(np.array([cx, cy], dtype=np.float32), valid_mask)


def cells_to_proj_points(cells_xy: np.ndarray, x_min: float, y_min: float, L: float) -> np.ndarray:
    out = np.zeros((len(cells_xy), 2), dtype=np.float32)
    out[:, 0] = x_min + cells_xy[:, 0] * L
    out[:, 1] = y_min + cells_xy[:, 1] * L
    return out


def sample_straight_line_cells(
    start_cell: np.ndarray,
    target_cell: np.ndarray,
    L: float,
    spacing_m: float,
    min_points: int,
) -> np.ndarray:
    diff = target_cell - start_cell
    dist_cells = float(np.linalg.norm(diff))
    dist_m = dist_cells * L

    if dist_m < 1e-9:
        return np.stack([start_cell, target_cell], axis=0).astype(np.float32)

    num = max(min_points, int(np.ceil(dist_m / spacing_m)) + 1)
    t = np.linspace(0.0, 1.0, num, dtype=np.float32)[:, None]
    pts = (1.0 - t) * start_cell[None, :] + t * target_cell[None, :]
    return pts.astype(np.float32)


def snap_path_to_valid(path_cells: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(valid_mask.astype(bool))
    if len(xs) == 0:
        raise RuntimeError("No valid cells available.")

    occ = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    out = np.zeros_like(path_cells, dtype=np.float32)

    if cKDTree is not None:
        tree = cKDTree(occ)
        _, nn = tree.query(path_cells.astype(np.float32), k=1)
        out[:] = occ[np.asarray(nn, dtype=np.int32)]
    else:
        for i, p in enumerate(path_cells):
            d = occ - p[None, :]
            j = int(np.argmin(np.sum(d * d, axis=1)))
            out[i] = occ[j]

    # Remove consecutive duplicates while keeping endpoints
    keep = [0]
    for i in range(1, len(out)):
        if np.linalg.norm(out[i] - out[keep[-1]]) > 1e-6:
            keep.append(i)

    if keep[-1] != len(out) - 1:
        keep.append(len(out) - 1)

    return out[np.asarray(keep, dtype=np.int32)]


# ============================================================
# Main node
# ============================================================

class GreedyProjectedPolicyNode(Node):
    def __init__(self):
        super().__init__("greedy_projected_policy")

        self.points_xyz = None
        self.temps_C = None
        self.frame_id = None
        self.last_snapshot_time = 0.0

        self.path_pub = self.create_publisher(ProjectedHeaterPath, OUTPUT_TOPIC, 10)
        self.sub = self.create_subscription(
            PointCloud2,
            THERMAL_TOPIC,
            self.pointcloud_callback,
            10,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.timer = self.create_timer(1.0 / CONTROL_RATE_HZ, self.control_loop)

        self.active_target_cell = None
        self.active_target_temp_C = None
        self.active_projection = None
        self.last_publish_time = 0.0
        self.visited_targets = deque(maxlen=VISITED_KEEP_COUNT)

        self.get_logger().info(f"Subscribing to {THERMAL_TOPIC}")
        self.get_logger().info(f"Publishing projected paths to {OUTPUT_TOPIC}")

    # --------------------------------------------------------
    # Input
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

        self.points_xyz = np.asarray(pts, dtype=np.float32)
        self.temps_C = np.asarray(temps, dtype=np.float32)
        self.frame_id = msg.header.frame_id
        self.last_snapshot_time = time.time()

    # --------------------------------------------------------
    # TF
    # --------------------------------------------------------

    def get_current_tool_position(self, target_frame: str, child_frame: str) -> np.ndarray | None:
        try:
            if not self.tf_buffer.can_transform(
                target_frame,
                child_frame,
                rclpy.time.Time(),
            ):
                return None

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
            return None

    # --------------------------------------------------------
    # Target choice
    # --------------------------------------------------------

    def pick_target_cell(
        self,
        T_raw_C: np.ndarray,
        valid_mask: np.ndarray,
        start_cell: np.ndarray,
        L: float,
    ) -> tuple[np.ndarray, float]:
        mask = valid_mask.astype(bool).copy()

        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            raise RuntimeError("No valid cells in projected footprint.")

        dx0 = (xs.astype(np.float32) - float(start_cell[0])) * L
        dy0 = (ys.astype(np.float32) - float(start_cell[1])) * L
        # keep0 = np.sqrt(dx0 * dx0 + dy0 * dy0) >= MIN_TARGET_SEPARATION_M
        min_dist_m = max(MIN_TARGET_SEPARATION_M, MIN_NEXT_TARGET_DIST_M)
        keep0 = np.sqrt(dx0 * dx0 + dy0 * dy0) >= min_dist_m

        if np.any(keep0):
            mask[:] = False
            mask[ys[keep0], xs[keep0]] = True
        else:
            mask = valid_mask.astype(bool).copy()

        # Exclude recently visited targets
        if len(self.visited_targets) > 0:
            ys2, xs2 = np.nonzero(mask)
            keep_recent = np.ones_like(xs2, dtype=bool)
            for vx, vy in self.visited_targets:
                dx = (xs2.astype(np.float32) - float(vx)) * L
                dy = (ys2.astype(np.float32) - float(vy)) * L
                keep_recent &= (np.sqrt(dx * dx + dy * dy) >= VISITED_EXCLUSION_M)
            if np.any(keep_recent):
                new_mask = np.zeros_like(mask, dtype=bool)
                new_mask[ys2[keep_recent], xs2[keep_recent]] = True
                mask = new_mask

        vals = np.where(mask, T_raw_C, np.inf).astype(np.float32)
        iy, ix = np.unravel_index(int(np.argmin(vals)), vals.shape)

        if not np.isfinite(vals[iy, ix]):
            vals = np.where(valid_mask.astype(bool), T_raw_C, np.inf).astype(np.float32)
            iy, ix = np.unravel_index(int(np.argmin(vals)), vals.shape)

        return np.array([float(ix), float(iy)], dtype=np.float32), float(T_raw_C[iy, ix])

    # --------------------------------------------------------
    # Replan policy
    # --------------------------------------------------------

    def should_replan(
        self,
        current_cell: np.ndarray,
        current_target_cell: np.ndarray | None,
        current_target_temp_C: float | None,
        T_raw_C: np.ndarray,
        L: float,
    ) -> bool:
        if current_target_cell is None:
            return True

        dx = (float(current_cell[0]) - float(current_target_cell[0])) * L
        dy = (float(current_cell[1]) - float(current_target_cell[1])) * L
        dist_m = float(np.sqrt(dx * dx + dy * dy))

        if dist_m <= REPLAN_TRIGGER_DIST_M:
            return True

        iy = int(round(float(current_target_cell[1])))
        ix = int(round(float(current_target_cell[0])))
        iy = int(np.clip(iy, 0, T_raw_C.shape[0] - 1))
        ix = int(np.clip(ix, 0, T_raw_C.shape[1] - 1))
        latest_target_temp = float(T_raw_C[iy, ix])

        if current_target_temp_C is not None:
            if latest_target_temp >= current_target_temp_C + TARGET_TEMP_RISE_KEEP_C:
                return True

        return False

    # --------------------------------------------------------
    # Publish
    # --------------------------------------------------------

    def publish_projected_path(
        self,
        proj: dict,
        start_cell: np.ndarray,
        target_cell: np.ndarray,
        path_cells: np.ndarray,
        target_temp_C: float,
    ):
        x_min = float(proj["x_min"])
        y_min = float(proj["y_min"])
        L = float(proj["L"])
        Nx = int(proj["Nx"])
        Ny = int(proj["Ny"])

        start_proj = cells_to_proj_points(start_cell[None, :], x_min, y_min, L)[0]
        target_proj = cells_to_proj_points(target_cell[None, :], x_min, y_min, L)[0]
        path_proj = cells_to_proj_points(path_cells, x_min, y_min, L)

        msg = ProjectedHeaterPath()
        msg.header = Header()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id

        msg.x_min = x_min
        msg.y_min = y_min
        msg.cell_size_m = L
        msg.nx = Nx
        msg.ny = Ny

        msg.center = make_point_xyz(
            proj["center"][0],
            proj["center"][1],
            proj["center"][2],
        )
        msg.axis_u = make_vector3_xyz(
            proj["axis_u"][0],
            proj["axis_u"][1],
            proj["axis_u"][2],
        )
        msg.axis_v = make_vector3_xyz(
            proj["axis_v"][0],
            proj["axis_v"][1],
            proj["axis_v"][2],
        )
        msg.axis_n = make_vector3_xyz(
            proj["axis_n"][0],
            proj["axis_n"][1],
            proj["axis_n"][2],
        )

        msg.start_proj = make_point_xyz(start_proj[0], start_proj[1], 0.0)
        msg.target_proj = make_point_xyz(target_proj[0], target_proj[1], 0.0)

        # Straight line, so control points can just mirror a few key points
        ctrl_cells = np.array(
            [
                start_cell,
                path_cells[min(len(path_cells) - 1, max(1, len(path_cells) // 3))],
                path_cells[min(len(path_cells) - 1, max(2, 2 * len(path_cells) // 3))],
                target_cell,
            ],
            dtype=np.float32,
        )
        ctrl_proj = cells_to_proj_points(ctrl_cells, x_min, y_min, L)

        msg.control_proj = [make_point_xyz(p[0], p[1], 0.0) for p in ctrl_proj]
        msg.path_proj = [make_point_xyz(p[0], p[1], 0.0) for p in path_proj]
        msg.path_cells = [make_point_xyz(p[0], p[1], 0.0) for p in path_cells]

        msg.final_cost = float(target_temp_C)
        msg.best_branch_index = 0

        self.path_pub.publish(msg)
        self.last_publish_time = time.time()

        self.active_target_cell = target_cell.copy()
        self.active_target_temp_C = float(target_temp_C)
        self.active_projection = proj

        self.get_logger().info(
            f"Published greedy path: start=({start_cell[0]:.1f},{start_cell[1]:.1f}) "
            f"target=({target_cell[0]:.1f},{target_cell[1]:.1f}) "
            f"T_target={target_temp_C:.2f}C "
            f"pts={len(path_cells)}"
        )

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    def control_loop(self):
        if self.points_xyz is None or self.temps_C is None or self.frame_id is None:
            return

        now = time.time()
        if (now - self.last_publish_time) < PUBLISH_COOLDOWN_S:
            return

        try:
            proj = rasterize_snapshot(
                points_xyz=self.points_xyz,
                temps_C=self.temps_C,
                grid_spacing_m=GRID_SPACING_M,
            )
        except Exception as exc:
            self.get_logger().warn(f"Projection failed: {exc}")
            return

        valid_mask = proj["valid_mask"]
        T_raw_C = proj["T0_C_raw"]
        L = float(proj["L"])

        tool_xyz = self.get_current_tool_position(self.frame_id, EEF_FRAME)

        if tool_xyz is not None:
            start_cell = choose_start_cell_from_robot(
                tool_xyz=tool_xyz,
                center=proj["center"],
                axis_u=proj["axis_u"],
                axis_v=proj["axis_v"],
                x_min=proj["x_min"],
                y_min=proj["y_min"],
                L=L,
                valid_mask=valid_mask,
            )
        else:
            if not USE_FALLBACK_START:
                self.get_logger().warn(
                    f"TF unavailable for {EEF_FRAME} in frame {self.frame_id}. Skipping cycle."
                )
                return
            start_cell = choose_start_cell(valid_mask)

        if not self.should_replan(
            current_cell=start_cell,
            current_target_cell=self.active_target_cell,
            current_target_temp_C=self.active_target_temp_C,
            T_raw_C=T_raw_C,
            L=L,
        ):
            return

        target_cell, target_temp_C = self.pick_target_cell(
            T_raw_C=T_raw_C,
            valid_mask=valid_mask,
            start_cell=start_cell,
            L=L,
        )

        dx = (float(start_cell[0]) - float(target_cell[0])) * L
        dy = (float(start_cell[1]) - float(target_cell[1])) * L
        target_dist_m = float(np.sqrt(dx * dx + dy * dy))

        # if target_dist_m <= TARGET_REACHED_DIST_M:
        #     self.visited_targets.append((float(target_cell[0]), float(target_cell[1])))
        #     return
        if target_dist_m <= TARGET_REACHED_DIST_M:
            self.visited_targets.append((float(target_cell[0]), float(target_cell[1])))

            self.active_target_cell = None
            self.active_target_temp_C = None
            self.active_projection = None

            self.get_logger().warn(
                f"Selected target too close: dist={target_dist_m:.3f} m. "
                "Clearing active target and retrying next cycle."
            )
            return

        raw_path_cells = sample_straight_line_cells(
            start_cell=start_cell,
            target_cell=target_cell,
            L=L,
            spacing_m=PATH_SAMPLE_SPACING_M,
            min_points=MIN_PATH_POINTS,
        )
        path_cells = snap_path_to_valid(raw_path_cells, valid_mask)

        if len(path_cells) < 2:
            self.get_logger().warn("Generated path is too short after footprint snapping.")
            return

        if self.active_target_cell is not None:
            dx_prev = (float(start_cell[0]) - float(self.active_target_cell[0])) * L
            dy_prev = (float(start_cell[1]) - float(self.active_target_cell[1])) * L
            prev_dist_m = float(np.sqrt(dx_prev * dx_prev + dy_prev * dy_prev))
            if prev_dist_m <= TARGET_REACHED_DIST_M:
                self.visited_targets.append(
                    (float(self.active_target_cell[0]), float(self.active_target_cell[1]))
                )

        self.publish_projected_path(
            proj=proj,
            start_cell=start_cell,
            target_cell=target_cell,
            path_cells=path_cells,
            target_temp_C=target_temp_C,
        )


def main(args=None):
    rclpy.init(args=args)
    node = GreedyProjectedPolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
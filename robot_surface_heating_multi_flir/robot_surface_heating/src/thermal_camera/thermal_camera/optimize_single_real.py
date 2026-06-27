#!/usr/bin/env python3
# pyright: reportInvalidTypeForm=false

import time
import numpy as np
import warp as wp
import matplotlib.pyplot as plt
import rclpy

from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
import tf2_ros
from tf2_ros import TransformException

from std_msgs.msg import Header
from geometry_msgs.msg import Point, Vector3
from thermal_camera_interfaces.msg import ProjectedHeaterPath

import thermal_camera.adhesive_single as bb

import json
from std_msgs.msg import String

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


# ============================================================
# User settings
# ============================================================

# SHOW_PLOT = False
SHOW_PLOT = True

# THERMAL_TOPIC = "/raw_thermal_pointcloud"
THERMAL_TOPIC = "/selected_surface_points"

THERMAL_SCALE = 1000.0

GRID_SPACING_M = 0.020

# # --- normal settings ---
# NUM_BRANCHES = 10
# R_EXCL_START_M = 0.2
# R_EXCL_TARGETS_M = 0.05
# # --- normal settings ---

# --- local settings ---
R_EXCL_START_M = 0.08
R_EXCL_TARGETS_M = 0.02
NUM_BRANCHES = 3
# --- local settings ---

V_MPS = 0.08
DT = 0.25
T_TOTAL = 4.0

HOT_PERCENTILE = 80.0
SCALE_K = 0.75

W_CURV = 5.0
W_HEAT = 1.0
# W_SCREEN = 0.0
W_SCREEN = 100.0

ITERS = 200
LR = 0.01

# Fixed heater parameters
SIGMA_M = 0.025
H_PEAK = 320.0

MIN_POINTS_FOR_SNAPSHOT = 20

# Same style as projection_composite_single:
# use a nominal start in projected cell coordinates, then snap to nearest valid cell
# Current robot tool frame used as optimization start
EEF_FRAME = "heat_gun"

# Optional fallback if TF is unavailable
# USE_FALLBACK_START = True
USE_FALLBACK_START = False
START_FRAC_X = 0.18
START_FRAC_Y = 0.18

# Reject candidate targets that are too close to hot cells
USE_HOT_TARGET_CLEARANCE = False
HOT_TARGET_CLEARANCE_M = 0.08

# Optional fallback if the clearance mask becomes too restrictive
HOT_TARGET_CLEARANCE_RELAX_FRACTION = 0.5


SELECTED_SURFACE_BUFFER_M = 0.015
OFF_SURFACE_FALLOFF_M = 0.040

# ============================================================
# ROS snapshot node
# ============================================================

class ThermalSnapshotNode(Node):
    def __init__(self):
        super().__init__("optimize_single_real")

        self.points_xyz = None
        self.temps_C = None
        self.frame_id = None
        self.have_snapshot = False

        self.sub = self.create_subscription(
            PointCloud2,
            THERMAL_TOPIC,
            self.pointcloud_callback,
            10,
        )

        self.path_pub = self.create_publisher(
            ProjectedHeaterPath,
            "/heater_path_projected",
            10,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def pointcloud_callback(self, msg: PointCloud2):
        pts = []
        temps = []

        # for p in point_cloud2.read_points(
        #     msg,
        #     field_names=("x", "y", "z", "thermal"),
        #     skip_nans=True,
        # ):
        #     x, y, z, t = p

        #     if not np.isfinite(t):
        #         continue

        #     pts.append([x, y, z])
        #     temps.append(float(t) / THERMAL_SCALE)
        for p in point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z", "thermal"),
            skip_nans=True,
        ):
            x, y, z, t = p

            if not np.isfinite(t):
                continue

            t = float(t)

            # Robust conversion to Celsius
            if t > 1.0e4:
                # likely Kelvin * 100, e.g. 27315 -> 0 C
                t_c = (t - 27315.0) / 100.0
            elif t > 200.0:
                # likely Kelvin
                t_c = t - 273.15
            else:
                # likely already Celsius
                t_c = t

            pts.append([x, y, z])
            temps.append(t_c)

        if len(pts) < MIN_POINTS_FOR_SNAPSHOT:
            return

        self.points_xyz = np.asarray(pts, dtype=np.float32)
        self.temps_C = np.asarray(temps, dtype=np.float32)
        self.frame_id = msg.header.frame_id
        self.have_snapshot = True


def wait_for_snapshot():
    rclpy.init()
    node = ThermalSnapshotNode()

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.have_snapshot:
                return (
                    node,
                    node.points_xyz.copy(),
                    node.temps_C.copy(),
                    node.frame_id,
                )
    except KeyboardInterrupt:
        pass

    raise RuntimeError("No ROS thermal pointcloud snapshot received.")

# def get_current_tool_position(node: ThermalSnapshotNode, target_frame: str, child_frame: str) -> np.ndarray | None:
#     try:
#         t = node.tf_buffer.lookup_transform(
#             target_frame,
#             child_frame,
#             rclpy.time.Time()
#         )
#         return np.array([
#             t.transform.translation.x,
#             t.transform.translation.y,
#             t.transform.translation.z,
#         ], dtype=np.float32)
#     except TransformException as ex:
#         node.get_logger().warn(
#             f"Could not transform {child_frame} into {target_frame}: {ex}"
#         )
#         return None
def get_current_tool_position(
    node: ThermalSnapshotNode,
    target_frame: str,
    child_frame: str,
    timeout_sec: float = 2.0,
) -> np.ndarray | None:
    start = time.time()

    while rclpy.ok() and (time.time() - start) < timeout_sec:
        rclpy.spin_once(node, timeout_sec=0.05)

        try:
            if node.tf_buffer.can_transform(
                target_frame,
                child_frame,
                rclpy.time.Time(),
            ):
                t = node.tf_buffer.lookup_transform(
                    target_frame,
                    child_frame,
                    rclpy.time.Time(),
                )
                return np.array([
                    t.transform.translation.x,
                    t.transform.translation.y,
                    t.transform.translation.z,
                ], dtype=np.float32)
        except TransformException:
            pass

    try:
        frames_yaml = node.tf_buffer.all_frames_as_yaml()
        node.get_logger().warn(
            f"Could not transform {child_frame} into {target_frame} within {timeout_sec:.1f}s."
        )
        node.get_logger().info(frames_yaml)
    except Exception:
        node.get_logger().warn(
            f"Could not transform {child_frame} into {target_frame} within {timeout_sec:.1f}s."
        )

    return None


# ============================================================
# Projection helpers
# ============================================================

def build_projection_frame(points_xyz: np.ndarray):
    center = points_xyz.mean(axis=0)
    centered = points_xyz - center

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

def build_hot_clearance_allowed_mask(
    T0_C_raw: np.ndarray,
    valid_mask: np.ndarray,
    T_hot_C: float,
    L: float,
    clearance_m: float,
) -> np.ndarray:
    """
    Return a mask of cells that are valid target candidates and are not
    within clearance_m of any hot cell.
    """
    valid_bool = valid_mask.astype(bool)
    hot_mask = valid_bool & np.isfinite(T0_C_raw) & (T0_C_raw >= T_hot_C)

    # If nothing is hot yet, do not restrict anything
    if not np.any(hot_mask):
        return valid_bool.astype(np.uint8)

    ys, xs = np.nonzero(hot_mask)
    hot_cells = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)

    all_ys, all_xs = np.nonzero(valid_bool)
    cand_cells = np.stack([all_xs.astype(np.float32), all_ys.astype(np.float32)], axis=1)

    clearance_cells = float(clearance_m / max(L, 1e-9))
    clearance2_cells = clearance_cells * clearance_cells

    keep = np.ones((cand_cells.shape[0],), dtype=bool)

    if cKDTree is not None:
        tree = cKDTree(hot_cells)
        neighbors = tree.query_ball_point(cand_cells, r=clearance_cells)
        keep = np.array([len(n) == 0 for n in neighbors], dtype=bool)
    else:
        for i, p in enumerate(cand_cells):
            dx = hot_cells[:, 0] - p[0]
            dy = hot_cells[:, 1] - p[1]
            d2 = dx * dx + dy * dy
            if np.any(d2 <= clearance2_cells):
                keep[i] = False

    out = np.zeros_like(valid_bool, dtype=np.uint8)
    kept_cells = cand_cells[keep].astype(np.int32)
    out[kept_cells[:, 1], kept_cells[:, 0]] = 1
    return out

# def rasterize_snapshot(points_xyz: np.ndarray, temps_C: np.ndarray, grid_spacing_m: float):
#     center, axis_u, axis_v, axis_n = build_projection_frame(points_xyz)
#     u, v = project_to_plane(points_xyz, center, axis_u, axis_v)

#     u_min = float(np.min(u))
#     u_max = float(np.max(u))
#     v_min = float(np.min(v))
#     v_max = float(np.max(v))

#     Nx = int(np.ceil((u_max - u_min) / grid_spacing_m)) + 1
#     Ny = int(np.ceil((v_max - v_min) / grid_spacing_m)) + 1

#     if Nx < 3 or Ny < 3:
#         raise RuntimeError(f"Projected grid too small: Nx={Nx}, Ny={Ny}")

#     ix = np.clip(np.round((u - u_min) / grid_spacing_m).astype(np.int32), 0, Nx - 1)
#     iy = np.clip(np.round((v - v_min) / grid_spacing_m).astype(np.int32), 0, Ny - 1)

#     temp_sum = np.zeros((Ny, Nx), dtype=np.float32)
#     temp_count = np.zeros((Ny, Nx), dtype=np.int32)

#     np.add.at(temp_sum, (iy, ix), temps_C)
#     np.add.at(temp_count, (iy, ix), 1)

#     valid_mask = temp_count > 0

#     T0_C = np.zeros((Ny, Nx), dtype=np.float32)
#     T0_C[valid_mask] = temp_sum[valid_mask] / temp_count[valid_mask]
#     T0_C = nearest_fill_grid(T0_C, valid_mask)

#     x_min = u_min
#     x_max = u_min + (Nx - 1) * grid_spacing_m
#     y_min = v_min
#     y_max = v_min + (Ny - 1) * grid_spacing_m

#     return {
#         "T0_C": T0_C,
#         "valid_mask": valid_mask.astype(np.uint8),
#         "Nx": Nx,
#         "Ny": Ny,
#         "L": float(grid_spacing_m),
#         "x_min": float(x_min),
#         "x_max": float(x_max),
#         "y_min": float(y_min),
#         "y_max": float(y_max),
#         "center": center,
#         "axis_u": axis_u,
#         "axis_v": axis_v,
#         "axis_n": axis_n,
#     }
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


def choose_start_cell(valid_mask: np.ndarray):
    Ny, Nx = valid_mask.shape
    x0 = START_FRAC_X * (Nx - 1)
    y0 = START_FRAC_Y * (Ny - 1)

    ys, xs = np.nonzero(valid_mask.astype(bool))
    if len(xs) == 0:
        raise RuntimeError("No valid cells available for start selection.")

    d2 = (xs.astype(np.float32) - x0) ** 2 + (ys.astype(np.float32) - y0) ** 2
    j = int(np.argmin(d2))
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

    ys, xs = np.nonzero(valid_mask.astype(bool))
    if len(xs) == 0:
        raise RuntimeError("No valid cells available for start selection.")

    d2 = (xs.astype(np.float32) - float(cx)) ** 2 + (ys.astype(np.float32) - float(cy)) ** 2
    j = int(np.argmin(d2))

    return np.array([float(xs[j]), float(ys[j])], dtype=np.float32)


# ============================================================
# Dense spline plotting helper
# ============================================================

def evaluate_spline_dense(ctrl, degree=3, num=400):
    knots = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1], dtype=np.float32)
    u0 = float(knots[degree])
    u1 = float(knots[len(ctrl)])
    u_dense = np.linspace(u0, u1, num, dtype=np.float32)

    W_dense = np.stack(
        [bb.bspline_weights_at_u(5, degree, u, knots) for u in u_dense],
        axis=0,
    )
    return (W_dense @ ctrl).astype(np.float32)


def spline_to_world_xy(spline_cells, x_min, y_min, L):
    sx = x_min + spline_cells[:, 0] * L
    sy = y_min + spline_cells[:, 1] * L
    return sx, sy


def make_point_xyz(x, y, z=0.0):
    p = Point()
    p.x = float(x)
    p.y = float(y)
    p.z = float(z)
    return p


def make_vector3_xyz(x, y, z):
    v = Vector3()
    v.x = float(x)
    v.y = float(y)
    v.z = float(z)
    return v


def cells_to_proj_points(path_cells: np.ndarray, x_min: float, y_min: float, L: float):
    pts = []
    for cx, cy in path_cells:
        px = x_min + float(cx) * L
        py = y_min + float(cy) * L
        pts.append([px, py])
    return np.asarray(pts, dtype=np.float32)


def build_projected_surface_buffer_mask(valid_mask, L, buffer_m):
    valid = valid_mask.astype(bool)

    if not np.any(valid):
        return valid.astype(np.uint8)

    ys, xs = np.nonzero(valid)
    valid_cells = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)

    Ny, Nx = valid.shape
    yy, xx = np.mgrid[0:Ny, 0:Nx]
    query_cells = np.stack([xx.ravel().astype(np.float32), yy.ravel().astype(np.float32)], axis=1)

    buffer_cells = float(buffer_m / max(L, 1e-9))

    if cKDTree is not None:
        tree = cKDTree(valid_cells)
        dists, _ = tree.query(query_cells, k=1)
        keep = dists <= buffer_cells
    else:
        keep = np.zeros((query_cells.shape[0],), dtype=bool)
        for i, q in enumerate(query_cells):
            d2 = np.sum((valid_cells - q[None, :]) ** 2, axis=1)
            keep[i] = np.min(d2) <= buffer_cells * buffer_cells

    return keep.reshape(Ny, Nx).astype(np.uint8)


def build_surface_distance_cost(
    valid_mask: np.ndarray,
    L: float,
    buffer_m: float,
    falloff_m: float,
) -> np.ndarray:
    valid = valid_mask.astype(bool)
    Ny, Nx = valid.shape

    if not np.any(valid):
        return np.ones((Ny, Nx), dtype=np.float32)

    ys, xs = np.nonzero(valid)
    valid_cells = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)

    yy, xx = np.mgrid[0:Ny, 0:Nx]
    query_cells = np.stack(
        [xx.ravel().astype(np.float32), yy.ravel().astype(np.float32)],
        axis=1,
    )

    if cKDTree is not None:
        tree = cKDTree(valid_cells)
        dist_cells, _ = tree.query(query_cells, k=1)
    else:
        dist_cells = np.zeros((query_cells.shape[0],), dtype=np.float32)
        for i, q in enumerate(query_cells):
            d2 = np.sum((valid_cells - q[None, :]) ** 2, axis=1)
            dist_cells[i] = np.sqrt(np.min(d2))

    dist_m = dist_cells.reshape(Ny, Nx).astype(np.float32) * float(L)

    excess_m = np.maximum(dist_m - float(buffer_m), 0.0)
    cost = (excess_m / max(float(falloff_m), 1e-9)) ** 2

    return cost.astype(np.float32)

# ============================================================
# Main
# ============================================================

def main():
    node, points_xyz, temps_C_raw, frame_id = wait_for_snapshot()

    try:
        wp.init()
        device = wp.get_preferred_device()

        proj = rasterize_snapshot(
            points_xyz=points_xyz,
            temps_C=temps_C_raw,
            grid_spacing_m=GRID_SPACING_M,
        )

        # T0_C = proj["T0_C"]
        # valid_mask = proj["valid_mask"]
        T0_C_raw = proj["T0_C_raw"]
        T0_C = proj["T0_C_filled"]      # optimization uses filled grid
        valid_mask = proj["valid_mask"] # targets must stay on true occupied footprint
        Nx = proj["Nx"]
        Ny = proj["Ny"]
        L = proj["L"]
        x_min = proj["x_min"]
        x_max = proj["x_max"]
        y_min = proj["y_min"]
        y_max = proj["y_max"]

        # Assign starting point
        # P0 = choose_start_cell(valid_mask)
        tool_xyz = get_current_tool_position(node, frame_id, EEF_FRAME)

        if tool_xyz is not None:
            P0 = choose_start_cell_from_robot(
                tool_xyz=tool_xyz,
                center=proj["center"],
                axis_u=proj["axis_u"],
                axis_v=proj["axis_v"],
                x_min=x_min,
                y_min=y_min,
                L=L,
                valid_mask=valid_mask,
            )
            print(
                f"[INFO] start from current robot pose: "
                f"tool_xyz=({tool_xyz[0]:.4f}, {tool_xyz[1]:.4f}, {tool_xyz[2]:.4f}) "
                f"-> P0=({P0[0]:.1f}, {P0[1]:.1f})"
            )
        else:
            if USE_FALLBACK_START:
                P0 = choose_start_cell(valid_mask)
                print(
                    f"[WARN] Falling back to nominal start because TF for {EEF_FRAME} was unavailable."
                )
            else:
                raise RuntimeError(
                    f"Could not get current robot pose for frame '{EEF_FRAME}' in frame '{frame_id}'."
                )

        # allowed_mask = valid_mask.copy()

        # # T_valid = T0_C[allowed_mask.astype(bool)]
        # T_valid = T0_C_raw[allowed_mask.astype(bool)]
        # T_valid = T_valid[np.isfinite(T_valid)]
        # if T_valid.size == 0:
        #     raise RuntimeError("No valid projected temperatures available.")
        base_allowed_mask = valid_mask.copy()

        T_valid = T0_C_raw[base_allowed_mask.astype(bool)]
        T_valid = T_valid[np.isfinite(T_valid)]
        if T_valid.size == 0:
            raise RuntimeError("No valid projected temperatures available.")

        T0_K = (T0_C + 273.15).astype(np.float32)
        T_hot_C = float(np.percentile(T_valid, HOT_PERCENTILE))
        T_hot_K = float(T_hot_C + 273.15)
        frac_hot = float(np.mean(T_valid >= T_hot_C))
        inv_scale = 1.0 / SCALE_K

        if USE_HOT_TARGET_CLEARANCE:
            allowed_mask = build_hot_clearance_allowed_mask(
                T0_C_raw=T0_C_raw,
                valid_mask=valid_mask,
                T_hot_C=T_hot_C,
                L=L,
                clearance_m=HOT_TARGET_CLEARANCE_M,
            )

            num_allowed = int(np.sum(allowed_mask))
            if num_allowed == 0:
                relaxed_clearance = HOT_TARGET_CLEARANCE_M * HOT_TARGET_CLEARANCE_RELAX_FRACTION
                print(
                    f"[WARN] Hot-clearance mask removed all candidate targets. "
                    f"Retrying with relaxed clearance {relaxed_clearance:.3f} m."
                )
                allowed_mask = build_hot_clearance_allowed_mask(
                    T0_C_raw=T0_C_raw,
                    valid_mask=valid_mask,
                    T_hot_C=T_hot_C,
                    L=L,
                    clearance_m=relaxed_clearance,
                )

                num_allowed = int(np.sum(allowed_mask))
                if num_allowed == 0:
                    print("[WARN] Relaxed hot-clearance still removed all targets. Falling back to valid_mask.")
                    allowed_mask = valid_mask.copy()

            print(
                f"[INFO] hot target clearance: radius={HOT_TARGET_CLEARANCE_M:.3f} m "
                f"allowed_targets={int(np.sum(allowed_mask))}"
            )
        else:
            allowed_mask = valid_mask.copy()

        print(
            f"[INFO] ROS frame={frame_id} points={points_xyz.shape[0]} "
            f"grid: Nx={Nx} Ny={Ny} L={L:.5f} dt={DT} H={int(round(T_TOTAL / DT))} "
            f"v={V_MPS:.3f} m/s T={T_TOTAL:.2f} s"
        )
        print(
            f"[INFO] T0_C: min={float(np.min(T_valid)):.2f} "
            f"max={float(np.max(T_valid)):.2f} "
            f"mean={float(np.mean(T_valid)):.2f} | "
            f"hot%={100.0 * frac_hot:.2f}% (>=p{HOT_PERCENTILE:.0f}={T_hot_C:.2f}C)"
        )
        print(
            f"[INFO] heater params: sigma_m={SIGMA_M:.5f} h_peak={H_PEAK:.6g}"
        )

        # T0_dev_2d = wp.array(T0_K.reshape(-1), dtype=wp.float32, device=device).reshape((Ny, Nx))
        # mask_screen_dev = wp.zeros((Ny, Nx), dtype=wp.uint8, device=device)
        T0_dev_2d = wp.array(T0_K.reshape(-1), dtype=wp.float32, device=device).reshape((Ny, Nx))

        # off_selected_surface_mask = (~valid_mask.astype(bool)).astype(np.uint8)
        surface_distance_cost = build_surface_distance_cost(
            valid_mask=valid_mask,
            L=L,
            buffer_m=SELECTED_SURFACE_BUFFER_M,
            falloff_m=OFF_SURFACE_FALLOFF_M,
        )

        mask_screen_dev = wp.array(
            surface_distance_cost.reshape(-1),
            dtype=wp.float32,
            device=device,
        ).reshape((Ny, Nx))

        print(
            f"[INFO] surface distance penalty: "
            f"buffer={SELECTED_SURFACE_BUFFER_M:.3f} m "
            f"falloff={OFF_SURFACE_FALLOFF_M:.3f} m "
            f"max_cost={float(np.max(surface_distance_cost)):.3f} "
            f"W_SCREEN={W_SCREEN:.3f}"
        )

        targets_ixiy, targets_cells, targets_T = bb.pick_k_targets_cold_exclusion(
            T0_C,
            P0,
            L,
            R_EXCL_START_M,
            R_EXCL_TARGETS_M,
            NUM_BRANCHES,
            allowed_mask=allowed_mask,
        )

        if len(targets_cells) == 0:
            raise RuntimeError("No feasible targets found under exclusion constraints.")

        print("[INFO] Branch targets:")
        for k, ((ix, iy), p, tC) in enumerate(zip(targets_ixiy, targets_cells, targets_T)):
            dxm = (p[0] - P0[0]) * L
            dym = (p[1] - P0[1]) * L
            dist_m = float(np.sqrt(dxm * dxm + dym * dym))
            print(
                f"  - k={k + 1}: (ix,iy)=({ix},{iy}) "
                f"cells=({p[0]:.1f},{p[1]:.1f}) "
                f"T={tC:.2f}C dist_from_start={dist_m:.3f}m"
            )

        start_batch = time.perf_counter()
        ctrl_opt_b, spline_opt_b, J_final_b, J_hist_b, _W = bb.optimize_to_targets_batched(
            T0_C=T0_C,
            T0_dev_2d=T0_dev_2d,
            mask_screen_dev=mask_screen_dev,
            device=device,
            P0=P0,
            P4_list=targets_cells,
            sigma_m=float(SIGMA_M),
            h_peak=float(H_PEAK),
            L=float(L),
            dt=float(DT),
            T_total=float(T_TOTAL),
            v_mps=float(V_MPS),
            T_hot_K=float(T_hot_K),
            inv_scale=float(inv_scale),
            w_heat=float(W_HEAT),
            w_curv=float(W_CURV),
            w_screen=float(W_SCREEN),
            iters=int(ITERS),
            lr=float(LR),
        )
        batch_time = time.perf_counter() - start_batch

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
            print(
                f"[INFO] branch {k + 1}/{len(targets_cells)} done: "
                f"J_final={float(J_final_b[k]):.6f} "
                f"target=({targets_cells[k][0]:.0f},{targets_cells[k][1]:.0f})"
            )

        best_idx = int(np.argmin([b["J_final"] for b in branch_results]))
        # best_idx = int(np.argmax([b["J_final"] for b in branch_results]))
        best = branch_results[best_idx]
        print(
            f"[INFO] BEST branch = {best_idx + 1} "
            f"J={best['J_final']:.6f} "
            f"target=({best['P4'][0]:.0f},{best['P4'][1]:.0f}) "
            f"[{batch_time:.3f} s]"
        )

        # best_spline_dense_cells = evaluate_spline_dense(best["ctrl_opt"], degree=3, num=400)
        best_spline_dense_cells = evaluate_spline_dense(best["ctrl_opt"], degree=3, num=50)
        best_spline_dense_proj = cells_to_proj_points(best_spline_dense_cells, x_min, y_min, L)

        ctrl_proj = cells_to_proj_points(best["ctrl_opt"], x_min, y_min, L)

        target_proj = cells_to_proj_points(best["P4"][None, :], x_min, y_min, L)[0]
        start_proj = cells_to_proj_points(P0[None, :], x_min, y_min, L)[0]

        p0_x, p0_y = bb.cells_to_meters_xy(P0[0], P0[1], x_min, y_min, L)

        msg = ProjectedHeaterPath()
        msg.header = Header()
        msg.header.stamp = node.get_clock().now().to_msg()
        msg.header.frame_id = frame_id

        msg.x_min = float(x_min)
        msg.y_min = float(y_min)
        msg.cell_size_m = float(L)
        msg.nx = int(Nx)
        msg.ny = int(Ny)

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

        msg.control_proj = [
            make_point_xyz(p[0], p[1], 0.0) for p in ctrl_proj
        ]

        msg.path_proj = [
            make_point_xyz(p[0], p[1], 0.0) for p in best_spline_dense_proj
        ]

        msg.path_cells = [
            make_point_xyz(p[0], p[1], 0.0) for p in best_spline_dense_cells
        ]

        msg.final_cost = float(best["J_final"])
        msg.best_branch_index = int(best_idx)
        
        # while(True):
        #     print("Publishing...")
        #     node.path_pub.publish(msg)
        #     time.sleep(1)
            
        node.path_pub.publish(msg)            
        node.get_logger().info(
            f"Published projected heater path with {len(msg.path_proj)} points to /heater_path_projected"
        )

        plt.figure(figsize=(9.0, 6.8))
        # im = plt.imshow(
        #     T0_C,
        #     origin="lower",
        #     extent=[x_min, x_max, y_min, y_max],
        #     cmap="inferno",
        #     interpolation="nearest",
        #     aspect="equal",
        # )
        T_plot = np.ma.masked_where(~valid_mask.astype(bool), T0_C_raw)

        cmap = plt.cm.inferno.copy()
        cmap.set_bad(alpha=0.0)
        
        # Optional Plotting
        if SHOW_PLOT:
            im = plt.imshow(
                T_plot,
                origin="lower",
                extent=[x_min, x_max, y_min, y_max],
                cmap=cmap,
                interpolation="nearest",
                aspect="equal",
            )
        
            plt.colorbar(im, label="Temperature [°C]")
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.title("Projection branching candidates (initial temperature field)")
            plt.scatter([p0_x], [p0_y], s=120, marker="o", label="Start (P0)")

            ys_occ, xs_occ = np.nonzero(valid_mask.astype(bool))
            x_occ = x_min + xs_occ * L
            y_occ = y_min + ys_occ * L
            plt.scatter(x_occ, y_occ, s=4, c="white", alpha=0.0, linewidths=0)

            for b in branch_results:
                spline_dense = evaluate_spline_dense(b["ctrl_opt"], degree=3, num=400)
                sx, sy = spline_to_world_xy(spline_dense, x_min, y_min, L)

                P4 = b["P4"]
                tx, ty = bb.cells_to_meters_xy(P4[0], P4[1], x_min, y_min, L)

                if b is best:
                    plt.plot(sx, sy, linewidth=3.2, label=f"Best spline (branch {b['k'] + 1})")
                    plt.scatter([tx], [ty], s=140, marker="X", label="Best target")
                else:
                    plt.plot(sx, sy, linewidth=1.6, alpha=0.75)
                    plt.scatter([tx], [ty], s=80, marker="x", alpha=0.85)

            plt.legend(loc="upper right", fontsize=9)
            plt.tight_layout()

            plt.figure(figsize=(8.5, 4.5))
            for b in branch_results:
                plt.plot(
                    b["J_hist"],
                    linewidth=2.0 if b is best else 1.2,
                    alpha=1.0 if b is best else 0.7,
                    label=f"branch {b['k'] + 1} (J={b['J_final']:.3g})",
                )
            plt.xlabel("Iteration")
            plt.ylabel("J_total")
            plt.title("Optimization histories (all branches)")
            plt.legend()
            plt.tight_layout()

            plt.show()

        while(True):
            # print("Publishing...")
            # node.path_pub.publish(msg)
            time.sleep(1)

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
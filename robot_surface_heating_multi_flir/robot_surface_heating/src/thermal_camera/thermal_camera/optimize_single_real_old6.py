#!/usr/bin/env python3
# pyright: reportInvalidTypeForm=false

import os
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

try:
    from scipy.spatial import cKDTree
except Exception:
    cKDTree = None


# ============================================================
# User settings
# ============================================================

SHOW_PLOT = True

THERMAL_TOPIC = "/selected_surface_points"
THERMAL_SCALE = 1000.0

GRID_SPACING_M = 0.020

# Branching
NUM_BRANCHES = 18
R_EXCL_START_M = 0.10
R_EXCL_TARGETS_M = 0.04

# Path timing
V_MPS = 0.13
DT = 0.25
T_TOTAL = 4.0

# Internal optimizer shaping
HOT_PERCENTILE = 90.0
SCALE_K = 0.75

W_CURV = 25.0
W_HEAT = 5e-3
W_SCREEN = 0.0

ITERS = 200
LR = 0.01

# Heater parameters
SIGMA_M = 0.022
H_PEAK = 320.0

# Temperature band
TARGET_TLOW_C = 50.0
TARGET_THIGH_C = 70.0
TARGET_TMIDDLE_C = 0.5 * (TARGET_TLOW_C + TARGET_THIGH_C)

# Priority field weights
W_PRIORITY_INSTANT = 1.0
W_PRIORITY_MEMORY = 1.25
W_PRIORITY_EDGE = 0.35

# Final branch selection weights
W_SELECT_PRIORITY = 2.0
W_SELECT_HOT = 1.0
W_SELECT_CURV = 0.08
W_SELECT_ENDPOINT = 0.1

# Persistent underheat memory
MEMORY_ALPHA = 0.88
MEMORY_FILE = "/tmp/optimize_single_real_underheat_memory.npz"

# Target selection weighting
TARGET_PICK_USE_PRIORITY = True

# Exposure kernel window
GAUSSIAN_RADIUS_SIGMAS = 3.0

MIN_POINTS_FOR_SNAPSHOT = 100

EEF_FRAME = "heat_gun"

USE_FALLBACK_START = False
START_FRAC_X = 0.18
START_FRAC_Y = 0.18


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
# Path helpers
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


def path_curvature_cost(path_cells: np.ndarray) -> float:
    if path_cells.shape[0] < 3:
        return 0.0
    d2 = path_cells[2:] - 2.0 * path_cells[1:-1] + path_cells[:-2]
    return float(np.mean(np.sum(d2 * d2, axis=1)))


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


# ============================================================
# Priority-field helpers
# ============================================================

def compute_edge_bias(valid_mask: np.ndarray) -> np.ndarray:
    Ny, Nx = valid_mask.shape
    ys, xs = np.indices((Ny, Nx), dtype=np.float32)

    dist_left = xs
    dist_right = (Nx - 1) - xs
    dist_bottom = ys
    dist_top = (Ny - 1) - ys

    dist_to_edge = np.minimum(
        np.minimum(dist_left, dist_right),
        np.minimum(dist_bottom, dist_top),
    )

    max_dist = max(1.0, float(np.max(dist_to_edge[valid_mask.astype(bool)]))) if np.any(valid_mask) else 1.0
    edge_bias = 1.0 - (dist_to_edge / max_dist)
    edge_bias = np.clip(edge_bias, 0.0, 1.0)

    edge_bias = np.where(valid_mask.astype(bool), edge_bias, 0.0)
    return edge_bias.astype(np.float32)


def normalize_nonnegative_field(field: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    out = np.where(valid_mask.astype(bool), field, 0.0).astype(np.float32)
    vmax = float(np.max(out)) if np.any(valid_mask) else 0.0
    if vmax > 1e-9:
        out = out / vmax
    return out.astype(np.float32)


def load_underheat_memory(shape: tuple[int, int]) -> np.ndarray:
    if not os.path.exists(MEMORY_FILE):
        return np.zeros(shape, dtype=np.float32)

    try:
        data = np.load(MEMORY_FILE, allow_pickle=False)
        mem = data["memory"].astype(np.float32)
        if mem.shape != shape:
            return np.zeros(shape, dtype=np.float32)
        return mem
    except Exception:
        return np.zeros(shape, dtype=np.float32)


def save_underheat_memory(memory: np.ndarray):
    tmp_path = MEMORY_FILE + ".tmp"
    np.savez(tmp_path, memory=memory.astype(np.float32))
    if os.path.exists(tmp_path):
        os.replace(tmp_path, MEMORY_FILE)


def build_priority_fields(
    T0_C_filled: np.ndarray,
    valid_mask: np.ndarray,
    t_low_C: float,
    t_hot_ref_C: float,
):
    instant_deficit = np.maximum(t_low_C - T0_C_filled, 0.0).astype(np.float32)
    instant_deficit = np.where(valid_mask.astype(bool), instant_deficit, 0.0)

    prev_memory = load_underheat_memory(T0_C_filled.shape)
    prev_memory = np.where(valid_mask.astype(bool), prev_memory, 0.0).astype(np.float32)

    memory = MEMORY_ALPHA * prev_memory + (1.0 - MEMORY_ALPHA) * instant_deficit
    memory = np.where(valid_mask.astype(bool), memory, 0.0).astype(np.float32)
    save_underheat_memory(memory)

    hot_field = np.maximum(T0_C_filled - t_hot_ref_C, 0.0).astype(np.float32)
    hot_field = np.where(valid_mask.astype(bool), hot_field, 0.0)

    edge_bias = compute_edge_bias(valid_mask)

    inst_n = normalize_nonnegative_field(instant_deficit, valid_mask)
    mem_n = normalize_nonnegative_field(memory, valid_mask)
    hot_n = normalize_nonnegative_field(hot_field, valid_mask)

    priority = (
        W_PRIORITY_INSTANT * inst_n
        + W_PRIORITY_MEMORY * mem_n
        + W_PRIORITY_EDGE * edge_bias * np.maximum(inst_n, mem_n)
    )
    priority = normalize_nonnegative_field(priority, valid_mask)

    return {
        "instant_deficit": instant_deficit,
        "memory_deficit": memory,
        "edge_bias": edge_bias,
        "hot_field": hot_field,
        "priority": priority,
        "hot_norm": hot_n,
    }


def pick_k_targets_priority_exclusion(
    priority_field: np.ndarray,
    T_C: np.ndarray,
    valid_mask: np.ndarray,
    start_cells: np.ndarray,
    L_spacing: float,
    r_excl_start_m: float,
    r_excl_targets_m: float,
    K: int,
):
    Ny, Nx = T_C.shape
    xs = np.arange(Nx, dtype=np.float32)
    ys = np.arange(Ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    dx0 = (X - float(start_cells[0])) * L_spacing
    dy0 = (Y - float(start_cells[1])) * L_spacing
    dist0 = np.sqrt(dx0 * dx0 + dy0 * dy0)

    allowed = (dist0 >= float(r_excl_start_m)) & valid_mask.astype(bool)

    targets_ixiy: list[tuple[int, int]] = []
    targets_cells: list[np.ndarray] = []
    targets_T: list[float] = []

    for _ in range(K):
        if not np.any(allowed):
            break

        score_map = np.where(allowed, priority_field, -np.inf).astype(np.float32)
        idx = int(np.argmax(score_map))
        iy, ix = np.unravel_index(idx, (Ny, Nx))

        if not np.isfinite(score_map[iy, ix]) or score_map[iy, ix] <= 0.0:
            break

        targets_ixiy.append((int(ix), int(iy)))
        targets_cells.append(np.array([float(ix), float(iy)], dtype=np.float32))
        targets_T.append(float(T_C[iy, ix]))

        dx = (X - float(ix)) * L_spacing
        dy = (Y - float(iy)) * L_spacing
        dist = np.sqrt(dx * dx + dy * dy)
        allowed = allowed & (dist >= float(r_excl_targets_m))

    if len(targets_cells) == 0:
        return bb.pick_k_targets_cold_exclusion(
            T_C,
            start_cells,
            L_spacing,
            r_excl_start_m,
            r_excl_targets_m,
            K,
            allowed_mask=valid_mask,
        )

    return targets_ixiy, targets_cells, targets_T


def score_branch_normalized(
    path_cells: np.ndarray,
    priority_field: np.ndarray,
    hot_norm_field: np.ndarray,
    valid_mask: np.ndarray,
    endpoint_priority: float,
    L: float,
    sigma_m: float,
    dt: float,
    h_peak: float,
):
    Ny, Nx = priority_field.shape
    sigma_cells = max(sigma_m / max(L, 1e-9), 1e-6)
    radius_cells = max(1, int(np.ceil(GAUSSIAN_RADIUS_SIGMAS * sigma_cells)))

    sum_exposure = 0.0
    sum_priority = 0.0
    sum_hot = 0.0

    for p in path_cells:
        hx = float(p[0])
        hy = float(p[1])

        cx = int(np.floor(hx))
        cy = int(np.floor(hy))

        ix0 = max(0, cx - radius_cells)
        ix1 = min(Nx - 1, cx + radius_cells)
        iy0 = max(0, cy - radius_cells)
        iy1 = min(Ny - 1, cy + radius_cells)

        xs = np.arange(ix0, ix1 + 1, dtype=np.float32)
        ys = np.arange(iy0, iy1 + 1, dtype=np.float32)
        X, Y = np.meshgrid(xs, ys, indexing="xy")

        dx_m = (X - hx) * L
        dy_m = (Y - hy) * L
        r2_m = dx_m * dx_m + dy_m * dy_m

        weights = np.exp(-0.5 * r2_m / max(sigma_m * sigma_m, 1e-12)).astype(np.float32)
        exposure = (h_peak * dt) * weights

        local_valid = valid_mask[iy0:iy1 + 1, ix0:ix1 + 1].astype(bool)
        if not np.any(local_valid):
            continue

        local_priority = priority_field[iy0:iy1 + 1, ix0:ix1 + 1]
        local_hot = hot_norm_field[iy0:iy1 + 1, ix0:ix1 + 1]

        exp_valid = exposure[local_valid]
        sum_exposure += float(np.sum(exp_valid))
        sum_priority += float(np.sum(exp_valid * local_priority[local_valid]))
        sum_hot += float(np.sum(exp_valid * local_hot[local_valid]))

    if sum_exposure <= 1e-9:
        avg_priority = 0.0
        avg_hot = 0.0
    else:
        avg_priority = sum_priority / sum_exposure
        avg_hot = sum_hot / sum_exposure

    curv_penalty = path_curvature_cost(path_cells)

    selection_score = (
        W_SELECT_PRIORITY * avg_priority
        + W_SELECT_ENDPOINT * float(endpoint_priority)
        - W_SELECT_HOT * avg_hot
        - W_SELECT_CURV * curv_penalty
    )

    selection_cost = -selection_score

    return {
        "avg_priority": float(avg_priority),
        "avg_hot": float(avg_hot),
        "curv_penalty": float(curv_penalty),
        "endpoint_priority": float(endpoint_priority),
        "selection_score": float(selection_score),
        "selection_cost": float(selection_cost),
    }


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

        T0_C_raw = proj["T0_C_raw"]
        T0_C = proj["T0_C_filled"]
        valid_mask = proj["valid_mask"]
        Nx = proj["Nx"]
        Ny = proj["Ny"]
        L = proj["L"]
        x_min = proj["x_min"]
        x_max = proj["x_max"]
        y_min = proj["y_min"]
        y_max = proj["y_max"]

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
                print("[WARN] Falling back to nominal start because TF was unavailable.")
            else:
                raise RuntimeError(
                    f"Could not get current robot pose for frame '{EEF_FRAME}' in frame '{frame_id}'."
                )

        allowed_mask = valid_mask.copy()

        T_valid = T0_C_raw[allowed_mask.astype(bool)]
        T_valid = T_valid[np.isfinite(T_valid)]
        if T_valid.size == 0:
            raise RuntimeError("No valid projected temperatures available.")

        T0_K = (T0_C + 273.15).astype(np.float32)
        T_hot_ref_C = float(np.percentile(T_valid, HOT_PERCENTILE))
        T_hot_K = float(T_hot_ref_C + 273.15)
        frac_hot = float(np.mean(T_valid >= T_hot_ref_C))
        inv_scale = 1.0 / SCALE_K

        field_maps = build_priority_fields(
            T0_C_filled=T0_C,
            valid_mask=valid_mask,
            t_low_C=TARGET_TLOW_C,
            t_hot_ref_C=T_hot_ref_C,
        )

        priority_field = field_maps["priority"]
        hot_norm_field = field_maps["hot_norm"]
        instant_deficit = field_maps["instant_deficit"]
        memory_deficit = field_maps["memory_deficit"]
        edge_bias = field_maps["edge_bias"]

        print(
            f"[INFO] ROS frame={frame_id} points={points_xyz.shape[0]} "
            f"grid: Nx={Nx} Ny={Ny} L={L:.5f} dt={DT} H={int(round(T_TOTAL / DT))} "
            f"v={V_MPS:.3f} m/s T={T_TOTAL:.2f} s"
        )
        print(
            f"[INFO] T0_C: min={float(np.min(T_valid)):.2f} "
            f"max={float(np.max(T_valid)):.2f} "
            f"mean={float(np.mean(T_valid)):.2f} | "
            f"hot%={100.0 * frac_hot:.2f}% (>=p{HOT_PERCENTILE:.0f}={T_hot_ref_C:.2f}C)"
        )
        print(
            f"[INFO] params: branches={NUM_BRANCHES} "
            f"r_start={R_EXCL_START_M:.3f}m "
            f"r_targets={R_EXCL_TARGETS_M:.3f}m "
            f"sigma={SIGMA_M:.4f}m "
            f"Wcurv={W_CURV:.3f}"
        )
        print(
            f"[INFO] selection weights: "
            f"Wpriority={W_SELECT_PRIORITY:.3f} "
            f"Wendpoint={W_SELECT_ENDPOINT:.3f} "
            f"Whot={W_SELECT_HOT:.3f} "
            f"Wcurv={W_SELECT_CURV:.3f}"
        )

        T0_dev_2d = wp.array(T0_K.reshape(-1), dtype=wp.float32, device=device).reshape((Ny, Nx))
        mask_screen_dev = wp.zeros((Ny, Nx), dtype=wp.uint8, device=device)

        if TARGET_PICK_USE_PRIORITY:
            targets_ixiy, targets_cells, targets_T = pick_k_targets_priority_exclusion(
                priority_field=priority_field,
                T_C=T0_C,
                valid_mask=allowed_mask,
                start_cells=P0,
                L_spacing=L,
                r_excl_start_m=R_EXCL_START_M,
                r_excl_targets_m=R_EXCL_TARGETS_M,
                K=NUM_BRANCHES,
            )
        else:
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
            inst_def = max(TARGET_TLOW_C - float(tC), 0.0)
            prio = float(priority_field[iy, ix])
            mem_def = float(memory_deficit[iy, ix])
            edge = float(edge_bias[iy, ix])
            print(
                f"  - k={k + 1}: (ix,iy)=({ix},{iy}) "
                f"cells=({p[0]:.1f},{p[1]:.1f}) "
                f"T={tC:.2f}C inst_def={inst_def:.2f}C "
                f"mem={mem_def:.2f} edge={edge:.2f} prio={prio:.3f} "
                f"dist={dist_m:.3f}m"
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
            p4 = targets_cells[k]
            ix = int(round(float(p4[0])))
            iy = int(round(float(p4[1])))
            ix = max(0, min(Nx - 1, ix))
            iy = max(0, min(Ny - 1, iy))
            endpoint_priority = float(priority_field[iy, ix])

            selection_terms = score_branch_normalized(
                path_cells=spline_opt_b[k],
                priority_field=priority_field,
                hot_norm_field=hot_norm_field,
                valid_mask=valid_mask,
                endpoint_priority=endpoint_priority,
                L=L,
                sigma_m=SIGMA_M,
                dt=DT,
                h_peak=H_PEAK,
            )

            branch_results.append(
                {
                    "k": k,
                    "P4": targets_cells[k],
                    "ctrl_opt": ctrl_opt_b[k],
                    "spline_opt": spline_opt_b[k],
                    "J_final_optimizer": float(J_final_b[k]),
                    "J_hist": J_hist_b[k],
                    "avg_priority": selection_terms["avg_priority"],
                    "avg_hot": selection_terms["avg_hot"],
                    "curv_penalty": selection_terms["curv_penalty"],
                    "endpoint_priority": selection_terms["endpoint_priority"],
                    "selection_score": selection_terms["selection_score"],
                    "selection_cost": selection_terms["selection_cost"],
                }
            )

            print(
                f"[INFO] branch {k + 1}/{len(targets_cells)}: "
                f"Jopt={float(J_final_b[k]):.6f} "
                f"score={selection_terms['selection_score']:.6f} "
                f"avg_prio={selection_terms['avg_priority']:.6f} "
                f"end_prio={selection_terms['endpoint_priority']:.6f} "
                f"avg_hot={selection_terms['avg_hot']:.6f} "
                f"curv={selection_terms['curv_penalty']:.6f} "
                f"target=({targets_cells[k][0]:.0f},{targets_cells[k][1]:.0f})"
            )

        best_idx = int(np.argmax([b["selection_score"] for b in branch_results]))
        best = branch_results[best_idx]

        print(
            f"[INFO] BEST branch = {best_idx + 1} "
            f"score={best['selection_score']:.6f} "
            f"Jopt={best['J_final_optimizer']:.6f} "
            f"target=({best['P4'][0]:.0f},{best['P4'][1]:.0f}) "
            f"[{batch_time:.3f} s]"
        )

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

        msg.control_proj = [make_point_xyz(p[0], p[1], 0.0) for p in ctrl_proj]
        msg.path_proj = [make_point_xyz(p[0], p[1], 0.0) for p in best_spline_dense_proj]
        msg.path_cells = [make_point_xyz(p[0], p[1], 0.0) for p in best_spline_dense_cells]

        msg.final_cost = float(-best["selection_score"])
        msg.best_branch_index = int(best_idx)

        node.path_pub.publish(msg)
        node.get_logger().info(
            f"Published projected heater path with {len(msg.path_proj)} points to /heater_path_projected"
        )

        if SHOW_PLOT:
            plt.figure(figsize=(9.0, 6.8))
            T_plot = np.ma.masked_where(~valid_mask.astype(bool), T0_C_raw)
            cmap_temp = plt.cm.inferno.copy()
            cmap_temp.set_bad(alpha=0.0)

            im = plt.imshow(
                T_plot,
                origin="lower",
                extent=[x_min, x_max, y_min, y_max],
                cmap=cmap_temp,
                interpolation="nearest",
                aspect="equal",
            )
            plt.colorbar(im, label="Temperature [°C]")
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.title("Projection branching candidates")
            plt.scatter([p0_x], [p0_y], s=120, marker="o", label="Start (P0)")

            for b in branch_results:
                spline_dense = evaluate_spline_dense(b["ctrl_opt"], degree=3, num=300)
                sx, sy = spline_to_world_xy(spline_dense, x_min, y_min, L)
                P4 = b["P4"]
                tx, ty = bb.cells_to_meters_xy(P4[0], P4[1], x_min, y_min, L)

                if b is best:
                    plt.plot(
                        sx,
                        sy,
                        linewidth=3.0,
                        label=(
                            f"Best branch {b['k'] + 1} "
                            f"(score={b['selection_score']:.3f})"
                        ),
                    )
                    plt.scatter([tx], [ty], s=140, marker="X", label="Best target")
                else:
                    plt.plot(sx, sy, linewidth=1.2, alpha=0.70)
                    plt.scatter([tx], [ty], s=70, marker="x", alpha=0.80)

            plt.legend(loc="upper right", fontsize=9)
            plt.tight_layout()

            plt.figure(figsize=(10.0, 4.2))
            im2 = plt.imshow(
                priority_field,
                origin="lower",
                extent=[x_min, x_max, y_min, y_max],
                cmap="viridis",
                interpolation="nearest",
                aspect="equal",
            )
            plt.colorbar(im2, label="Priority field [normalized]")
            plt.xlabel("x [m]")
            plt.ylabel("y [m]")
            plt.title("Priority field used for branch selection")
            plt.tight_layout()

            plt.figure(figsize=(8.5, 4.5))
            for b in branch_results:
                plt.plot(
                    b["J_hist"],
                    linewidth=2.0 if b is best else 1.2,
                    alpha=1.0 if b is best else 0.7,
                    label=(
                        f"branch {b['k'] + 1} "
                        f"(Jopt={b['J_final_optimizer']:.3g}, "
                        f"score={b['selection_score']:.3g})"
                    ),
                )
            plt.xlabel("Iteration")
            plt.ylabel("Optimizer J_total")
            plt.title("Optimization histories")
            plt.legend()
            plt.tight_layout()

            plt.show()

        while True:
            time.sleep(1)

    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
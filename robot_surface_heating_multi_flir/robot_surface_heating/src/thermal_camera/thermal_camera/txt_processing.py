#!/usr/bin/env python3

import os
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree


# ============================================================
# User settings
# ============================================================

# TXT_FILE_NAME = "nd-composite-core.txt" 
TXT_FILE_NAME = "nd-metal-curved.txt"

SCALE_TO_METERS = 0.0254

SKIP_ROWS = 0
XYZ_COLUMNS = (0, 1, 2)

# ----------------------------
# Planar trim
# ----------------------------

# --- nd-compostie-core ---
# VERTICAL_AXIS = "z"
# KEEP_SIDE = "above"
# TRIM_MODE = "absolute"

# ABSOLUTE_PLANE_VALUE_M = 0.0
# TRIM_OFFSET_M = -0.754931 + 0.019
# --- nd-compostie-core ---

# --- nd-metal-curved ---
VERTICAL_AXIS = "z"
KEEP_SIDE = "above"
TRIM_MODE = "absolute"

ABSOLUTE_PLANE_VALUE_M = 0.0
TRIM_OFFSET_M = -0.766992 + 0.016
# --- nd-metal-curved ---

# ----------------------------
# Optional prefilter
# ----------------------------

USE_VOXEL_PREFILTER = False
VOXEL_PREFILTER_SIZE_M = 0.004

# ----------------------------
# Surface lattice
# ----------------------------

BUILD_SURFACE_LATTICE = True

LATTICE_SPACING_M = 0.02

MAX_UV_SNAP_DIST_M = 0.010
SNAP_K_NEIGHBORS = 8
SNAP_SIGMA_M = 0.010

# ----------------------------
# Lattice relaxation
# ----------------------------

RELAX_LATTICE = True

RELAX_ITERS = 150
RELAX_OMEGA = 0.10
ANCHOR_W = 0.02
MAX_UV_STEP_FRAC = 0.20

# ----------------------------
# Boundary cleanup
# ----------------------------

CLEAN_BAD_LATTICE_EDGES = True

MIN_GOOD_EDGE_FACTOR = 0.40
MAX_GOOD_EDGE_FACTOR = 1.80
MIN_NODE_DEGREE = 1
CLEANUP_ITERATIONS = 1

# ----------------------------
# Visualization
# ----------------------------

VISUALIZE_RAW = False
VISUALIZE_TRIMMED = False
VISUALIZE_PREFILTERED = False

VISUALIZE_SCAN_AND_LATTICE = False
VISUALIZE_LATTICE_ONLY = True

SHOW_LATTICE_LINES = True

ENABLE_RAW_POINT_PICKING = True
RAW_PICK_OUTPUT_NAME = "picked_raw_reference_points.txt"

# ============================================================
# File helpers
# ============================================================

def get_script_relative_path(file_name: str) -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, file_name)


def load_pointcloud_txt(
    path: str,
    scale_to_meters: float,
    skip_rows: int,
    xyz_columns: tuple[int, int, int],
) -> np.ndarray:
    try:
        data = np.loadtxt(path, skiprows=skip_rows, delimiter=",")
    except Exception:
        data = np.loadtxt(path, skiprows=skip_rows)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    points = data[:, xyz_columns].astype(np.float64)
    points *= float(scale_to_meters)

    valid = np.all(np.isfinite(points), axis=1)
    return points[valid]


def axis_to_index(axis_name: str) -> int:
    axis_name = axis_name.lower().strip()

    if axis_name == "x":
        return 0
    if axis_name == "y":
        return 1
    if axis_name == "z":
        return 2

    raise ValueError("Axis must be 'x', 'y', or 'z'")


# ============================================================
# Diagnostics
# ============================================================

def print_pointcloud_diagnostics(points: np.ndarray, label: str) -> None:
    print(f"\n[{label}]")

    if points.size == 0:
        print("No points.")
        return

    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    ranges = maxs - mins

    print(f"Point count: {len(points):,}")
    print("Bounds in meters:")
    print(f"  x: min={mins[0]:.6f}, max={maxs[0]:.6f}, range={ranges[0]:.6f}")
    print(f"  y: min={mins[1]:.6f}, max={maxs[1]:.6f}, range={ranges[1]:.6f}")
    print(f"  z: min={mins[2]:.6f}, max={maxs[2]:.6f}, range={ranges[2]:.6f}")
    print("Approximate size:")
    print(f"  {ranges[0]:.4f} m x {ranges[1]:.4f} m x {ranges[2]:.4f} m")


def print_lattice_spacing_diagnostics(
    lattice_points: np.ndarray,
    edges: np.ndarray,
    label: str,
) -> None:
    print(f"\n[{label}]")

    if len(lattice_points) == 0:
        print("No lattice points.")
        return

    if len(edges) == 0:
        print("No lattice edges.")
        return

    p0 = lattice_points[edges[:, 0]]
    p1 = lattice_points[edges[:, 1]]
    distances = np.linalg.norm(p1 - p0, axis=1)

    print(f"Node count: {len(lattice_points):,}")
    print(f"Edge count: {len(edges):,}")
    print(f"Target spacing: {LATTICE_SPACING_M:.6f} m")
    print(f"Min spacing:    {distances.min():.6f} m")
    print(f"Mean spacing:   {distances.mean():.6f} m")
    print(f"Max spacing:    {distances.max():.6f} m")
    print(f"Std spacing:    {distances.std():.6f} m")


# ============================================================
# Pointcloud operations
# ============================================================

def trim_by_axis_plane(
    points: np.ndarray,
    axis_name: str,
    keep_side: str,
    trim_mode: str,
    absolute_plane_value_m: float,
    trim_offset_m: float,
) -> tuple[np.ndarray, float, np.ndarray]:
    axis_index = axis_to_index(axis_name)
    keep_side = keep_side.lower().strip()
    trim_mode = trim_mode.lower().strip()

    values = points[:, axis_index]

    if trim_mode == "absolute":
        plane_value = float(absolute_plane_value_m + trim_offset_m)
    elif trim_mode == "relative_min":
        plane_value = float(values.min() + trim_offset_m)
    elif trim_mode == "relative_max":
        plane_value = float(values.max() - trim_offset_m)
    else:
        raise ValueError("TRIM_MODE must be 'absolute', 'relative_min', or 'relative_max'")

    if keep_side == "above":
        keep_mask = values > plane_value
    elif keep_side == "below":
        keep_mask = values < plane_value
    else:
        raise ValueError("KEEP_SIDE must be 'above' or 'below'")

    return points[keep_mask], plane_value, keep_mask


def voxel_downsample(points: np.ndarray, voxel_size_m: float) -> np.ndarray:
    cloud = make_open3d_cloud(points, color=(0.6, 0.6, 0.6))
    cloud_down = cloud.voxel_down_sample(voxel_size=float(voxel_size_m))
    return np.asarray(cloud_down.points)


# ============================================================
# Lattice generation
# ============================================================

def build_projection_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = points.mean(axis=0)
    centered = points - center

    _, _, vt = np.linalg.svd(centered, full_matrices=False)

    axis_u = vt[0]
    axis_v = vt[1]
    axis_n = vt[2]

    axis_u = axis_u / np.linalg.norm(axis_u)
    axis_v = axis_v / np.linalg.norm(axis_v)
    axis_n = axis_n / np.linalg.norm(axis_n)

    return center, axis_u, axis_v, axis_n


def project_points_to_uv(
    points: np.ndarray,
    center: np.ndarray,
    axis_u: np.ndarray,
    axis_v: np.ndarray,
) -> np.ndarray:
    q = points - center[None, :]
    u = q @ axis_u
    v = q @ axis_v
    return np.column_stack((u, v))


def make_grid_edges_from_valid_mask(valid_mask_flat: np.ndarray, nx: int, ny: int) -> np.ndarray:
    index_grid = -np.ones((ny, nx), dtype=np.int32)
    index_grid[valid_mask_flat.reshape(ny, nx)] = np.arange(np.count_nonzero(valid_mask_flat))

    edges = []

    for iy in range(ny):
        for ix in range(nx):
            i = index_grid[iy, ix]
            if i < 0:
                continue

            if ix + 1 < nx:
                j = index_grid[iy, ix + 1]
                if j >= 0:
                    edges.append((i, j))

            if iy + 1 < ny:
                j = index_grid[iy + 1, ix]
                if j >= 0:
                    edges.append((i, j))

    return np.asarray(edges, dtype=np.int32)


def snap_uv_to_surface(
    uv_query: np.ndarray,
    source_points: np.ndarray,
    source_uv: np.ndarray,
    tree: cKDTree,
    k_neighbors: int,
    snap_sigma_m: float,
) -> np.ndarray:
    k = min(int(k_neighbors), len(source_points))
    distances, indices = tree.query(uv_query, k=k)

    if k == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    weights = np.exp(-0.5 * (distances / max(float(snap_sigma_m), 1e-9)) ** 2)
    weights /= np.clip(weights.sum(axis=1, keepdims=True), 1e-12, None)

    neighbor_points = source_points[indices]
    return np.sum(weights[:, :, None] * neighbor_points, axis=1)


def build_surface_lattice_from_pointcloud(
    points: np.ndarray,
    spacing_m: float,
    max_uv_snap_dist_m: float,
    k_neighbors: int,
    snap_sigma_m: float,
) -> dict:
    center, axis_u, axis_v, axis_n = build_projection_frame(points)
    source_uv = project_points_to_uv(points, center, axis_u, axis_v)

    u_min, v_min = source_uv.min(axis=0)
    u_max, v_max = source_uv.max(axis=0)

    u_values = np.arange(u_min, u_max + spacing_m * 0.5, spacing_m)
    v_values = np.arange(v_min, v_max + spacing_m * 0.5, spacing_m)

    uu, vv = np.meshgrid(u_values, v_values, indexing="xy")
    grid_uv = np.column_stack((uu.ravel(), vv.ravel()))

    ny, nx = uu.shape

    tree = cKDTree(source_uv)
    nearest_dist, _ = tree.query(grid_uv, k=1)

    valid_mask_flat = nearest_dist <= float(max_uv_snap_dist_m)
    valid_grid_uv = grid_uv[valid_mask_flat]

    lattice_points = snap_uv_to_surface(
        valid_grid_uv,
        source_points=points,
        source_uv=source_uv,
        tree=tree,
        k_neighbors=k_neighbors,
        snap_sigma_m=snap_sigma_m,
    )

    edges = make_grid_edges_from_valid_mask(valid_mask_flat, nx, ny)

    return {
        "lattice_points": lattice_points,
        "edges": edges,
        "valid_mask_flat": valid_mask_flat,
        "nx": nx,
        "ny": ny,
        "grid_uv": grid_uv,
        "valid_grid_uv": valid_grid_uv,
        "source_points": points,
        "source_uv": source_uv,
        "tree": tree,
        "center": center,
        "axis_u": axis_u,
        "axis_v": axis_v,
        "axis_n": axis_n,
        "u_min": float(u_min),
        "u_max": float(u_max),
        "v_min": float(v_min),
        "v_max": float(v_max),
    }


# ============================================================
# Lattice relaxation
# ============================================================

def relax_lattice_uv(
    lattice_uv: np.ndarray,
    lattice_points: np.ndarray,
    edges: np.ndarray,
    source_points: np.ndarray,
    source_uv: np.ndarray,
    tree: cKDTree,
    target_spacing_m: float,
    k_neighbors: int,
    snap_sigma_m: float,
    iterations: int,
    omega: float,
    anchor_w: float,
    max_step_frac: float,
) -> tuple[np.ndarray, np.ndarray]:
    uv = lattice_uv.copy()
    uv_anchor = lattice_uv.copy()
    points = lattice_points.copy()

    max_step = float(max_step_frac * target_spacing_m)

    if len(edges) == 0:
        return uv, points

    for _ in range(max(0, int(iterations))):
        delta = np.zeros_like(uv)
        counts = np.zeros((len(uv), 1), dtype=np.float64)

        i = edges[:, 0]
        j = edges[:, 1]

        pi = points[i]
        pj = points[j]

        edge_vec_3d = pj - pi
        edge_len = np.linalg.norm(edge_vec_3d, axis=1)
        edge_len_safe = np.maximum(edge_len, 1e-9)

        uv_vec = uv[j] - uv[i]
        uv_len = np.linalg.norm(uv_vec, axis=1)
        uv_len_safe = np.maximum(uv_len, 1e-9)
        uv_dir = uv_vec / uv_len_safe[:, None]

        error = edge_len - float(target_spacing_m)

        move = 0.5 * float(omega) * error[:, None] * uv_dir

        np.add.at(delta, i, move)
        np.add.at(delta, j, -move)

        np.add.at(counts, i, 1.0)
        np.add.at(counts, j, 1.0)

        delta /= np.maximum(counts, 1.0)

        if anchor_w > 0.0:
            delta += float(anchor_w) * (uv_anchor - uv)

        step_norm = np.linalg.norm(delta, axis=1)
        too_large = step_norm > max_step
        if np.any(too_large):
            delta[too_large] *= (max_step / step_norm[too_large])[:, None]

        uv += delta

        points = snap_uv_to_surface(
            uv,
            source_points=source_points,
            source_uv=source_uv,
            tree=tree,
            k_neighbors=k_neighbors,
            snap_sigma_m=snap_sigma_m,
        )

    return uv, points


# ============================================================
# Lattice cleanup
# ============================================================

def remove_bad_edges(
    lattice_points: np.ndarray,
    edges: np.ndarray,
    target_spacing_m: float,
    min_factor: float,
    max_factor: float,
) -> np.ndarray:
    if len(edges) == 0:
        return edges

    p0 = lattice_points[edges[:, 0]]
    p1 = lattice_points[edges[:, 1]]
    distances = np.linalg.norm(p1 - p0, axis=1)

    min_allowed = float(min_factor * target_spacing_m)
    max_allowed = float(max_factor * target_spacing_m)

    keep = (distances >= min_allowed) & (distances <= max_allowed)
    return edges[keep]


def remove_low_degree_nodes(
    lattice_points: np.ndarray,
    edges: np.ndarray,
    min_degree: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(lattice_points) == 0:
        return lattice_points, edges

    if len(edges) == 0:
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32)

    degree = np.zeros(len(lattice_points), dtype=np.int32)

    np.add.at(degree, edges[:, 0], 1)
    np.add.at(degree, edges[:, 1], 1)

    keep_nodes = degree >= int(min_degree)

    if not np.any(keep_nodes):
        return np.empty((0, 3), dtype=np.float64), np.empty((0, 2), dtype=np.int32)

    old_to_new = -np.ones(len(lattice_points), dtype=np.int32)
    old_to_new[keep_nodes] = np.arange(np.count_nonzero(keep_nodes))

    keep_edges = keep_nodes[edges[:, 0]] & keep_nodes[edges[:, 1]]
    edges_kept = edges[keep_edges]
    edges_new = old_to_new[edges_kept]

    points_new = lattice_points[keep_nodes]

    return points_new, edges_new.astype(np.int32)


def clean_lattice(
    lattice_points: np.ndarray,
    edges: np.ndarray,
    target_spacing_m: float,
    min_edge_factor: float,
    max_edge_factor: float,
    min_node_degree: int,
    cleanup_iterations: int,
) -> tuple[np.ndarray, np.ndarray]:
    points_clean = lattice_points.copy()
    edges_clean = edges.copy()

    for _ in range(max(1, int(cleanup_iterations))):
        edges_clean = remove_bad_edges(
            points_clean,
            edges_clean,
            target_spacing_m=target_spacing_m,
            min_factor=min_edge_factor,
            max_factor=max_edge_factor,
        )

        points_clean, edges_clean = remove_low_degree_nodes(
            points_clean,
            edges_clean,
            min_degree=min_node_degree,
        )

        if len(points_clean) == 0 or len(edges_clean) == 0:
            break

    return points_clean, edges_clean


# ============================================================
# Visualization
# ============================================================

def make_open3d_cloud(points: np.ndarray, color: tuple[float, float, float]) -> o3d.geometry.PointCloud:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)

    colors = np.tile(np.array(color, dtype=np.float64), (len(points), 1))
    cloud.colors = o3d.utility.Vector3dVector(colors)

    return cloud


# def make_lattice_lines(
#     lattice_points: np.ndarray,
#     edges: np.ndarray,
#     color: tuple[float, float, float] = (1.0, 0.2, 0.1),
# ) -> o3d.geometry.LineSet:
#     line_set = o3d.geometry.LineSet()
#     line_set.points = o3d.utility.Vector3dVector(lattice_points)

#     if len(edges) > 0:
#         line_set.lines = o3d.utility.Vector2iVector(edges.astype(np.int32))
#         colors = np.tile(np.array(color, dtype=np.float64), (len(edges), 1))
#         line_set.colors = o3d.utility.Vector3dVector(colors)

#     return line_set

def make_lattice_lines(
    lattice_points: np.ndarray,
    edges: np.ndarray,
    color: tuple[float, float, float] = (1.0, 0.2, 0.1),
) -> o3d.geometry.LineSet:
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(lattice_points)

    if len(edges) == 0:
        return line_set

    p0 = lattice_points[edges[:, 0]]
    p1 = lattice_points[edges[:, 1]]
    distances = np.linalg.norm(p1 - p0, axis=1)

    d_min = float(distances.min())
    d_max = float(distances.max())
    denom = max(d_max - d_min, 1e-12)

    t = (distances - d_min) / denom

    # Blue = shortest, green/yellow = middle, red = longest
    colors = np.zeros((len(edges), 3), dtype=np.float64)
    colors[:, 0] = t
    colors[:, 1] = 1.0 - np.abs(t - 0.5) * 2.0
    colors[:, 2] = 1.0 - t

    line_set.lines = o3d.utility.Vector2iVector(edges.astype(np.int32))
    line_set.colors = o3d.utility.Vector3dVector(colors)

    print("\n[Line color scale]")
    print(f"Blue  = shortest edge: {d_min:.6f} m")
    print(f"Red   = longest edge:  {d_max:.6f} m")

    return line_set

def visualize_geometries(geometries: list, title: str) -> None:
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)
    print(f"Opening Open3D viewer: {title}")
    o3d.visualization.draw_geometries([*geometries, frame], window_name=title)


def visualize_points(points: np.ndarray, title: str, color: tuple[float, float, float]) -> None:
    if len(points) == 0:
        print(f"Skipping visualization for {title}: no points.")
        return

    visualize_geometries([make_open3d_cloud(points, color)], title)


def pick_points_from_raw_cloud(points: np.ndarray, output_name: str) -> np.ndarray:
    if len(points) == 0:
        print("No raw points available for picking.")
        return np.empty((0, 3), dtype=np.float64)

    cloud = make_open3d_cloud(points, color=(0.6, 0.6, 0.6))
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05)

    print("\n[Raw point picking]")
    print("Instructions:")
    print("  1. Hold Shift + left click to pick points.")
    print("  2. Pick the reference points in order.")
    print("  3. Close the Open3D window when done.")
    print("  4. The selected XYZ coordinates will print below.")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name="Pick raw reference points", width=1400, height=900)
    vis.add_geometry(cloud)
    vis.add_geometry(frame)
    vis.run()
    picked_indices = vis.get_picked_points()
    vis.destroy_window()

    if len(picked_indices) == 0:
        print("No points were picked.")
        return np.empty((0, 3), dtype=np.float64)

    picked_points = points[np.asarray(picked_indices, dtype=np.int64)]

    print("\n[Picked raw reference points]")
    print("Use these as model_pts, in this order:")
    print("model_pts = np.array([")
    for p in picked_points:
        print(f"    [{p[0]: .8f}, {p[1]: .8f}, {p[2]: .8f}],")
    print("], dtype=np.float64)")

    output_path = get_script_relative_path(output_name)
    np.savetxt(output_path, picked_points, fmt="%.10f", delimiter=",")
    print(f"\nSaved picked points to: {output_path}")

    return picked_points


# ============================================================
# Main
# ============================================================

def main() -> None:
    txt_path = get_script_relative_path(TXT_FILE_NAME)

    print("Pointcloud lattice test")
    print(f"TXT path: {txt_path}")
    print(f"Scale to meters: {SCALE_TO_METERS}")

    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"Could not find TXT file: {txt_path}")

    points_raw = load_pointcloud_txt(
        txt_path,
        scale_to_meters=SCALE_TO_METERS,
        skip_rows=SKIP_ROWS,
        xyz_columns=XYZ_COLUMNS,
    )

    print_pointcloud_diagnostics(points_raw, "Raw pointcloud")

    if ENABLE_RAW_POINT_PICKING:
        pick_points_from_raw_cloud(points_raw, RAW_PICK_OUTPUT_NAME)

    points_trimmed, plane_value, _ = trim_by_axis_plane(
        points_raw,
        axis_name=VERTICAL_AXIS,
        keep_side=KEEP_SIDE,
        trim_mode=TRIM_MODE,
        absolute_plane_value_m=ABSOLUTE_PLANE_VALUE_M,
        trim_offset_m=TRIM_OFFSET_M,
    )

    kept_percent = 100.0 * len(points_trimmed) / max(len(points_raw), 1)

    print("\n[Trim settings]")
    print(f"Vertical axis: {VERTICAL_AXIS}")
    print(f"Keep side: {KEEP_SIDE}")
    print(f"Trim mode: {TRIM_MODE}")
    print(f"Trim offset: {TRIM_OFFSET_M:.6f} m")
    print(f"Plane value: {plane_value:.6f} m")
    print(f"Kept points: {len(points_trimmed):,} / {len(points_raw):,} ({kept_percent:.2f}%)")

    print_pointcloud_diagnostics(points_trimmed, "Trimmed pointcloud")

    points_for_lattice = points_trimmed

    if USE_VOXEL_PREFILTER:
        points_for_lattice = voxel_downsample(points_trimmed, VOXEL_PREFILTER_SIZE_M)
        print_pointcloud_diagnostics(points_for_lattice, "Voxel prefiltered pointcloud")

    if VISUALIZE_RAW:
        visualize_points(points_raw, "Raw pointcloud", color=(0.6, 0.6, 0.6))

    if VISUALIZE_TRIMMED:
        visualize_points(points_trimmed, "Trimmed pointcloud", color=(0.1, 0.8, 0.2))

    if VISUALIZE_PREFILTERED and USE_VOXEL_PREFILTER:
        visualize_points(points_for_lattice, "Voxel prefiltered pointcloud", color=(0.1, 0.4, 1.0))

    if not BUILD_SURFACE_LATTICE:
        return

    lattice = build_surface_lattice_from_pointcloud(
        points_for_lattice,
        spacing_m=LATTICE_SPACING_M,
        max_uv_snap_dist_m=MAX_UV_SNAP_DIST_M,
        k_neighbors=SNAP_K_NEIGHBORS,
        snap_sigma_m=SNAP_SIGMA_M,
    )

    lattice_points = lattice["lattice_points"]
    lattice_edges = lattice["edges"]
    lattice_uv = lattice["valid_grid_uv"]

    print_pointcloud_diagnostics(lattice_points, "Surface lattice before relaxation")

    print("\n[Lattice settings]")
    print(f"Lattice spacing: {LATTICE_SPACING_M:.6f} m")
    print(f"UV grid shape: nx={lattice['nx']}, ny={lattice['ny']}")
    print(f"Candidate grid nodes: {lattice['nx'] * lattice['ny']:,}")
    print(f"Valid snapped nodes: {len(lattice_points):,}")
    print(f"Initial lattice edges: {len(lattice_edges):,}")
    print(f"Max UV snap distance: {MAX_UV_SNAP_DIST_M:.6f} m")
    print(f"Snap neighbors: {SNAP_K_NEIGHBORS}")
    print(f"Snap sigma: {SNAP_SIGMA_M:.6f} m")

    print_lattice_spacing_diagnostics(
        lattice_points,
        lattice_edges,
        "Lattice spacing before relaxation",
    )

    if RELAX_LATTICE:
        print("\n[Relaxation settings]")
        print(f"Iterations: {RELAX_ITERS}")
        print(f"Omega: {RELAX_OMEGA}")
        print(f"Anchor weight: {ANCHOR_W}")
        print(f"Max UV step: {MAX_UV_STEP_FRAC:.3f} x spacing = {MAX_UV_STEP_FRAC * LATTICE_SPACING_M:.6f} m")

        lattice_uv, lattice_points = relax_lattice_uv(
            lattice_uv=lattice_uv,
            lattice_points=lattice_points,
            edges=lattice_edges,
            source_points=lattice["source_points"],
            source_uv=lattice["source_uv"],
            tree=lattice["tree"],
            target_spacing_m=LATTICE_SPACING_M,
            k_neighbors=SNAP_K_NEIGHBORS,
            snap_sigma_m=SNAP_SIGMA_M,
            iterations=RELAX_ITERS,
            omega=RELAX_OMEGA,
            anchor_w=ANCHOR_W,
            max_step_frac=MAX_UV_STEP_FRAC,
        )

        print_pointcloud_diagnostics(lattice_points, "Surface lattice after relaxation")

        print_lattice_spacing_diagnostics(
            lattice_points,
            lattice_edges,
            "Lattice spacing after relaxation",
        )

    if CLEAN_BAD_LATTICE_EDGES:
        lattice_points_clean, lattice_edges_clean = clean_lattice(
            lattice_points,
            lattice_edges,
            target_spacing_m=LATTICE_SPACING_M,
            min_edge_factor=MIN_GOOD_EDGE_FACTOR,
            max_edge_factor=MAX_GOOD_EDGE_FACTOR,
            min_node_degree=MIN_NODE_DEGREE,
            cleanup_iterations=CLEANUP_ITERATIONS,
        )

        print("\n[Cleanup settings]")
        print(f"Min good edge: {MIN_GOOD_EDGE_FACTOR:.2f} x spacing = {MIN_GOOD_EDGE_FACTOR * LATTICE_SPACING_M:.6f} m")
        print(f"Max good edge: {MAX_GOOD_EDGE_FACTOR:.2f} x spacing = {MAX_GOOD_EDGE_FACTOR * LATTICE_SPACING_M:.6f} m")
        print(f"Min node degree: {MIN_NODE_DEGREE}")
        print(f"Cleanup iterations: {CLEANUP_ITERATIONS}")

        print_pointcloud_diagnostics(lattice_points_clean, "Surface lattice after cleanup")

        print_lattice_spacing_diagnostics(
            lattice_points_clean,
            lattice_edges_clean,
            "Lattice spacing after cleanup",
        )

        final_lattice_points = lattice_points_clean
        final_lattice_edges = lattice_edges_clean

    else:
        final_lattice_points = lattice_points
        final_lattice_edges = lattice_edges

    # lattice_cloud = make_open3d_cloud(final_lattice_points, color=(1.0, 0.1, 0.05))
    lattice_cloud = make_open3d_cloud(final_lattice_points, color=(0.0, 0.0, 0.0))
    geometries = [lattice_cloud]

    if SHOW_LATTICE_LINES:
        geometries.append(make_lattice_lines(final_lattice_points, final_lattice_edges))

    if VISUALIZE_SCAN_AND_LATTICE:
        scan_cloud = make_open3d_cloud(points_for_lattice, color=(0.55, 0.55, 0.55))
        visualize_geometries([scan_cloud, *geometries], "Trimmed scan with relaxed surface lattice")

    if VISUALIZE_LATTICE_ONLY:
        visualize_geometries(geometries, "Relaxed surface lattice only")


if __name__ == "__main__":
    main()
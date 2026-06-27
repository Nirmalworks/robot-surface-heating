#!/usr/bin/env python3
# pyright: reportInvalidTypeForm=false

import os
import json
import numpy as np
import trimesh
import matplotlib.pyplot as plt
import warp as wp


# STL_NAME = "RSS_v3.stl"
STL_NAME = "saddle_mold.STL"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# STL_PATH = os.path.join(BASE_DIR, "..", "stl", STL_NAME)
STL_PATH = os.path.join(BASE_DIR, STL_NAME)

# UP_AXIS = "Y"
UP_AXIS = "Z"
STL_SCALE_TO_METERS = 0.001

L_SPACING_M = 0.02
PLANE_MARGIN_M = 0.0

RELAX_ITERS = 1000
RELAX_OMEGA = 0.05
ANCHOR_W = 0.20
MAX_STEP_FRAC = 0.08
REPROJECT_EVERY = 1
FIX_BOUNDARY_PLANE = True

PLOT_STRIDE_POINTS = 1
PLOT_STRIDE_EDGES = 1

NPZ_CACHE_NAME = "cad_lattice_cache.npz"
NPZ_CACHE_PATH = os.path.join(BASE_DIR, "..", "npz", NPZ_CACHE_NAME)
# NPZ_CACHE_PATH = os.path.join(BASE_DIR, "npz", NPZ_CACHE_NAME)
SAVE_NPZ_CACHE = False
FORCE_REBUILD_CACHE = False


def make_edges_rect(Ny: int, Nx: int) -> np.ndarray:
    edges = []
    for iy in range(Ny):
        for ix in range(Nx):
            i = iy * Nx + ix
            if ix + 1 < Nx:
                edges.append((i, i + 1))
            if iy + 1 < Ny:
                edges.append((i, i + Nx))
    return np.array(edges, dtype=np.int32)


def make_grid_neighbors(Ny: int, Nx: int) -> np.ndarray:
    n = Ny * Nx
    nbr = -np.ones((n, 4), dtype=np.int32)

    for iy in range(Ny):
        for ix in range(Nx):
            i = iy * Nx + ix
            k = 0
            if ix - 1 >= 0:
                nbr[i, k] = i - 1
                k += 1
            if ix + 1 < Nx:
                nbr[i, k] = i + 1
                k += 1
            if iy - 1 >= 0:
                nbr[i, k] = i - Nx
                k += 1
            if iy + 1 < Ny:
                nbr[i, k] = i + Nx
                k += 1

    return nbr


def axis_config(up_axis: str) -> tuple[int, tuple[int, int]]:
    a = up_axis.upper().strip()
    if a == "X":
        return 0, (1, 2)
    if a == "Y":
        return 1, (0, 2)
    return 2, (0, 1)


def ensure_ray_intersector(mesh: trimesh.Trimesh):
    try:
        from trimesh.ray.ray_pyembree import RayMeshIntersector  # type: ignore
        return RayMeshIntersector(mesh)
    except Exception:
        return mesh.ray


def snap_grid_to_top_surface(
    mesh: trimesh.Trimesh,
    A: np.ndarray,
    B: np.ndarray,
    *,
    up_idx: int,
    plane_idx: tuple[int, int],
    up_pad: float,
) -> tuple[np.ndarray, np.ndarray]:
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")

    ab = np.stack([A.reshape(-1), B.reshape(-1)], axis=1).astype(np.float64)
    n = ab.shape[0]

    bounds = mesh.bounds.astype(np.float64)
    up_top = float(bounds[1, up_idx])
    origin_up = up_top + float(up_pad)

    origins = np.zeros((n, 3), dtype=np.float64)
    origins[:, plane_idx[0]] = ab[:, 0]
    origins[:, plane_idx[1]] = ab[:, 1]
    origins[:, up_idx] = origin_up

    directions = np.zeros((n, 3), dtype=np.float64)
    directions[:, up_idx] = -1.0

    intersector = ensure_ray_intersector(mesh)
    loc, idx_ray, _ = intersector.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=True,
    )

    P = np.zeros((n, 3), dtype=np.float64)
    valid = np.zeros((n,), dtype=bool)

    if loc.shape[0] == 0:
        return P.astype(np.float32), valid

    up_vals = loc[:, up_idx].astype(np.float64)
    best_up = np.full((n,), -np.inf, dtype=np.float64)
    best_loc = np.zeros((n, 3), dtype=np.float64)

    for p, r, u in zip(loc, idx_ray, up_vals):
        ri = int(r)
        if u > best_up[ri]:
            best_up[ri] = u
            best_loc[ri] = p

    valid = np.isfinite(best_up) & (best_up > -np.inf)
    P[valid] = best_loc[valid]
    return P.astype(np.float32), valid


def snap_grid_to_top_surface_with_normals(
    mesh: trimesh.Trimesh,
    A: np.ndarray,
    B: np.ndarray,
    *,
    up_idx: int,
    plane_idx: tuple[int, int],
    up_pad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")

    ab = np.stack([A.reshape(-1), B.reshape(-1)], axis=1).astype(np.float64)
    n = ab.shape[0]

    bounds = mesh.bounds.astype(np.float64)
    up_top = float(bounds[1, up_idx])
    origin_up = up_top + float(up_pad)

    origins = np.zeros((n, 3), dtype=np.float64)
    origins[:, plane_idx[0]] = ab[:, 0]
    origins[:, plane_idx[1]] = ab[:, 1]
    origins[:, up_idx] = origin_up

    directions = np.zeros((n, 3), dtype=np.float64)
    directions[:, up_idx] = -1.0

    intersector = ensure_ray_intersector(mesh)
    loc, idx_ray, idx_tri = intersector.intersects_location(
        ray_origins=origins,
        ray_directions=directions,
        multiple_hits=True,
    )

    P = np.zeros((n, 3), dtype=np.float64)
    N = np.zeros((n, 3), dtype=np.float64)
    valid = np.zeros((n,), dtype=bool)

    if loc.shape[0] == 0:
        return P.astype(np.float32), N.astype(np.float32), valid

    up_vals = loc[:, up_idx].astype(np.float64)
    best_up = np.full((n,), -np.inf, dtype=np.float64)
    best_loc = np.zeros((n, 3), dtype=np.float64)
    best_tri = np.full((n,), -1, dtype=np.int64)

    for p, r, u, t in zip(loc, idx_ray, up_vals, idx_tri):
        ri = int(r)
        if u > best_up[ri]:
            best_up[ri] = u
            best_loc[ri] = p
            best_tri[ri] = int(t)

    valid = np.isfinite(best_up) & (best_up > -np.inf) & (best_tri >= 0)
    P[valid] = best_loc[valid]

    face_n = mesh.face_normals.astype(np.float64)
    tri = best_tri[valid]
    N[valid] = face_n[tri]

    nn = np.linalg.norm(N[valid], axis=1)
    good = nn > 1e-12
    Nn = np.zeros_like(N[valid])
    Nn[good] = N[valid][good] / nn[good, None]
    N[valid] = Nn

    c = mesh.centroid.astype(np.float64)
    v = P[valid] - c[None, :]
    s = np.sum(N[valid] * v, axis=1)
    flip = s < 0.0
    if np.any(flip):
        N_sub = N[valid]
        N_sub[flip] *= -1.0
        N[valid] = N_sub

    return P.astype(np.float32), N.astype(np.float32), valid


def lattice_cache_meta(
    *,
    stl_path: str,
    stl_scale_to_meters: float,
    up_axis: str,
    l_spacing_m: float,
    plane_margin_m: float,
    relax_iters: int,
    relax_omega: float,
    anchor_w: float,
    max_step_frac: float,
    reproject_every: int,
    fix_boundary_plane: bool,
) -> dict:
    try:
        st = os.stat(stl_path)
        stl_mtime = int(st.st_mtime)
        stl_size = int(st.st_size)
    except Exception:
        stl_mtime = -1
        stl_size = -1

    return {
        "stl_path": stl_path,
        "stl_mtime": stl_mtime,
        "stl_size": stl_size,
        "stl_scale_to_meters": float(stl_scale_to_meters),
        "up_axis": str(up_axis),
        "l_spacing_m": float(l_spacing_m),
        "plane_margin_m": float(plane_margin_m),
        "relax_iters": int(relax_iters),
        "relax_omega": float(relax_omega),
        "anchor_w": float(anchor_w),
        "max_step_frac": float(max_step_frac),
        "reproject_every": int(reproject_every),
        "fix_boundary_plane": bool(fix_boundary_plane),
    }


def save_lattice_npz(npz_path: str, points: np.ndarray, normals: np.ndarray, meta: dict):
    os.makedirs(os.path.dirname(npz_path), exist_ok=True)
    np.savez_compressed(
        npz_path,
        points=points.astype(np.float32),
        normals=normals.astype(np.float32),
        meta_json=json.dumps(meta),
    )


def load_lattice_npz(npz_path: str) -> tuple[np.ndarray, np.ndarray, dict]:
    z = np.load(npz_path, allow_pickle=False)
    points = z["points"].astype(np.float32)
    normals = z["normals"].astype(np.float32)
    meta_json = str(z["meta_json"].tolist()) if "meta_json" in z else "{}"
    try:
        meta = json.loads(meta_json)
    except Exception:
        meta = {}
    return points, normals, meta


@wp.kernel
def relax_kernel(
    plane: wp.array(dtype=wp.float32, ndim=2),
    plane0: wp.array(dtype=wp.float32, ndim=2),
    world: wp.array(dtype=wp.float32, ndim=2),
    neighbors: wp.array(dtype=wp.int32, ndim=2),
    boundary: wp.array(dtype=wp.uint8),
    valid: wp.array(dtype=wp.uint8),
    L: float,
    omega: float,
    anchor_w: float,
    max_step: float,
    delta_out: wp.array(dtype=wp.float32, ndim=2),
):
    tid = wp.tid()

    if valid[tid] == 0:
        delta_out[tid, 0] = 0.0
        delta_out[tid, 1] = 0.0
        return

    if boundary[tid] != 0:
        delta_out[tid, 0] = 0.0
        delta_out[tid, 1] = 0.0
        return

    da = wp.float32(0.0)
    db = wp.float32(0.0)
    deg = wp.float32(0.0)

    for t in range(4):
        nb = neighbors[tid, t]
        if nb < 0:
            continue
        if valid[nb] == 0:
            continue

        dx = world[nb, 0] - world[tid, 0]
        dy = world[nb, 1] - world[tid, 1]
        dz = world[nb, 2] - world[tid, 2]
        dist = wp.sqrt(dx * dx + dy * dy + dz * dz)
        if dist < 1e-9:
            continue

        err = dist - wp.float32(L)

        pa = plane[nb, 0] - plane[tid, 0]
        pb = plane[nb, 1] - plane[tid, 1]
        plen = wp.sqrt(pa * pa + pb * pb)
        if plen < 1e-12:
            continue

        ua = pa / plen
        ub = pb / plen

        da = da + wp.float32(omega) * err * ua
        db = db + wp.float32(omega) * err * ub
        deg = deg + wp.float32(1.0)

    if deg > wp.float32(0.0):
        da = da / deg
        db = db / deg

    da = da + wp.float32(anchor_w) * (plane0[tid, 0] - plane[tid, 0])
    db = db + wp.float32(anchor_w) * (plane0[tid, 1] - plane[tid, 1])

    step = wp.sqrt(da * da + db * db)
    if step > wp.float32(max_step) and step > 1e-12:
        s = wp.float32(max_step) / step
        da = da * s
        db = db * s

    plane[tid, 0] = plane[tid, 0] + da
    plane[tid, 1] = plane[tid, 1] + db

    delta_out[tid, 0] = da
    delta_out[tid, 1] = db


def load_or_build_lattice_points_normals(
    *,
    stl_path: str = STL_PATH,
    npz_path: str = NPZ_CACHE_PATH,
    force_rebuild: bool = False,
    save_cache: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
    meta = lattice_cache_meta(
        stl_path=stl_path,
        stl_scale_to_meters=float(STL_SCALE_TO_METERS),
        up_axis=str(UP_AXIS),
        l_spacing_m=float(L_SPACING_M),
        plane_margin_m=float(PLANE_MARGIN_M),
        relax_iters=int(RELAX_ITERS),
        relax_omega=float(RELAX_OMEGA),
        anchor_w=float(ANCHOR_W),
        max_step_frac=float(MAX_STEP_FRAC),
        reproject_every=int(REPROJECT_EVERY),
        fix_boundary_plane=bool(FIX_BOUNDARY_PLANE),
    )

    if (not force_rebuild) and os.path.isfile(npz_path):
        try:
            pts, nrm, m0 = load_lattice_npz(npz_path)
            return pts, nrm, {**m0, **meta}
        except Exception:
            pass

    if not os.path.isfile(stl_path):
        raise FileNotFoundError(f"STL path not found: {stl_path}")

    up_idx, plane_idx = axis_config(UP_AXIS)

    mesh = trimesh.load_mesh(stl_path, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("Loaded file did not produce a Trimesh")

    scale = float(STL_SCALE_TO_METERS)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("STL_SCALE_TO_METERS must be positive and finite")

    mesh.apply_scale(scale)
    try:
        mesh.process(validate=True)
    except Exception:
        mesh = mesh.copy()

    bounds = mesh.bounds.astype(np.float64)

    a0 = float(bounds[0, plane_idx[0]]) - float(PLANE_MARGIN_M)
    a1 = float(bounds[1, plane_idx[0]]) + float(PLANE_MARGIN_M)
    b0 = float(bounds[0, plane_idx[1]]) - float(PLANE_MARGIN_M)
    b1 = float(bounds[1, plane_idx[1]]) + float(PLANE_MARGIN_M)

    L = float(L_SPACING_M)
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("L_SPACING_M must be positive and finite")

    Nx = int(np.floor((a1 - a0) / max(L, 1e-12))) + 1
    Ny = int(np.floor((b1 - b0) / max(L, 1e-12))) + 1

    av = a0 + np.arange(Nx, dtype=np.float64) * L
    bv = b0 + np.arange(Ny, dtype=np.float64) * L
    A, B = np.meshgrid(av, bv, indexing="xy")

    neighbors = make_grid_neighbors(Ny, Nx)

    up_range = float(bounds[1, up_idx] - bounds[0, up_idx])
    up_pad = max(0.05, 0.25 * up_range + 0.05)

    P0, valid0 = snap_grid_to_top_surface(mesh, A, B, up_idx=up_idx, plane_idx=plane_idx, up_pad=up_pad)

    boundary = np.zeros((Ny, Nx), dtype=np.uint8)
    if FIX_BOUNDARY_PLANE:
        boundary[0, :] = 1
        boundary[-1, :] = 1
        boundary[:, 0] = 1
        boundary[:, -1] = 1
    boundary_flat = boundary.reshape(-1)

    plane0 = np.stack([A.reshape(-1), B.reshape(-1)], axis=1).astype(np.float32)
    plane_init = plane0.copy()

    wp.init()
    dev = wp.get_preferred_device()

    neighbors_dev = wp.array(neighbors, dtype=wp.int32, device=dev).reshape((neighbors.shape[0], 4))
    boundary_dev = wp.array(boundary_flat, dtype=wp.uint8, device=dev)
    valid_dev = wp.array(valid0.astype(np.uint8), dtype=wp.uint8, device=dev)

    plane_dev = wp.array(plane_init, dtype=wp.float32, device=dev).reshape((plane_init.shape[0], 2))
    plane0_dev = wp.array(plane0, dtype=wp.float32, device=dev).reshape((plane0.shape[0], 2))
    world_dev = wp.array(P0.astype(np.float32), dtype=wp.float32, device=dev).reshape((P0.shape[0], 3))
    delta_dev = wp.zeros((plane_init.shape[0], 2), dtype=wp.float32, device=dev)

    max_step = float(MAX_STEP_FRAC) * float(L)

    for it in range(int(RELAX_ITERS)):
        wp.launch(
            kernel=relax_kernel,
            dim=plane_init.shape[0],
            inputs=[
                plane_dev,
                plane0_dev,
                world_dev,
                neighbors_dev,
                boundary_dev,
                valid_dev,
                float(L),
                float(RELAX_OMEGA),
                float(ANCHOR_W),
                float(max_step),
                delta_dev,
            ],
            device=dev,
        )

        if int(REPROJECT_EVERY) > 0 and ((it + 1) % int(REPROJECT_EVERY) == 0):
            plane_host = plane_dev.numpy()
            Ar = plane_host[:, 0].reshape(Ny, Nx)
            Br = plane_host[:, 1].reshape(Ny, Nx)
            Pw, valid = snap_grid_to_top_surface(mesh, Ar, Br, up_idx=up_idx, plane_idx=plane_idx, up_pad=up_pad)
            world_dev = wp.array(Pw.astype(np.float32), dtype=wp.float32, device=dev).reshape((Pw.shape[0], 3))
            valid_dev = wp.array(valid.astype(np.uint8), dtype=wp.uint8, device=dev)

    plane_final = plane_dev.numpy()
    Ar = plane_final[:, 0].reshape(Ny, Nx)
    Br = plane_final[:, 1].reshape(Ny, Nx)
    P_w, N_w, valid = snap_grid_to_top_surface_with_normals(mesh, Ar, Br, up_idx=up_idx, plane_idx=plane_idx, up_pad=up_pad)

    pts = P_w[valid].astype(np.float32)
    nrm = N_w[valid].astype(np.float32)

    if save_cache:
        try:
            save_lattice_npz(npz_path, pts, nrm, meta)
        except Exception:
            pass

    return pts, nrm, meta


def edge_length_stats(P_world: np.ndarray, edges: np.ndarray, valid: np.ndarray) -> tuple[float, float, float, np.ndarray]:
    vi = valid[edges[:, 0]]
    vj = valid[edges[:, 1]]
    keep = vi & vj
    if not np.any(keep):
        return float("nan"), float("nan"), float("nan"), np.zeros((0,), dtype=np.float32)

    e = edges[keep]
    Pi = P_world[e[:, 0]]
    Pj = P_world[e[:, 1]]
    d = np.linalg.norm(Pj - Pi, axis=1)
    return float(d.min()), float(d.mean()), float(d.max()), d.astype(np.float32)


def set_axes_equal_3d(ax, P: np.ndarray):
    x0, x1 = float(P[:, 0].min()), float(P[:, 0].max())
    y0, y1 = float(P[:, 1].min()), float(P[:, 1].max())
    z0, z1 = float(P[:, 2].min()), float(P[:, 2].max())

    xr = max(x1 - x0, 1e-9)
    yr = max(y1 - y0, 1e-9)
    zr = max(z1 - z0, 1e-9)
    r = max(xr, yr, zr)

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    cz = 0.5 * (z0 + z1)

    ax.set_xlim(cx - 0.5 * r, cx + 0.5 * r)
    ax.set_ylim(cy - 0.5 * r, cy + 0.5 * r)
    ax.set_zlim(cz - 0.5 * r, cz + 0.5 * r)


def plot_lattice_3d(P_world: np.ndarray, edges: np.ndarray, valid: np.ndarray, *, stride_points: int, stride_edges: int):
    Pv = P_world[valid]
    if Pv.shape[0] == 0:
        raise RuntimeError("No valid surface points to plot")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    sp = max(1, int(stride_points))
    pts = Pv[::sp]
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=6)

    se = max(1, int(stride_edges))
    vi = valid[edges[:, 0]]
    vj = valid[edges[:, 1]]
    ekeep = edges[vi & vj]

    for k in range(0, ekeep.shape[0], se):
        i, j = int(ekeep[k, 0]), int(ekeep[k, 1])
        ax.plot(
            [P_world[i, 0], P_world[j, 0]],
            [P_world[i, 1], P_world[j, 1]],
            [P_world[i, 2], P_world[j, 2]],
            linewidth=0.6,
        )

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title("STL top-surface lattice (after relaxation)")

    set_axes_equal_3d(ax, Pv)
    plt.tight_layout()
    plt.show()


def main():
    if not os.path.isfile(STL_PATH):
        raise FileNotFoundError(f"STL_PATH not found: {STL_PATH}")

    up_idx, plane_idx = axis_config(UP_AXIS)

    mesh = trimesh.load_mesh(STL_PATH, force="mesh")
    if not isinstance(mesh, trimesh.Trimesh):
        raise RuntimeError("Loaded file did not produce a Trimesh")

    scale = float(STL_SCALE_TO_METERS)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("STL_SCALE_TO_METERS must be positive and finite")

    mesh.apply_scale(scale)
    try:
        mesh.process(validate=True)
    except Exception:
        mesh = mesh.copy()

    bounds = mesh.bounds.astype(np.float64)
    dims = (bounds[1] - bounds[0]).astype(np.float64)

    axis_names = ["X", "Y", "Z"]
    print(f"[INFO] Loaded STL: {STL_PATH}")
    print(f"[INFO] UP_AXIS={UP_AXIS} (up_idx={up_idx})  plane_axes={axis_names[plane_idx[0]]}{axis_names[plane_idx[1]]}")
    print(f"[INFO] STL_SCALE_TO_METERS={scale:g} (applied to mesh)")
    print(
        "[INFO] Bounding box (meters): "
        f"min=({bounds[0,0]:.6f},{bounds[0,1]:.6f},{bounds[0,2]:.6f}) "
        f"max=({bounds[1,0]:.6f},{bounds[1,1]:.6f},{bounds[1,2]:.6f})"
    )
    print(f"[INFO] Bounding box dimensions (meters): dx={dims[0]:.6f} dy={dims[1]:.6f} dz={dims[2]:.6f}")

    a0 = float(bounds[0, plane_idx[0]]) - float(PLANE_MARGIN_M)
    a1 = float(bounds[1, plane_idx[0]]) + float(PLANE_MARGIN_M)
    b0 = float(bounds[0, plane_idx[1]]) - float(PLANE_MARGIN_M)
    b1 = float(bounds[1, plane_idx[1]]) + float(PLANE_MARGIN_M)

    L = float(L_SPACING_M)
    if not np.isfinite(L) or L <= 0.0:
        raise ValueError("L_SPACING_M must be positive and finite")

    Nx = int(np.floor((a1 - a0) / max(L, 1e-12))) + 1
    Ny = int(np.floor((b1 - b0) / max(L, 1e-12))) + 1

    av = a0 + np.arange(Nx, dtype=np.float64) * L
    bv = b0 + np.arange(Ny, dtype=np.float64) * L
    A, B = np.meshgrid(av, bv, indexing="xy")

    edges = make_edges_rect(Ny, Nx)
    neighbors = make_grid_neighbors(Ny, Nx)

    up_range = float(bounds[1, up_idx] - bounds[0, up_idx])
    up_pad = max(0.05, 0.25 * up_range + 0.05)

    print(f"[INFO] Grid: Nx={Nx} Ny={Ny} target_spacing={L:.6f} m  plane_margin={PLANE_MARGIN_M:.3f} m")

    P0, valid0 = snap_grid_to_top_surface(mesh, A, B, up_idx=up_idx, plane_idx=plane_idx, up_pad=up_pad)
    dmin0, dmean0, dmax0, _ = edge_length_stats(P0, edges, valid0)
    print(f"[INFO] Initial snapped spacing (3D edges): min/mean/max = {dmin0:.6f} / {dmean0:.6f} / {dmax0:.6f} (m)")
    print(f"[INFO] Initial valid nodes: {int(np.sum(valid0))}/{valid0.size} ({100.0*float(np.mean(valid0)):.1f}%)")

    boundary = np.zeros((Ny, Nx), dtype=np.uint8)
    if FIX_BOUNDARY_PLANE:
        boundary[0, :] = 1
        boundary[-1, :] = 1
        boundary[:, 0] = 1
        boundary[:, -1] = 1
    boundary_flat = boundary.reshape(-1)

    plane0 = np.stack([A.reshape(-1), B.reshape(-1)], axis=1).astype(np.float32)
    plane_init = plane0.copy()

    wp.init()
    dev = wp.get_preferred_device()

    neighbors_dev = wp.array(neighbors, dtype=wp.int32, device=dev).reshape((neighbors.shape[0], 4))
    boundary_dev = wp.array(boundary_flat, dtype=wp.uint8, device=dev)
    valid_dev = wp.array(valid0.astype(np.uint8), dtype=wp.uint8, device=dev)

    plane_dev = wp.array(plane_init, dtype=wp.float32, device=dev).reshape((plane_init.shape[0], 2))
    plane0_dev = wp.array(plane0, dtype=wp.float32, device=dev).reshape((plane0.shape[0], 2))
    world_dev = wp.array(P0.astype(np.float32), dtype=wp.float32, device=dev).reshape((P0.shape[0], 3))
    delta_dev = wp.zeros((plane_init.shape[0], 2), dtype=wp.float32, device=dev)

    max_step = float(MAX_STEP_FRAC) * float(L)

    for it in range(int(RELAX_ITERS)):
        wp.launch(
            kernel=relax_kernel,
            dim=plane_init.shape[0],
            inputs=[
                plane_dev,
                plane0_dev,
                world_dev,
                neighbors_dev,
                boundary_dev,
                valid_dev,
                float(L),
                float(RELAX_OMEGA),
                float(ANCHOR_W),
                float(max_step),
                delta_dev,
            ],
            device=dev,
        )

        if int(REPROJECT_EVERY) > 0 and ((it + 1) % int(REPROJECT_EVERY) == 0):
            plane_host = plane_dev.numpy()
            Ar = plane_host[:, 0].reshape(Ny, Nx)
            Br = plane_host[:, 1].reshape(Ny, Nx)
            Pw, valid = snap_grid_to_top_surface(mesh, Ar, Br, up_idx=up_idx, plane_idx=plane_idx, up_pad=up_pad)
            world_dev = wp.array(Pw.astype(np.float32), dtype=wp.float32, device=dev).reshape((Pw.shape[0], 3))
            valid_dev = wp.array(valid.astype(np.uint8), dtype=wp.uint8, device=dev)

    plane_final = plane_dev.numpy()
    Ar = plane_final[:, 0].reshape(Ny, Nx)
    Br = plane_final[:, 1].reshape(Ny, Nx)
    P_w, N_w, valid = snap_grid_to_top_surface_with_normals(mesh, Ar, Br, up_idx=up_idx, plane_idx=plane_idx, up_pad=up_pad)

    dmin, dmean, dmax, d_all = edge_length_stats(P_w, edges, valid)
    print(f"[RESULT] After relaxation spacing (3D edges): min/mean/max = {dmin:.6f} / {dmean:.6f} / {dmax:.6f} (m)")
    print(f"[RESULT] Final valid nodes: {int(np.sum(valid))}/{valid.size} ({100.0*float(np.mean(valid)):.1f}%)")

    if d_all.shape[0] > 0:
        err = d_all - L
        rms = float(np.sqrt(np.mean(err * err)))
        print(f"[RESULT] Edge error vs target: mean={float(err.mean()):.6e} m  rms={rms:.6e} m")

    if SAVE_NPZ_CACHE:
        pts = P_w[valid].astype(np.float32)
        nrm = N_w[valid].astype(np.float32)
        meta = lattice_cache_meta(
            stl_path=STL_PATH,
            stl_scale_to_meters=scale,
            up_axis=UP_AXIS,
            l_spacing_m=L,
            plane_margin_m=PLANE_MARGIN_M,
            relax_iters=RELAX_ITERS,
            relax_omega=RELAX_OMEGA,
            anchor_w=ANCHOR_W,
            max_step_frac=MAX_STEP_FRAC,
            reproject_every=REPROJECT_EVERY,
            fix_boundary_plane=FIX_BOUNDARY_PLANE,
        )
        try:
            if FORCE_REBUILD_CACHE or (not os.path.isfile(NPZ_CACHE_PATH)):
                save_lattice_npz(NPZ_CACHE_PATH, pts, nrm, meta)
                print(f"[INFO] Saved lattice cache: {NPZ_CACHE_PATH}")
        except Exception as e:
            print(f"[WARN] Failed to save lattice cache: {e}")

    plot_lattice_3d(
        P_w,
        edges,
        valid,
        stride_points=int(PLOT_STRIDE_POINTS),
        stride_edges=int(PLOT_STRIDE_EDGES),
    )


if __name__ == "__main__":
    main()
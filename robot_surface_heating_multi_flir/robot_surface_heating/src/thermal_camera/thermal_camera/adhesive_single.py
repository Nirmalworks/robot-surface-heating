#!/usr/bin/env python3
# pyright: reportInvalidTypeForm=false

"""
batch_branch.py
===============

Same behavior as branch_horizon.py, but optimizes multiple branches in parallel (batched)
on the same device using Warp kernels.

All plotting and selection behavior is unchanged. Only the branch optimization is batched.
"""

import os
import time
import numpy as np
import warp as wp
import matplotlib.pyplot as plt

from thermal_camera.learned_model_util import ThermalModel


# ============================================================
# Synthetic temperature field (Celsius)
# ============================================================

def gaussian_2d(X, Y, x0, y0, sx, sy, amp):
    return amp * np.exp(-0.5 * (((X - x0) / sx) ** 2 + ((Y - y0) / sy) ** 2))


def smooth_noise_field(Ny, Nx, rng, num_bumps=24):
    y = np.arange(Ny, dtype=np.float32)
    x = np.arange(Nx, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="xy")

    field = np.zeros((Ny, Nx), dtype=np.float32)
    for _ in range(num_bumps):
        x0 = rng.uniform(0, Nx - 1)
        y0 = rng.uniform(0, Ny - 1)
        sx = rng.uniform(0.08 * Nx, 0.18 * Nx)
        sy = rng.uniform(0.08 * Ny, 0.18 * Ny)
        amp = rng.normal(0.0, 1.0)
        field += gaussian_2d(X, Y, x0, y0, sx, sy, amp)

    field -= field.mean()
    denom = float(np.max(np.abs(field)) + 1e-9)
    field /= denom
    return field


def make_initial_temperature_C(
    Nx: int,
    Ny: int,
    seed: int = 4,
    Tmin_C: float = 30.0,
    Tmax_C: float = 45.0,
    num_hotspots: int = 5,
):
    rng = np.random.default_rng(seed)

    y = np.arange(Ny, dtype=np.float32)
    x = np.arange(Nx, dtype=np.float32)
    X, Y = np.meshgrid(x, y, indexing="xy")

    baseline = 32.0
    drift = 1.0 * smooth_noise_field(Ny, Nx, rng, num_bumps=24)
    small_noise = rng.normal(0.0, 0.08, size=(Ny, Nx)).astype(np.float32)
    T = baseline + drift + small_noise

    for _ in range(num_hotspots):
        x0 = rng.uniform(0.15 * (Nx - 1), 0.85 * (Nx - 1))
        y0 = rng.uniform(0.15 * (Ny - 1), 0.85 * (Ny - 1))
        sx = rng.uniform(0.05 * Nx, 0.10 * Nx)
        sy = rng.uniform(0.05 * Ny, 0.10 * Ny)
        amp = rng.uniform(4.0, 7.0)
        T += gaussian_2d(X, Y, x0, y0, sx, sy, amp)

    tmin = float(T.min())
    tmax = float(T.max())
    T = (T - tmin) / ((tmax - tmin) + 1e-9)
    T = Tmin_C + (Tmax_C - Tmin_C) * T
    return T.astype(np.float32)


# ============================================================
# B-spline basis / weights (host)
# ============================================================

def bspline_basis_one(u, i, p, U):
    if p == 0:
        if (U[i] <= u < U[i + 1]) or (u == U[-1] and U[i + 1] == U[-1]):
            return 1.0
        return 0.0

    d1 = U[i + p] - U[i]
    d2 = U[i + p + 1] - U[i + 1]

    t1 = 0.0
    t2 = 0.0
    if d1 > 1e-12:
        t1 = (u - U[i]) / d1 * bspline_basis_one(u, i, p - 1, U)
    if d2 > 1e-12:
        t2 = (U[i + p + 1] - u) / d2 * bspline_basis_one(u, i + 1, p - 1, U)
    return t1 + t2


def bspline_weights_at_u(num_ctrl, degree, u, knot_vector):
    U = np.asarray(knot_vector, dtype=np.float64)
    w = np.zeros((num_ctrl,), dtype=np.float64)
    for i in range(num_ctrl):
        w[i] = bspline_basis_one(float(u), i, int(degree), U)
    s = w.sum()
    if s > 1e-12:
        w /= s
    return w.astype(np.float32)


def spline_eval_from_W(W, ctrl):
    return W @ ctrl


def cells_to_meters_xy(cx, cy, x_min, y_min, L):
    return x_min + cx * L, y_min + cy * L


def compute_u_schedule_from_initial_spline(ctrl_cells, knots, degree, L_spacing, v_mps, dt, T_total, dense=1200):
    # Number of discrete time steps we will sample along the trajectory
    H = int(round(T_total / dt))

    # Build a dense set of spline parameters u so we can approximate arc length
    u_dense = np.linspace(float(knots[degree]), float(knots[len(ctrl_cells)]), dense, dtype=np.float32)

    # For each u, compute spline basis weights (W) and evaluate the spline point in cell coords
    W_dense = np.stack([bspline_weights_at_u(5, degree, u, knots) for u in u_dense], axis=0)
    pts_dense = spline_eval_from_W(W_dense, ctrl_cells)

    # Approximate cumulative distance along the spline by summing segment lengths (convert cells -> meters)
    diffs_cells = pts_dense[1:] - pts_dense[:-1]
    seglen_m = np.linalg.norm(diffs_cells, axis=1) * L_spacing
    s_m = np.concatenate([[0.0], np.cumsum(seglen_m)], axis=0)

    # Desired travel distance at each time step if we move at constant speed v
    s_targets = np.arange(H, dtype=np.float32) * (v_mps * dt)

    # Clamp so we do not request distance beyond the end of the path
    s_targets = np.clip(s_targets, 0.0, float(s_m[-1]))

    # Convert "distance along path" targets into corresponding spline parameters u
    # by locating which dense segment each target distance falls into
    u_sched = np.zeros((H,), dtype=np.float32)
    for k in range(H):
        st = float(s_targets[k])

        # Find segment index j where s_m[j] <= st < s_m[j+1]
        j = int(np.searchsorted(s_m, st, side="right") - 1)
        j = max(0, min(j, dense - 2))

        # Linearly interpolate u within that segment based on how far st is between s0 and s1
        s0 = float(s_m[j])
        s1 = float(s_m[j + 1])
        t = 0.0 if (s1 - s0) < 1e-12 else (st - s0) / (s1 - s0)
        u_sched[k] = float(u_dense[j] + t * (u_dense[j + 1] - u_dense[j]))

    # Build the actual basis matrix W for the scheduled u's:
    # used later as pts = W @ ctrl
    W_sched = np.stack([bspline_weights_at_u(5, degree, u, knots) for u in u_sched], axis=0)

    # Return:
    # - u values per timestep
    # - weights per timestep (H x 5)
    # - estimated total path length (meters)
    return u_sched, W_sched, float(s_m[-1])

# ============================================================
# Branching target selection
# ============================================================

def pick_k_targets_cold_exclusion(
    T_C: np.ndarray,
    start_cells: np.ndarray,
    L_spacing: float,
    r_excl_start_m: float,
    r_excl_targets_m: float,
    K: int,
    allowed_mask: np.ndarray,   # (Ny,Nx) bool-like, True where targets are allowed
):
    Ny, Nx = T_C.shape
    xs = np.arange(Nx, dtype=np.float32)
    ys = np.arange(Ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    dx0 = (X - float(start_cells[0])) * L_spacing
    dy0 = (Y - float(start_cells[1])) * L_spacing
    dist0 = np.sqrt(dx0 * dx0 + dy0 * dy0)
    # mask = dist0 >= float(r_excl_start_m)
    mask = (dist0 >= float(r_excl_start_m)) & (allowed_mask.astype(bool))

    targets_ixiy: list[tuple[int, int]] = []
    targets_cells: list[np.ndarray] = []
    targets_T: list[float] = []

    for _ in range(K):
        if not np.any(mask):
            break

        T_masked = np.where(mask, T_C, np.inf).astype(np.float32)
        idx = int(np.argmin(T_masked))
        iy, ix = np.unravel_index(idx, (Ny, Nx))

        if not np.isfinite(T_masked[iy, ix]):
            break

        targets_ixiy.append((int(ix), int(iy)))
        targets_cells.append(np.array([float(ix), float(iy)], dtype=np.float32))
        targets_T.append(float(T_C[iy, ix]))

        dx = (X - float(ix)) * L_spacing
        dy = (Y - float(iy)) * L_spacing
        dist = np.sqrt(dx * dx + dy * dy)
        mask = mask & (dist >= float(r_excl_targets_m))

    return targets_ixiy, targets_cells, targets_T


# ============================================================
# Warp kernels (batched)
# ============================================================

@wp.func
def softplus(x: float) -> float:
    return wp.log(1.0 + wp.exp(x))


@wp.kernel
def spline_kernel_batched(
    W: wp.array2d(dtype=wp.float32),       # (H,5)
    ctrl: wp.array2d(dtype=wp.vec2),       # (B,5)
    pts: wp.array2d(dtype=wp.vec2),        # (B,H)
    H: int,
):
    t = wp.tid()
    b = t // H
    k = t - b * H

    p = wp.vec2(0.0, 0.0)
    for i in range(5):
        p += W[k, i] * ctrl[b, i]
    pts[b, k] = p


@wp.kernel
def heat_added_to_hot_cost_kernel_batched(
    T0: wp.array2d(dtype=wp.float32),      # (Ny,Nx) Kelvin
    pts: wp.array2d(dtype=wp.vec2),        # (B,H) heater positions in *cell coords*
    Nx: int,
    Ny: int,

    L_spacing: float,
    sigma_m: float,
    h_peak: float,
    dt: float,

    T_hot_K: float,
    inv_scale: float,

    J: wp.array(dtype=wp.float32),         # (B,)
    H: int,
):
    t = wp.tid()
    b = t // H
    k = t - b * H

    p = pts[b, k]
    hx = p[0]
    hy = p[1]

    R_CELLS = 4
    inv_sigma2 = float(1.0) / (sigma_m * sigma_m + float(1.0e-12))

    acc = float(0.0)

    cx = int(wp.floor(hx))
    cy = int(wp.floor(hy))

    for oy in range(-R_CELLS, R_CELLS + 1):
        iy = cy + oy
        if iy < 0 or iy >= Ny:
            continue

        for ox in range(-R_CELLS, R_CELLS + 1):
            ix = cx + ox
            if ix < 0 or ix >= Nx:
                continue

            dx_cells = float(ix) - hx
            dy_cells = float(iy) - hy
            dx_m = dx_cells * L_spacing
            dy_m = dy_cells * L_spacing
            r2 = dx_m * dx_m + dy_m * dy_m

            w = wp.exp(float(-0.5) * r2 * inv_sigma2)
            heat = h_peak * dt * w

            z = (T0[iy, ix] - T_hot_K) * inv_scale
            hot = softplus(z)
            hotness = hot * hot

            acc = acc + heat * hotness

    wp.atomic_add(J, b, acc)


@wp.kernel
def curvature_cost_kernel_batched(
    pts: wp.array2d(dtype=wp.vec2),        # (B,H)
    J: wp.array(dtype=wp.float32),         # (B,)
    H: int,
):
    t = wp.tid()
    b = t // H
    k = t - b * H

    if k <= 0 or k >= H - 1:
        return

    d2 = pts[b, k + 1] - 2.0 * pts[b, k] + pts[b, k - 1]
    wp.atomic_add(J, b, d2[0] * d2[0] + d2[1] * d2[1])

# @wp.kernel
# def screen_keepout_cost_kernel_batched(
#     pts: wp.array2d(dtype=wp.vec2),        # (B,H)
#     mask_screen: wp.array2d(dtype=wp.uint8),  # (Ny,Nx), 1 if screen
#     Nx: int,
#     Ny: int,
#     J: wp.array(dtype=wp.float32),         # (B,)
#     H: int,
# ):
#     t = wp.tid()
#     b = t // H
#     k = t - b * H

#     p = pts[b, k]
#     hx = p[0]
#     hy = p[1]

#     ix = int(wp.round(hx))
#     iy = int(wp.round(hy))

#     if ix < 0:
#         ix = 0
#     if ix > Nx - 1:
#         ix = Nx - 1
#     if iy < 0:
#         iy = 0
#     if iy > Ny - 1:
#         iy = Ny - 1

#     if mask_screen[iy, ix] != 0:
#         wp.atomic_add(J, b, 1.0)
@wp.kernel
def screen_keepout_cost_kernel_batched(
    pts: wp.array2d(dtype=wp.vec2),
    mask_screen: wp.array2d(dtype=wp.float32),
    Nx: int,
    Ny: int,
    J: wp.array(dtype=wp.float32),
    H: int,
):
    t = wp.tid()
    b = t // H
    k = t - b * H

    p = pts[b, k]
    hx = p[0]
    hy = p[1]

    outside_x = float(0.0)
    outside_y = float(0.0)

    if hx < 0.0:
        outside_x = -hx
        hx = 0.0
    elif hx > float(Nx - 1):
        outside_x = hx - float(Nx - 1)
        hx = float(Nx - 1)

    if hy < 0.0:
        outside_y = -hy
        hy = 0.0
    elif hy > float(Ny - 1):
        outside_y = hy - float(Ny - 1)
        hy = float(Ny - 1)

    x0 = int(wp.floor(hx))
    y0 = int(wp.floor(hy))
    x1 = x0 + 1
    y1 = y0 + 1

    if x1 >= Nx:
        x1 = Nx - 1
    if y1 >= Ny:
        y1 = Ny - 1

    tx = hx - float(x0)
    ty = hy - float(y0)

    c00 = mask_screen[y0, x0]
    c10 = mask_screen[y0, x1]
    c01 = mask_screen[y1, x0]
    c11 = mask_screen[y1, x1]

    c0 = c00 * (1.0 - tx) + c10 * tx
    c1 = c01 * (1.0 - tx) + c11 * tx
    c = c0 * (1.0 - ty) + c1 * ty

    outside_cost = outside_x * outside_x + outside_y * outside_y

    wp.atomic_add(J, b, c + outside_cost)

@wp.kernel
def combine_costs_kernel_batched(
    J_heat: wp.array(dtype=wp.float32),
    J_curv: wp.array(dtype=wp.float32),
    J_screen: wp.array(dtype=wp.float32),
    w_heat: float,
    w_curv: float,
    w_screen: float,
    J_out: wp.array(dtype=wp.float32),
):
    b = wp.tid()
    J_out[b] = w_heat * J_heat[b] + w_curv * J_curv[b] + w_screen * J_screen[b]


@wp.kernel
def sum_costs_kernel(
    J_in: wp.array(dtype=wp.float32),
    J_out: wp.array(dtype=wp.float32),
):
    b = wp.tid()
    wp.atomic_add(J_out, 0, J_in[b])


# ============================================================
# Adam optimizer (host)
# ============================================================

def adam_update_batch(x, g, m, v, t, lr):
    b1, b2, eps = 0.9, 0.999, 1e-8
    m[:] = b1 * m + (1 - b1) * g
    v[:] = b2 * v + (1 - b2) * (g * g)
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    x[:] -= lr * mhat / (np.sqrt(vhat) + eps)


# ============================================================
# Batched optimization over targets
# ============================================================

def optimize_to_targets_batched(
    *,
    T0_C: np.ndarray,
    T0_dev_2d: wp.array,
    mask_screen_dev: wp.array,     # (Ny,Nx) uint8
    device,
    P0: np.ndarray,
    P4_list: list[np.ndarray],
    sigma_m: float,
    h_peak: float,
    L: float,
    dt: float,
    T_total: float,
    v_mps: float,
    T_hot_K: float,
    inv_scale: float,
    w_heat: float,
    w_curv: float,
    w_screen: float,
    iters: int,
    lr: float,
):
    # Grid sizes and rollout length (H time steps in the planned trajectory)
    Ny, Nx = T0_C.shape
    H = int(round(T_total / dt))

    # Number of branches. One branch per candidate endpoint target (P4).
    B = len(P4_list)

    # Stack all endpoints into one array so we can initialize all branches in one pass.
    P4_arr = np.stack(P4_list, axis=0).astype(np.float32)  # (B,2)

    # Initialize each branch with a simple straight-line 5-control-point spline:
    # P0, then 3 interior points evenly spaced, then P4.
    ctrl_init = np.zeros((B, 5, 2), dtype=np.float32)
    for b in range(B):
        P4 = P4_arr[b]
        d = P4 - P0
        ctrl_init[b, 0] = P0
        ctrl_init[b, 1] = P0 + 0.25 * d
        ctrl_init[b, 2] = P0 + 0.50 * d
        ctrl_init[b, 3] = P0 + 0.75 * d
        ctrl_init[b, 4] = P4

    # Shared spline definition (degree + knots) used for every branch.
    degree = 3
    knots = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1], np.float32)

    # Build a time-sampled spline basis matrix W for the horizon:
    # W maps 5 control points -> H sampled positions along the spline.
    # Important: W is shared across all branches to keep the GPU work batched.
    _, W_sched_np, _ = compute_u_schedule_from_initial_spline(
        ctrl_cells=ctrl_init[0],
        knots=knots,
        degree=degree,
        L_spacing=L,
        v_mps=v_mps,
        dt=dt,
        T_total=T_total,
        dense=1200,
    )
    W_dev = wp.array(W_sched_np, dtype=wp.float32, device=device).reshape((H, 5))

    # Optimization variables: only the 3 interior control points (P1..P3).
    # Endpoints P0 and P4 stay fixed.
    x = ctrl_init[:, 1:4, :].reshape(B, 6).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    # Store cost histories per branch, and final cost values at the end.
    J_hist = [[] for _ in range(B)]
    J_final = np.full((B,), np.inf, dtype=np.float32)

    # GPU buffer for the sampled trajectory points (B branches, H points each).
    pts_dev = wp.empty((B, H), dtype=wp.vec2, device=device, requires_grad=True)

    for it in range(1, iters + 1):
        # Rebuild the full control point sets from the current interior variables x.
        ctrl_host = ctrl_init.copy()
        ctrl_host[:, 1:4, :] = x.reshape(B, 3, 2)

        # Convert to Warp vectors for kernels (shape (B,5)).
        ctrl_flat = ctrl_host.reshape(-1, 2)
        ctrl_vec = [wp.vec2(float(p[0]), float(p[1])) for p in ctrl_flat]
        ctrl_dev = wp.array(ctrl_vec, dtype=wp.vec2, device=device, requires_grad=True).reshape((B, 5))

        # Per-branch cost terms (all computed on GPU).
        # Scalar costs per branch, then combined into J_total
        J_heat_dev = wp.zeros(B, dtype=wp.float32, device=device, requires_grad=True)
        J_curv_dev = wp.zeros(B, dtype=wp.float32, device=device, requires_grad=True)
        J_screen_dev = wp.zeros(B, dtype=wp.float32, device=device, requires_grad=True)
        J_total_dev = wp.zeros(B, dtype=wp.float32, device=device, requires_grad=True)

        # Sums all branches, one backward pass later
        J_sum_dev = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        tape = wp.Tape()
        with tape:
            # Evaluate all branch splines at H time steps: ctrl -> pts_dev
            wp.launch(
                spline_kernel_batched,
                dim=B * H,
                inputs=[W_dev, ctrl_dev, pts_dev, H],
                device=device,
            )

            # Heat cost: penalize depositing heat in already-hot regions
            # (uses T0 snapshot, Gaussian heater footprint, and "hotness" shaping)
            wp.launch(
                heat_added_to_hot_cost_kernel_batched,
                dim=B * H,
                inputs=[
                    T0_dev_2d,
                    pts_dev,
                    Nx,
                    Ny,
                    float(L),
                    float(sigma_m),
                    float(h_peak),
                    float(dt),
                    float(T_hot_K),
                    float(inv_scale),
                    J_heat_dev,
                    H,
                ],
                device=device,
            )

            # Curvature cost: discourages sharp turns in the sampled path
            wp.launch(
                curvature_cost_kernel_batched,
                dim=B * H,
                inputs=[pts_dev, J_curv_dev, H],
                device=device,
            )

            # Keep-out cost: penalize any sampled point that lands inside screen mask
            wp.launch(
                screen_keepout_cost_kernel_batched,
                dim=B * H,
                inputs=[
                    pts_dev,
                    mask_screen_dev,
                    Nx,
                    Ny,
                    J_screen_dev,
                    H,
                ],
                device=device,
            )

            # Weighted sum of the three costs -> one objective per branch
            wp.launch(
                combine_costs_kernel_batched,
                dim=B,
                inputs=[
                    J_heat_dev,
                    J_curv_dev,
                    J_screen_dev,
                    float(w_heat),
                    float(w_curv),
                    float(w_screen),
                    J_total_dev,
                ],
                device=device,
            )

            # Sum across branches into a single scalar so Tape can backprop once
            wp.launch(
                sum_costs_kernel,
                dim=B,
                inputs=[J_total_dev, J_sum_dev],
                device=device,
            )

        # Backprop gradients from the summed objective to control points
        tape.backward(J_sum_dev)

        # Pull costs back to CPU for logging and branch selection.
        J_np = J_total_dev.numpy().astype(np.float32)
        J_final = J_np.copy()
        for b in range(B):
            J_hist[b].append(float(J_np[b]))

        # Pull gradients of ctrl_dev back to CPU, then keep only interior point grads
        g = tape.gradients[ctrl_dev].numpy()  # (B,5) vec2
        g_xy = np.zeros((B, 5, 2), dtype=np.float32)
        for b in range(B):
            for i in range(5):
                g_xy[b, i, 0] = float(g[b, i][0])
                g_xy[b, i, 1] = float(g[b, i][1])

        # Only optimize P1..P3, so we update those 6 variables per branch
        g_int = g_xy[:, 1:4, :].reshape(B, 6)
        adam_update_batch(x, g_int, m, v, it, lr)

    # Build final optimized control points for all branches
    ctrl_opt = ctrl_init.copy()
    ctrl_opt[:, 1:4, :] = x.reshape(B, 3, 2)

    # Evaluate final splines on CPU using the same W schedule
    spline_opt = np.einsum("hk,bkc->bhc", W_sched_np, ctrl_opt).astype(np.float32)

    # Force the last sampled point to match the intended endpoint exactly
    spline_opt[:, -1, :] = P4_arr.astype(np.float32)

    return ctrl_opt, spline_opt, J_final.astype(np.float32), J_hist, W_sched_np


# ============================================================
# Main
# ============================================================

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # NPZ_PATH = os.path.join(BASE_DIR, "..", "npz", "11_22_25_learned_10mm.npz")
    NPZ_PATH = os.path.join(BASE_DIR, "..", "npz", "laptop_synthetic_10mm.npz")

    data = np.load(NPZ_PATH, allow_pickle=True)
    Nx = int(data["Nx"])
    Ny = int(data["Ny"])
    L = float(data["L_spacing"])
    x_min = float(data["x_min"])
    x_max = float(data["x_max"])
    y_min = float(data["y_min"])
    y_max = float(data["y_max"])
    mask_screen = data["mask_screen"].astype(np.uint8)      # (Ny,Nx), 1 inside screen
    mask_adhesive = data["mask_adhesive"].astype(np.uint8)  # (Ny,Nx), 1 in adhesive

    sigma_m = None
    if "sigma_cells" in data:
        s = data["sigma_cells"]
        try:
            sigma_cells_mean = float(np.mean(s))
            sigma_m = sigma_cells_mean * L
        except Exception:
            sigma_m = None
    if sigma_m is None:
        sigma_m = 0.02

    h_peak = None
    if "h_peak" in data:
        try:
            h_peak = float(np.mean(data["h_peak"]))
        except Exception:
            h_peak = None
    if h_peak is None:
        h_peak = 1.0

    model = ThermalModel.from_npz(NPZ_PATH)
    device = model.device
    wp.init()

    # v_mps = 0.1
    v_mps = 0.05 # slower for laptop model
    dt = 0.25
    T_total = 4.0
    H = int(round(T_total / dt))
    model.dt = dt

    # num_branches = 10
    # num_branches = 20
    # r_excl_start = 0.1
    # r_excl_targets = 0.1

    # laptop heating params
    num_branches = 8
    r_excl_start = 0.03
    r_excl_targets = 0.03

    hot_percentile = 90.0
    scale_K = 0.75
    inv_scale = 1.0 / scale_K

    w_curv = 1.0
    w_heat = 1e-3
    w_screen = 1.0

    iters = 200
    lr = 0.02

    T0_C = make_initial_temperature_C(
        Nx=Nx,
        Ny=Ny,
        # seed=10,
        # seed=1,
        # seed=2,
        # seed=3,
        seed=4,
        # seed=5,
        # seed=9,
        Tmin_C=30.0,
        Tmax_C=40.0,
        num_hotspots=5,
    )
    T0_K = (T0_C + 273.15).astype(np.float32)

    T_hot_C = float(np.percentile(T0_C, hot_percentile))
    T_hot_K = T_hot_C + 273.15
    frac_hot = float(np.mean(T0_C >= T_hot_C))

    print(f"[INFO] Nx={Nx} Ny={Ny} L={L} dt={model.dt} H={H} v={v_mps:.3f} m/s  T={T_total:.2f} s")
    print(
        f"[INFO] T0_C: min={T0_C.min():.2f} max={T0_C.max():.2f} mean={T0_C.mean():.2f} | "
        f"hot%={100.0*frac_hot:.2f}% (>=p{hot_percentile:.0f}={T_hot_C:.2f}C)"
    )
    print(f"[INFO] heater params: sigma_m={sigma_m:.5f}  h_peak={h_peak:.6g} (from NPZ if available)")

    T0_dev_2d = wp.array(T0_K.reshape(-1), dtype=wp.float32, device=device).reshape((Ny, Nx))
    mask_screen_dev = wp.array(mask_screen, dtype=wp.uint8, device=device).reshape((Ny, Nx))

    P0 = np.array([0.18 * (Nx - 1), 0.18 * (Ny - 1)], np.float32)

    targets_ixiy, targets_cells, targets_T = pick_k_targets_cold_exclusion(
        T0_C, P0, L, r_excl_start, r_excl_targets, num_branches,
        allowed_mask=mask_adhesive,
    )

    if len(targets_cells) == 0:
        raise RuntimeError("No feasible targets found under exclusion constraints.")

    print("[INFO] Branch targets:")
    for k, ((ix, iy), p, tC) in enumerate(zip(targets_ixiy, targets_cells, targets_T)):
        dxm = (p[0] - P0[0]) * L
        dym = (p[1] - P0[1]) * L
        dist_m = float(np.sqrt(dxm * dxm + dym * dym))
        print(
            f"  - k={k+1}: (ix,iy)=({ix},{iy})  cells=({p[0]:.1f},{p[1]:.1f})  "
            f"T={tC:.2f}C  dist_from_start={dist_m:.3f}m"
        )

    start_batch = time.perf_counter()

    ctrl_opt_b, spline_opt_b, J_final_b, J_hist_b, _W = optimize_to_targets_batched(
        T0_C=T0_C,
        T0_dev_2d=T0_dev_2d,
        mask_screen_dev=mask_screen_dev,
        device=device,
        P0=P0,
        P4_list=targets_cells,
        sigma_m=sigma_m,
        h_peak=h_peak,
        L=L,
        dt=dt,
        T_total=T_total,
        v_mps=v_mps,
        T_hot_K=T_hot_K,
        inv_scale=inv_scale,
        w_heat=w_heat,
        w_curv=w_curv,
        w_screen=w_screen,
        iters=iters,
        lr=lr,
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
            f"[INFO] branch {k+1}/{len(targets_cells)} done: J_final={float(J_final_b[k]):.6f}  "
            f"target=({targets_cells[k][0]:.0f},{targets_cells[k][1]:.0f})"
        )

    best_idx = int(np.argmin([b["J_final"] for b in branch_results]))
    best = branch_results[best_idx]
    print(
        f"[INFO] BEST branch = {best_idx+1}  J={best['J_final']:.6f}  "
        f"target=({best['P4'][0]:.0f},{best['P4'][1]:.0f})  [{batch_time:.3f} s]"
    )

    T_final = wp.array(T0_K.reshape(-1), dtype=wp.float32, device=device)
    for k in range(H):
        T_final = model.step(T_final, best["spline_opt"][k])

    Tfinal_C = (T_final.numpy().reshape(Ny, Nx) - 273.15).astype(np.float32)

    def spline_to_world_xy(spline_cells):
        sx = x_min + spline_cells[:, 0] * L
        sy = y_min + spline_cells[:, 1] * L
        return sx, sy

    p0_x, p0_y = cells_to_meters_xy(P0[0], P0[1], x_min, y_min, L)

    plt.figure(figsize=(9.0, 6.8))
    im = plt.imshow(
        T0_C,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap="inferno",
        interpolation="nearest",
        aspect="equal",
    )
    plt.colorbar(im, label="Temperature [°C]")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Branching Candidates (initial temperature field)")

    plt.scatter([p0_x], [p0_y], s=120, marker="o", label="Start (P0)")

    for b in branch_results:
        sx, sy = spline_to_world_xy(b["spline_opt"])
        P4 = b["P4"]
        tx, ty = cells_to_meters_xy(P4[0], P4[1], x_min, y_min, L)

        if b is best:
            plt.plot(sx, sy, linewidth=3.2, label=f"Best spline (branch {b['k']+1})")
            plt.scatter([tx], [ty], s=140, marker="X", label="Best target")
        else:
            plt.plot(sx, sy, linewidth=1.6, alpha=0.75, label=f"Spline branch {b['k']+1}")
            plt.scatter([tx], [ty], s=80, marker="x", alpha=0.85)

    plt.legend(loc="upper right", fontsize=9)
    plt.tight_layout()

    plt.figure(figsize=(8.5, 4.5))
    for b in branch_results:
        plt.plot(
            b["J_hist"],
            linewidth=2.0 if b is best else 1.2,
            alpha=1.0 if b is best else 0.7,
            label=f"branch {b['k']+1} (J={b['J_final']:.3g})",
        )
    plt.xlabel("Iteration")
    plt.ylabel("J_total")
    plt.title("Optimization histories (all branches)")
    plt.legend()
    plt.tight_layout()

    plt.figure(figsize=(9.0, 6.8))
    im2 = plt.imshow(
        Tfinal_C,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap="inferno",
        interpolation="nearest",
        aspect="equal",
    )
    plt.colorbar(im2, label="Temperature [°C]")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Final temperature after executing BEST branch")

    sx, sy = spline_to_world_xy(best["spline_opt"])
    plt.plot(sx, sy, linewidth=3.0, label="Executed spline (best)")
    plt.legend()
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()

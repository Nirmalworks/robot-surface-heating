#!/usr/bin/env python3
# pyright: reportInvalidTypeForm=false

import os
import numpy as np
import warp as wp
import matplotlib.pyplot as plt

from learned_model_util import ThermalModel


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
    H = int(round(T_total / dt))
    u_dense = np.linspace(float(knots[degree]), float(knots[len(ctrl_cells)]), dense, dtype=np.float32)

    W_dense = np.stack([bspline_weights_at_u(5, degree, u, knots) for u in u_dense], axis=0)  # (dense,5)
    pts_dense = spline_eval_from_W(W_dense, ctrl_cells)  # (dense,2) in cells

    diffs_cells = pts_dense[1:] - pts_dense[:-1]
    seglen_m = np.linalg.norm(diffs_cells, axis=1) * L_spacing
    s_m = np.concatenate([[0.0], np.cumsum(seglen_m)], axis=0)

    s_targets = np.arange(H, dtype=np.float32) * (v_mps * dt)
    s_targets = np.clip(s_targets, 0.0, float(s_m[-1]))

    u_sched = np.zeros((H,), dtype=np.float32)
    for k in range(H):
        st = float(s_targets[k])
        j = int(np.searchsorted(s_m, st, side="right") - 1)
        j = max(0, min(j, dense - 2))
        s0 = float(s_m[j])
        s1 = float(s_m[j + 1])
        t = 0.0 if (s1 - s0) < 1e-12 else (st - s0) / (s1 - s0)
        u_sched[k] = float(u_dense[j] + t * (u_dense[j + 1] - u_dense[j]))

    W_sched = np.stack([bspline_weights_at_u(5, degree, u, knots) for u in u_sched], axis=0)  # (H,5)
    return u_sched, W_sched, float(s_m[-1])


# ============================================================
# Endpoint selection: coldest outside exclusion radius
# ============================================================

def pick_coldest_outside_radius_cells(T_C: np.ndarray, start_cells: np.ndarray, L_spacing: float, r_exclude_m: float):
    Ny, Nx = T_C.shape
    xs = np.arange(Nx, dtype=np.float32)
    ys = np.arange(Ny, dtype=np.float32)
    X, Y = np.meshgrid(xs, ys, indexing="xy")

    dx = (X - float(start_cells[0])) * L_spacing
    dy = (Y - float(start_cells[1])) * L_spacing
    dist = np.sqrt(dx * dx + dy * dy)

    mask = dist >= float(r_exclude_m)
    if not np.any(mask):
        raise RuntimeError("Exclusion radius masks out all cells")

    T_masked = np.where(mask, T_C, np.inf).astype(np.float32)
    idx = int(np.argmin(T_masked))
    iy, ix = np.unravel_index(idx, (Ny, Nx))

    end_cells = np.array([float(ix), float(iy)], dtype=np.float32)
    end_T = float(T_C[iy, ix])
    end_dist = float(dist[iy, ix])
    return end_cells, end_T, end_dist


# ============================================================
# Warp kernels (helpers + spline + NEW cost)
# ============================================================

@wp.func
def clampf(x: float, lo: float, hi: float) -> float:
    return wp.max(lo, wp.min(hi, x))


@wp.func
def lerp(a: float, b: float, t: float) -> float:
    return a + t * (b - a)


@wp.func
def softplus(x: float) -> float:
    return wp.log(1.0 + wp.exp(x))


@wp.kernel
def spline_kernel(
    W: wp.array2d(dtype=wp.float32),   # (H,5)
    ctrl: wp.array(dtype=wp.vec2),     # (5,)
    pts: wp.array(dtype=wp.vec2),      # (H,)
    H: int,
):
    k = wp.tid()
    if k >= H:
        return

    p = wp.vec2(0.0, 0.0)
    for i in range(5):
        p += W[k, i] * ctrl[i]
    pts[k] = p


@wp.kernel
def heat_added_to_hot_cost_kernel(
    T0: wp.array2d(dtype=wp.float32),    # (Ny,Nx) Kelvin
    pts: wp.array(dtype=wp.vec2),        # (H,) heater positions in *cell coords*
    Nx: int,
    Ny: int,

    L_spacing: float,                    # meters per cell
    sigma_m: float,                      # Gaussian sigma in meters
    h_peak: float,                       # peak heating rate (learned units)
    dt: float,                           # seconds

    T_hot_K: float,
    inv_scale: float,                    # 1/scale_K for softplus

    J: wp.array(dtype=wp.float32),       # (1,)
):
    k = wp.tid()
    p = pts[k]
    hx = p[0]
    hy = p[1]

    R_CELLS = 4

    inv_sigma2 = float(1.0) / (sigma_m * sigma_m + float(1.0e-12))

    # dynamic variable (mutable in loops)
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

    wp.atomic_add(J, 0, acc)

@wp.kernel
def curvature_cost_kernel(
    pts: wp.array(dtype=wp.vec2),      # (H,)
    H: int,
    J: wp.array(dtype=wp.float32),     # (1,)
):
    k = wp.tid()
    if k <= 0 or k >= H - 1:
        return

    d2 = pts[k + 1] - 2.0 * pts[k] + pts[k - 1]
    wp.atomic_add(J, 0, d2[0] * d2[0] + d2[1] * d2[1])


@wp.kernel
def combine_costs_kernel(
    J_heat: wp.array(dtype=wp.float32),
    J_curv: wp.array(dtype=wp.float32),
    w_heat: float,
    w_curv: float,
    J_out: wp.array(dtype=wp.float32),
):
    if wp.tid() == 0:
        J_out[0] = w_heat * J_heat[0] + w_curv * J_curv[0]

# ============================================================
# Adam optimizer
# ============================================================

def adam_update(x, g, m, v, t, lr):
    b1, b2, eps = 0.9, 0.999, 1e-8
    m[:] = b1 * m + (1 - b1) * g
    v[:] = b2 * v + (1 - b2) * (g * g)
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    x[:] -= lr * mhat / (np.sqrt(vhat) + eps)


# ============================================================
# Main
# ============================================================

def main():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    NPZ_PATH = os.path.join(BASE_DIR, "..", "npz", "11_22_25_learned_10mm.npz")

    data = np.load(NPZ_PATH, allow_pickle=True)
    Nx = int(data["Nx"])
    Ny = int(data["Ny"])
    L = float(data["L_spacing"])
    x_min = float(data["x_min"])
    x_max = float(data["x_max"])
    y_min = float(data["y_min"])
    y_max = float(data["y_max"])

    # Try to pull heater params from NPZ (fallbacks are safe + explicit)
    # sigma_cells is commonly stored in "cell units"; convert to meters via L.
    sigma_m = None
    if "sigma_cells" in data:
        s = data["sigma_cells"]
        try:
            sigma_cells_mean = float(np.mean(s))
            sigma_m = sigma_cells_mean * L
        except Exception:
            sigma_m = None

    if sigma_m is None:
        # fallback: a conservative ~2cm footprint
        sigma_m = 0.02

    h_peak = None
    if "h_peak" in data:
        try:
            h_peak = float(np.mean(data["h_peak"]))
        except Exception:
            h_peak = None

    if h_peak is None:
        # fallback: 1.0 in learned-model units (still optimizes, just scaled)
        h_peak = 1.0

    model = ThermalModel.from_npz(NPZ_PATH)
    device = model.device
    wp.init()

    v_mps = 0.1
    dt = 0.25
    T_total = 4.0
    H = int(round(T_total / dt))

    model.dt = dt

    T0_C = make_initial_temperature_C(
        Nx=Nx,
        Ny=Ny,
        seed=9,
        # seed=1,
        # seed=2,
        Tmin_C=30.0,
        Tmax_C=45.0,
        num_hotspots=5,
    )

    T0_K = (T0_C + 273.15).astype(np.float32)

    # T_hot_C = 38.0
    # T_hot_K = T_hot_C + 273.15
    # frac_hot = float(np.mean(T0_C > T_hot_C))

    # dynamic "hot" threshold via percentile of the current snapshot
    hot_percentile = 90.0 
    T_hot_C = float(np.percentile(T0_C, hot_percentile))
    T_hot_K = T_hot_C + 273.15

    frac_hot = float(np.mean(T0_C >= T_hot_C))

    print(f"[INFO] Nx={Nx} Ny={Ny} L={L} dt={model.dt} H={H} v={v_mps:.3f} m/s  T={T_total:.2f} s")
    # print(f"[INFO] T0_C: min={T0_C.min():.2f} max={T0_C.max():.2f} mean={T0_C.mean():.2f} | hot%={100.0*frac_hot:.2f}% (>{T_hot_C:.1f}C)")
    print(
        f"[INFO] T0_C: min={T0_C.min():.2f} max={T0_C.max():.2f} mean={T0_C.mean():.2f} | "
        f"hot%={100.0*frac_hot:.2f}% (>=p{hot_percentile:.0f}={T_hot_C:.2f}C)"
    )
    print(f"[INFO] heater params: sigma_m={sigma_m:.5f}  h_peak={h_peak:.6g}  (from NPZ if available)")

    T0_dev_2d = wp.array(T0_K.reshape(-1), dtype=wp.float32, device=device).reshape((Ny, Nx))

    # Start point stays the same (cells)
    P0 = np.array([0.18 * (Nx - 1), 0.18 * (Ny - 1)], np.float32)

    # End point: coldest outside exclusion radius from start
    r_excl = 0.15
    P4, T_end_C, dist_end_m = pick_coldest_outside_radius_cells(T0_C, P0, L, r_excl)
    print(f"[INFO] end_cells=({P4[0]:.1f},{P4[1]:.1f})  end_T={T_end_C:.2f}C  end_dist={dist_end_m:.3f}m  r_excl={r_excl:.3f}m")

    # Initialize interior points on the line P0->P4
    d = P4 - P0
    ctrl_init_cells = np.stack(
        [P0, P0 + 0.25 * d, P0 + 0.50 * d, P0 + 0.75 * d, P4], axis=0
    ).astype(np.float32)

    degree = 3
    knots = np.array([0, 0, 0, 0, 0.5, 1, 1, 1, 1], np.float32)

    u_sched, W_sched_np, s_total_init = compute_u_schedule_from_initial_spline(
        ctrl_cells=ctrl_init_cells,
        knots=knots,
        degree=degree,
        L_spacing=L,
        v_mps=v_mps,
        dt=dt,
        T_total=T_total,
        dense=1200,
    )

    print(f"[INFO] path_len_init={s_total_init:.3f} m  travel={v_mps*T_total:.3f} m")

    W_dev = wp.array(W_sched_np, dtype=wp.float32, device=device).reshape((H, 5))
    pts_dev = wp.empty(H, dtype=wp.vec2, device=device, requires_grad=True)

    # Hotness smoothing (same idea as before, but now weights heat deposition)
    scale_K = 0.75
    inv_scale = 1.0 / scale_K

    w_curv = 5.0
    w_heat = 1e-4

    iters = 160
    lr = 0.05

    x = ctrl_init_cells[1:4].reshape(-1).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    J_hist = []

    for it in range(1, iters + 1):
        ctrl_host = ctrl_init_cells.copy()
        ctrl_host[1:4] = x.reshape(3, 2)

        ctrl_dev = wp.array(
            [wp.vec2(float(p[0]), float(p[1])) for p in ctrl_host],
            dtype=wp.vec2,
            device=device,
            requires_grad=True,
        )
    
        # --- cost accumulators ---
        J_heat_dev  = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        J_curv_dev  = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)
        J_total_dev = wp.zeros(1, dtype=wp.float32, device=device, requires_grad=True)

        tape = wp.Tape()
        with tape:
            wp.launch(spline_kernel, dim=H, inputs=[W_dev, ctrl_dev, pts_dev, H], device=device)

            # (1) penalize heat deposited into already-hot regions
            wp.launch(
                heat_added_to_hot_cost_kernel,
                dim=H,
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
                ],
                device=device,
            )

            # (2) curvature regularization on the sampled path (discourages sharp turns)
            wp.launch(
                curvature_cost_kernel,
                dim=H,
                inputs=[pts_dev, H, J_curv_dev],
                device=device,
            )

            # (3) combine into one scalar objective for backprop
            wp.launch(combine_costs_kernel, dim=1,
                    inputs=[J_heat_dev, J_curv_dev, float(w_heat), float(w_curv), J_total_dev],
                    device=device)

        tape.backward(J_total_dev)

        J_heat  = float(J_heat_dev.numpy()[0])
        J_curv  = float(J_curv_dev.numpy()[0])
        J_total = float(J_total_dev.numpy()[0])
        J_hist.append(J_total)

        g = tape.gradients[ctrl_dev].numpy()
        g = np.array([[float(v0[0]), float(v0[1])] for v0 in g], np.float32)
        g_int = g[1:4].reshape(-1)

        adam_update(x, g_int, m, v, it, lr)

        if it == 1 or it % 20 == 0 or it == iters:
            print(
                f"[iter {it:04d}] Jtot={J_total: .6f} "
                f"(Jheat={J_heat: .6f}, Jcurv={J_curv: .6f}, w={w_curv:g}) "
                f"|g|={np.linalg.norm(g_int):.6e} "
                f"|gP1|={np.linalg.norm(g[1]):.6e} "
                f"|gP2|={np.linalg.norm(g[2]):.6e} "
                f"|gP3|={np.linalg.norm(g[3]):.6e}"
            )


    ctrl_opt_cells = ctrl_init_cells.copy()
    ctrl_opt_cells[1:4] = x.reshape(3, 2)

    spline_init_cells = W_sched_np @ ctrl_init_cells
    spline_opt_cells = W_sched_np @ ctrl_opt_cells

    diffs_m = np.linalg.norm((spline_opt_cells[1:] - spline_opt_cells[:-1]) * L, axis=1)
    print(f"[INFO] step_len_m: min={diffs_m.min():.4f} mean={diffs_m.mean():.4f} max={diffs_m.max():.4f}  target={v_mps*dt:.4f}")

    T_final = wp.array(T0_K.reshape(-1), dtype=wp.float32, device=device)
    for k in range(H):
        T_final = model.step(T_final, spline_opt_cells[k])

    Tfinal_C = (T_final.numpy().reshape(Ny, Nx) - 273.15).astype(np.float32)

    init_cp_x, init_cp_y = cells_to_meters_xy(ctrl_init_cells[:, 0], ctrl_init_cells[:, 1], x_min, y_min, L)
    opt_cp_x, opt_cp_y = cells_to_meters_xy(ctrl_opt_cells[:, 0], ctrl_opt_cells[:, 1], x_min, y_min, L)

    init_sp_x, init_sp_y = cells_to_meters_xy(spline_init_cells[:, 0], spline_init_cells[:, 1], x_min, y_min, L)
    opt_sp_x, opt_sp_y = cells_to_meters_xy(spline_opt_cells[:, 0], spline_opt_cells[:, 1], x_min, y_min, L)

    h0_x, h0_y = opt_sp_x[0], opt_sp_y[0]
    h1_x, h1_y = opt_sp_x[-1], opt_sp_y[-1]

    plt.figure()
    plt.plot(J_hist, linewidth=2.0)
    plt.xlabel("Iteration")
    plt.ylabel("Cost (heat deposited into hot regions)")
    plt.tight_layout()

    plt.figure(figsize=(8.6, 6.6))
    im = plt.imshow(
        T0_C,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap="inferno",
        vmin=30.0,
        vmax=45.0,
        interpolation="nearest",
        aspect="equal",
    )
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    cbar = plt.colorbar(im)
    cbar.set_label("Temperature [°C]")

    plt.plot(init_cp_x, init_cp_y, "--", linewidth=1.2, label="Init control polygon")
    plt.plot(init_sp_x, init_sp_y, linewidth=2.0, label="Init spline")
    plt.plot(opt_cp_x, opt_cp_y, "--", linewidth=1.2, label="Opt control polygon")
    plt.plot(opt_sp_x, opt_sp_y, linewidth=2.6, label="Opt spline")

    plt.scatter(init_cp_x[0], init_cp_y[0], s=90, marker="o", label="Start (P0)")
    plt.scatter(init_cp_x[-1], init_cp_y[-1], s=90, marker="X", label="End (P4)")
    plt.scatter(init_cp_x[1:-1], init_cp_y[1:-1], s=60, marker="s", label="Init P1..P3")
    plt.scatter(opt_cp_x[1:-1], opt_cp_y[1:-1], s=60, marker="D", label="Opt P1..P3")

    for i in range(5):
        plt.text(opt_cp_x[i], opt_cp_y[i], f"  P{i}", fontsize=10, va="center")

    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.figure(figsize=(8.6, 6.6))
    im2 = plt.imshow(
        Tfinal_C,
        origin="lower",
        extent=[x_min, x_max, y_min, y_max],
        cmap="inferno",
        vmin=30.0,
        vmax=45.0,
        interpolation="nearest",
        aspect="equal",
    )
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    cbar2 = plt.colorbar(im2)
    cbar2.set_label("Temperature [°C]")

    plt.plot(opt_cp_x, opt_cp_y, "--", linewidth=1.2, label="Opt control polygon")
    plt.plot(opt_sp_x, opt_sp_y, linewidth=2.6, label="Opt spline")
    plt.scatter(opt_cp_x[0], opt_cp_y[0], s=90, marker="o", label="P0")
    plt.scatter(opt_cp_x[-1], opt_cp_y[-1], s=90, marker="X", label="P4")
    plt.scatter(opt_cp_x[1:-1], opt_cp_y[1:-1], s=60, marker="D", label="P1..P3")
    plt.scatter(h0_x, h0_y, s=110, marker="o", label="Heater start")
    plt.scatter(h1_x, h1_y, s=110, marker="x", label="Heater end")

    for i in range(5):
        plt.text(opt_cp_x[i], opt_cp_y[i], f"  P{i}", fontsize=10, va="center")

    plt.legend(loc="upper right")
    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
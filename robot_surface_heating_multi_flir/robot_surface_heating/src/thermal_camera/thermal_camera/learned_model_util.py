#!/usr/bin/env python3
# pyright: reportInvalidTypeForm=false

"""
Thermal Model Utilities
=======================

Single-file utility for:
  - Loading learned thermal NPZ models
  - Building Warp-based diffusion simulation
  - Stepping simulation with Gaussian heater
  - Rolling out heater paths
  - Supporting MPC, spline optimization, and Rerun logging

NO control logic, NO MPC loops, NO optimizers.
This is a physics + geometry engine.
"""

import numpy as np
import warp as wp
from dataclasses import dataclass
from typing import Optional, Tuple

# ============================================================
# Warp kernels: diffusion
# ============================================================

@wp.kernel
def warp_Kx(
    ei: wp.array(dtype=int),
    ej: wp.array(dtype=int),
    alpha_e: wp.array(dtype=wp.float32),
    x: wp.array(dtype=wp.float32),
    y: wp.array(dtype=wp.float32),
):
    e = wp.tid()
    i = ei[e]
    j = ej[e]
    a = alpha_e[e]
    diff = x[i] - x[j]
    wp.atomic_add(y, i,  a * diff)
    wp.atomic_add(y, j, -a * diff)


@wp.kernel
def jacobi_step(
    dt: float,
    diag_K: wp.array(dtype=wp.float32),
    r_diag: wp.array(dtype=wp.float32),
    rhs: wp.array(dtype=wp.float32),
    x_old: wp.array(dtype=wp.float32),
    Kx: wp.array(dtype=wp.float32),
    x_new: wp.array(dtype=wp.float32),
):
    i = wp.tid()
    Aii = 1.0 + dt * (diag_K[i] + r_diag[i])
    Si  = diag_K[i] * x_old[i] - Kx[i]
    x_new[i] = (rhs[i] + dt * Si) / Aii


# ============================================================
# Warp kernels: heater
# ============================================================

@wp.kernel
def build_heater_terms_gaussian(
    cx: float,
    cy: float,
    sigma_cells: float,
    h_peak: float,
    A_over_V: float,
    rho_cp: float,
    r_nat: float,
    T_amb: float,
    T_gun: float,
    Xc: wp.array(dtype=wp.float32),
    Yc: wp.array(dtype=wp.float32),
    r_diag: wp.array(dtype=wp.float32),
    b_vec: wp.array(dtype=wp.float32),
):
    i = wp.tid()

    dx = Xc[i] - cx
    dy = Yc[i] - cy
    r2 = dx * dx + dy * dy

    sig2 = sigma_cells * sigma_cells + 1.0e-12
    G = wp.exp(-0.5 * r2 / sig2)

    r_gun = (h_peak * G * A_over_V) / rho_cp
    r_diag[i] = r_nat + r_gun
    b_vec[i]  = r_nat * T_amb + r_gun * T_gun


@wp.kernel
def build_rhs_kernel(
    T_old: wp.array(dtype=wp.float32),
    b_vec: wp.array(dtype=wp.float32),
    rhs: wp.array(dtype=wp.float32),
    dt: float,
):
    i = wp.tid()
    rhs[i] = T_old[i] + dt * b_vec[i]


# ============================================================
# Data containers
# ============================================================

@dataclass
class ThermalGrid:
    Nx: int
    Ny: int
    n: int
    L: float
    x_min: float
    y_min: float
    Xc: np.ndarray
    Yc: np.ndarray
    P_world: np.ndarray


# ============================================================
# Main thermal model class
# ============================================================

class ThermalModel:
    """
    Learned thermal diffusion + heater simulation.

    Stateless w.r.t control:
      - You pass heater positions explicitly
      - You manage MPC / planning outside this class
    """

    # --------------------------------------------------------
    # Construction
    # --------------------------------------------------------

    @classmethod
    def from_npz(cls, path: str, device: Optional[str] = None):
        data = np.load(path, allow_pickle=True)

        Nx = int(data["Nx"])
        Ny = int(data["Ny"])
        n  = Nx * Ny
        L  = float(data["L_spacing"])

        edges   = data["edges"].astype(np.int32)
        alpha_e = data["alpha_e"].astype(np.float32)

        diag_K = np.zeros(n, dtype=np.float32)
        for e, a in zip(edges, alpha_e):
            diag_K[e[0]] += a
            diag_K[e[1]] += a

        xs = np.arange(Nx, dtype=np.float32)
        ys = np.arange(Ny, dtype=np.float32)
        Xc, Yc = np.meshgrid(xs, ys, indexing="xy")

        xv = np.linspace(data["x_min"], data["x_max"], Nx, dtype=np.float32)
        yv = np.linspace(data["y_min"], data["y_max"], Ny, dtype=np.float32)
        Xw, Yw = np.meshgrid(xv, yv, indexing="xy")

        P_world = np.stack(
            [Xw.reshape(-1), Yw.reshape(-1), np.zeros(n)],
            axis=-1
        )

        grid = ThermalGrid(
            Nx=Nx,
            Ny=Ny,
            n=n,
            L=L,
            x_min=float(data["x_min"]),
            y_min=float(data["y_min"]),
            Xc=Xc.reshape(-1),
            Yc=Yc.reshape(-1),
            P_world=P_world,
        )

        return cls(
            grid=grid,
            edges=edges,
            alpha_e=alpha_e,
            diag_K=diag_K,
            sigma_cells=float(data["sigma_cells"]),
            h_peak=float(data["h_peak"]),
            dt=float(data["dt"]),
            ambient_K=float(data["ambient_C"] + 273.15),
            gun_K=float(data["gun_temp_C"] + 273.15),
            A_over_V=float(data["A_over_V"]),
            device=device,
        )

    # --------------------------------------------------------

    def __init__(
        self,
        *,
        grid: ThermalGrid,
        edges: np.ndarray,
        alpha_e: np.ndarray,
        diag_K: np.ndarray,
        sigma_cells: float,
        h_peak: float,
        dt: float,
        ambient_K: float,
        gun_K: float,
        A_over_V: float,
        device: Optional[str],
    ):
        wp.init()
        self.device = device or wp.get_preferred_device()

        self.grid = grid
        self.dt = dt
        self.sigma_cells = sigma_cells
        self.h_peak = h_peak
        self.A_over_V = A_over_V
        self.ambient_K = ambient_K
        self.gun_K = gun_K

        self.rho_cp = 550.0 * 1700.0
        self.r_nat = (8.0 * A_over_V) / self.rho_cp

        self.n = grid.n
        self.E = edges.shape[0]

        self.ei = wp.array(edges[:, 0], dtype=int, device=self.device)
        self.ej = wp.array(edges[:, 1], dtype=int, device=self.device)
        self.alpha_e = wp.array(alpha_e, dtype=wp.float32, device=self.device)
        self.diag_K = wp.array(diag_K, dtype=wp.float32, device=self.device)
        self.Xc = wp.array(grid.Xc, dtype=wp.float32, device=self.device)
        self.Yc = wp.array(grid.Yc, dtype=wp.float32, device=self.device)

    # ========================================================
    # Initialization
    # ========================================================

    def init_temperature(self, base_C=25.0, noise_std_C=0.0):
        T = base_C + 273.15 + noise_std_C * np.random.randn(self.n)
        return wp.array(T.astype(np.float32), device=self.device)

    # ========================================================
    # Coordinate transforms
    # ========================================================

    def world_to_cell(self, p):
        return np.array([
            (p[0] - self.grid.x_min) / self.grid.L,
            (p[1] - self.grid.y_min) / self.grid.L
        ], dtype=np.float32)

    def cell_to_world(self, c):
        return np.array([
            self.grid.x_min + c[0] * self.grid.L,
            self.grid.y_min + c[1] * self.grid.L
        ], dtype=np.float32)

    # ========================================================
    # One simulation step
    # ========================================================

    def step(self, T_dev, heater_cell):
        r_diag = wp.empty(self.n, dtype=wp.float32, device=self.device)
        b_vec  = wp.empty(self.n, dtype=wp.float32, device=self.device)

        wp.launch(
            build_heater_terms_gaussian,
            dim=self.n,
            inputs=[
                float(heater_cell[0]),
                float(heater_cell[1]),
                self.sigma_cells,
                self.h_peak,
                self.A_over_V,
                self.rho_cp,
                self.r_nat,
                self.ambient_K,
                self.gun_K,
                self.Xc,
                self.Yc,
                r_diag,
                b_vec,
            ],
            device=self.device,
        )

        rhs = wp.empty_like(T_dev)
        wp.launch(
            build_rhs_kernel,
            dim=self.n,
            inputs=[T_dev, b_vec, rhs, self.dt],
            device=self.device,
        )

        x_old = T_dev
        x_new = wp.empty_like(x_old)

        for _ in range(20):
            Kx = wp.zeros(self.n, dtype=wp.float32, device=self.device)
            wp.launch(
                warp_Kx,
                dim=self.E,
                inputs=[self.ei, self.ej, self.alpha_e, x_old, Kx],
                device=self.device,
            )
            wp.launch(
                jacobi_step,
                dim=self.n,
                inputs=[self.dt, self.diag_K, r_diag, rhs, x_old, Kx, x_new],
                device=self.device,
            )
            x_old, x_new = x_new, x_old

        return x_old

    # ========================================================
    # Rollout helper
    # ========================================================

    def rollout(self, T0, heater_path_cells):
        T = wp.clone(T0)
        for c in heater_path_cells:
            T = self.step(T, c)
        return T


# ============================================================
# Heater path utilities
# ============================================================

class HeaterPathUtils:
    @staticmethod
    def straight_line(start, end, steps):
        return np.linspace(start, end, steps, dtype=np.float32)

    @staticmethod
    def enforce_max_speed(path, max_step):
        out = [path[0]]
        for p in path[1:]:
            d = p - out[-1]
            n = np.linalg.norm(d)
            if n > max_step:
                d *= max_step / (n + 1e-8)
            out.append(out[-1] + d)
        return np.array(out, dtype=np.float32)
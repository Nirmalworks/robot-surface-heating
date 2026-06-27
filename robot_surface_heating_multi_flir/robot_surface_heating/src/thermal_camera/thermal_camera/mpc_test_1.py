#!/usr/bin/env python3
"""
mpc_snapshot_test.py
====================

One-shot MPC snapshot using real sensor data + real robot state.

Steps:
1. Read ONE thermal pointcloud snapshot
2. Convert thermal data -> Celsius
3. Map pointcloud to learned-model grid (Ny, Nx)
4. Query current EEF (heat_gun) pose
5. Convert EEF XY -> cell coordinates
6. Run single-horizon MPC optimization
7. Print results and exit
"""

import rclpy
from rclpy.node import Node

import numpy as np
import warp as wp

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from moveit_msgs.srv import GetPositionFK
from geometry_msgs.msg import PoseStamped

import tf2_ros
import tf2_geometry_msgs

from thermal_camera.learned_model_util import ThermalModel
import thermal_camera.single_horizon as sh


# ----------------------------
# Thermal conversion (given)
# ----------------------------
def ktoc(val):
    """Convert centi-Kelvin -> Celsius"""
    return (val - 27315.0) / 100.0


# ----------------------------
# MPC Snapshot Node
# ----------------------------
class MPCSnapshotNode(Node):
    def __init__(self):
        super().__init__("mpc_snapshot_test")

        # ----------------------------
        # Load learned model
        # ----------------------------
        NPZ_PATH = "11_22_25_learned_10mm.npz"
        self.model = ThermalModel.from_npz(NPZ_PATH)
        self.grid = self.model.grid

        self.Nx = self.grid.Nx
        self.Ny = self.grid.Ny
        self.L  = self.grid.L
        self.x_min = self.grid.x_min
        self.y_min = self.grid.y_min

        wp.init()
        self.get_logger().info(
            f"Loaded learned model: Nx={self.Nx}, Ny={self.Ny}, L={self.L}"
        )

        # ----------------------------
        # TF + FK
        # ----------------------------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.fk_client = self.create_client(GetPositionFK, "compute_fk")
        self.fk_client.wait_for_service()

        # ----------------------------
        # Pointcloud subscription (one-shot)
        # ----------------------------
        self.pc_sub = self.create_subscription(
            PointCloud2,
            "/selected_surface_points",
            self.pointcloud_callback,
            1,
        )

        self.snapshot_taken = False

    # ----------------------------
    # FK query
    # ----------------------------
    def get_heat_gun_xy_world(self):
        req = GetPositionFK.Request()
        req.header.frame_id = "world"
        req.fk_link_names = ["heat_gun"]

        future = self.fk_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)

        res = future.result()
        if res is None or not res.pose_stamped:
            raise RuntimeError("Failed to get FK for heat_gun")

        pose = res.pose_stamped[0].pose
        return pose.position.x, pose.position.y

    # ----------------------------
    # Pointcloud callback
    # ----------------------------
    def pointcloud_callback(self, msg: PointCloud2):
        if self.snapshot_taken:
            return

        self.snapshot_taken = True
        self.get_logger().info("Received thermal pointcloud snapshot")

        # ----------------------------
        # Build temperature grid
        # ----------------------------
        T0_C = np.full((self.Ny, self.Nx), np.nan, dtype=np.float32)
        hit_count = np.zeros((self.Ny, self.Nx), dtype=np.int32)

        for p in point_cloud2.read_points(
            msg,
            field_names=("x", "y", "thermal"),
            skip_nans=True,
        ):
            x, y, thermal = p
            T = ktoc(thermal)

            ix = int(round((x - self.x_min) / self.L))
            iy = int(round((y - self.y_min) / self.L))

            if 0 <= ix < self.Nx and 0 <= iy < self.Ny:
                if np.isnan(T0_C[iy, ix]):
                    T0_C[iy, ix] = T
                else:
                    T0_C[iy, ix] += T
                hit_count[iy, ix] += 1

        mask = hit_count > 0
        T0_C[mask] /= hit_count[mask]

        # ----------------------------
        # Handle missing cells (nearest fill)
        # ----------------------------
        missing = np.isnan(T0_C)
        if np.any(missing):
            self.get_logger().warn(
                f"{np.sum(missing)} grid cells missing thermal data — filling"
            )
            from scipy.ndimage import distance_transform_edt

            _, idx = distance_transform_edt(
                missing,
                return_indices=True,
            )
            T0_C = T0_C[tuple(idx)]

        # ----------------------------
        # Get heater start (cells)
        # ----------------------------
        hx, hy = self.get_heat_gun_xy_world()

        cx = (hx - self.x_min) / self.L
        cy = (hy - self.y_min) / self.L
        start_cells = np.array([cx, cy], dtype=np.float32)

        self.get_logger().info(
            f"Heater start (cells): [{start_cells[0]:.2f}, {start_cells[1]:.2f}]"
        )

        # ----------------------------
        # Run single-horizon MPC
        # ----------------------------
        ctrl_opt, spline_cells, target_ixiy, target_cells = sh.optimize_spline_path(
            T0_C=T0_C,
            model=self.model,
            start_cells=start_cells,
            v_mps=0.1,
        )

        # ----------------------------
        # Results
        # ----------------------------
        self.get_logger().info("MPC optimization complete")
        self.get_logger().info(f"Target cells: {target_cells}")
        self.get_logger().info(f"Spline length: {len(spline_cells)} points")

        # Print first few spline points (cells)
        for i in range(min(5, len(spline_cells))):
            p = spline_cells[i]
            self.get_logger().info(f"spline[{i}]: ({p[0]:.2f}, {p[1]:.2f})")

        self.get_logger().info("Snapshot MPC test finished. Shutting down.")
        rclpy.shutdown()


# ----------------------------
# Main
# ----------------------------
def main():
    rclpy.init()
    node = MPCSnapshotNode()
    rclpy.spin(node)


if __name__ == "__main__":
    main()

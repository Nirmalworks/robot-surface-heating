#!/usr/bin/env python3

import json
import os
import struct
import threading
import time
from typing import Any

import numpy as np
import open3d as o3d
import rclpy
from geometry_msgs.msg import Pose, PoseArray
from rclpy.node import Node
from scipy.spatial import cKDTree
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Bool, Float64, Float64MultiArray, Header, String
from std_srvs.srv import Trigger


# ============================================================
# User settings
# ============================================================

RAW_THERMAL_TOPIC = "/raw_thermal_pointcloud"
SELECTED_SURFACE_TOPIC = "/selected_surface_points"
BOUNDARY_TOPIC = "selected_boundary_points"
METRICS_TOPIC = "/thermal_roi_metrics"
PERCENT_IN_BAND_TOPIC = "/selected_surface_percent_in_band"
REPLAN_TRIGGER_TOPIC = "/heater_replan_trigger"
ROI_STATUS_TOPIC = "/multi_roi_status"

# Normal mode remains the original behavior: one selected rectangle is published forever.
# ENABLE_MULTI_ROI_MODE = True
ENABLE_MULTI_ROI_MODE = True

# Only used when ENABLE_MULTI_ROI_MODE=True
MULTI_ROI_COUNT = 3
MULTI_ROI_TARGET_PERCENT = 90.0
MULTI_ROI_SWITCH_COOLDOWN_S = 2.0
MULTI_ROI_REPLAN_ON_SWITCH = True
MULTI_ROI_ALLOW_REVISIT = False
MULTI_ROI_REVISIT_BELOW_PERCENT = 70.0

# Cache applies in both normal and multi-ROI mode.
USE_SELECTION_CACHE = True
SAVE_SELECTION_CACHE = False

# ====== CHANGE SELECTED POINTS CACHE HERE ======
SELECTION_CACHE_PATH = os.path.expanduser("~/.ros/saddle_mold_large.npz")
# SELECTION_CACHE_PATH = os.path.expanduser("~/.ros/saddle_mold_local.npz")
# SELECTION_CACHE_PATH = os.path.expanduser("~/.ros/saddle_mold_contour.npz")

# Rectangle selection behavior.
RECT_MIN_POINTS = 4
LOG_EVERY_N_FRAMES = 30


def ktoc(val):
    return (val - 27315) / 100.0


def safe_ktoc_array(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=np.float64)
    if vals.size == 0:
        return vals
    med = float(np.nanmedian(vals))
    if med > 1.0e4:
        return (vals - 27315.0) / 100.0
    if med > 200.0:
        return vals - 273.15
    return vals


class ContinuousSurfaceSelector(Node):
    def __init__(self):
        super().__init__("continuous_surface_selector")

        self.subscription = self.create_subscription(
            PointCloud2,
            RAW_THERMAL_TOPIC,
            self.pointcloud_callback,
            10,
        )

        self.reselection_service = self.create_service(
            Trigger,
            "/reselect_surface",
            self.reselection_service_callback,
        )

        self.pub_selected_surface = self.create_publisher(PointCloud2, SELECTED_SURFACE_TOPIC, 10)
        self.boundary_pub = self.create_publisher(PoseArray, BOUNDARY_TOPIC, 10)
        self.metrics_pub = self.create_publisher(Float64MultiArray, METRICS_TOPIC, 10)
        self.roi_status_pub = self.create_publisher(String, ROI_STATUS_TOPIC, 10)
        self.replan_trigger_pub = self.create_publisher(Bool, REPLAN_TRIGGER_TOPIC, 10)

        self.percent_sub = self.create_subscription(
            Float64,
            PERCENT_IN_BAND_TOPIC,
            self.percent_in_band_callback,
            10,
        )

        self.current_points = None
        self.current_temperatures = None
        self.current_frame_id = None
        self.current_normals = None
        self.current_pcd = None

        self.boundary_points = []
        self.selection_defined = False
        self.reselection_requested = False
        self.boundary_mask = None
        self.boundary_msg = None
        self.boundary_frame_id = "base_link"

        # Original single-rectangle fields kept for backward compatibility with normal mode.
        self.rect_center = None
        self.rect_proj_matrix = None
        self.rect_axes_2d = None
        self.rect_points_uv = None
        self.rect_corners_snapped_3d = None
        self.rect_uv_min = None
        self.rect_uv_max = None
        self.rect_pts2_center = None
        self.rect_margin = 0.0

        # Multi-ROI state.
        self.roi_rects: list[dict[str, Any]] = []
        self.roi_masks: list[np.ndarray] = []
        self.roi_boundary_msgs: list[PoseArray] = []
        self.roi_complete: list[bool] = []
        self.active_roi_index = 0
        self.latest_percent_in_band = float("nan")
        self.last_switch_time = -1e9
        self.ignore_percent_until_sec = -1e9

        self.cache_load_attempted = False
        self.cache_loaded = False

        self.vis_thread = None
        self.vis_running = False
        self.log_counter = 0

        self.get_logger().info("Continuous Surface Selector running")
        self.get_logger().info(f"multi_roi_mode={ENABLE_MULTI_ROI_MODE}")
        self.get_logger().info("Use: ros2 service call /reselect_surface std_srvs/srv/Trigger")
        self.get_logger().info("Waiting for pointcloud data...")

    # --------------------------------------------------------
    # ROS callbacks
    # --------------------------------------------------------

    def reselection_service_callback(self, request, response):
        self.get_logger().info("Reselection requested via service call")
        self.reselection_requested = True
        response.success = True
        response.message = "Reselection triggered. Please define new boundary in the upcoming window."
        return response

    def percent_in_band_callback(self, msg: Float64):
        if not ENABLE_MULTI_ROI_MODE:
            return
        if not self.selection_defined or len(self.roi_rects) == 0:
            return

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        if now_sec < self.ignore_percent_until_sec:
            return
        if now_sec - self.last_switch_time < MULTI_ROI_SWITCH_COOLDOWN_S:
            return

        self.latest_percent_in_band = float(msg.data)

        if self.active_roi_index < 0 or self.active_roi_index >= len(self.roi_rects):
            return

        if self.latest_percent_in_band >= MULTI_ROI_TARGET_PERCENT:
            self.roi_complete[self.active_roi_index] = True
            self.get_logger().info(
                f"ROI {self.active_roi_index + 1}/{len(self.roi_rects)} reached "
                f"{self.latest_percent_in_band:.1f}% in band."
            )
            self.switch_to_next_roi()
            return

        if MULTI_ROI_ALLOW_REVISIT:
            for i, complete in enumerate(self.roi_complete):
                if i == self.active_roi_index:
                    continue
                if complete and self.latest_percent_in_band < MULTI_ROI_REVISIT_BELOW_PERCENT:
                    # Conservative placeholder. A true revisit policy needs per-ROI percent tracking.
                    pass

    def pointcloud_callback(self, msg: PointCloud2):
        points = []
        temperatures = []

        for p in point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z", "thermal"),
            skip_nans=True,
        ):
            x, y, z, temp = p
            points.append([x, y, z])
            temperatures.append(temp)

        if not points:
            self.get_logger().warn("Empty thermal point cloud")
            return

        points = np.asarray(points, dtype=np.float64)
        temperatures = np.asarray(temperatures, dtype=np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        colors = self.temperature_to_colors(temperatures)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
        normals = np.asarray(pcd.normals)
        normals[normals[:, 2] < 0] *= -1
        pcd.normals = o3d.utility.Vector3dVector(normals)

        self.current_points = points
        self.current_temperatures = temperatures
        self.current_frame_id = msg.header.frame_id
        self.current_normals = normals
        self.current_pcd = pcd

        if self.reselection_requested:
            self.reset_selection_state(clear_cache_loaded=True)
            self.reselection_requested = False
            self.get_logger().info("Reselection triggered - please define new boundary")

        if not self.selection_defined and not self.cache_load_attempted and USE_SELECTION_CACHE:
            self.cache_load_attempted = True
            if self.load_selection_cache():
                self.selection_defined = True
                self.cache_loaded = True
                self.get_logger().info(f"Loaded selection cache from {SELECTION_CACHE_PATH}")
                self.show_initial_selection()

        if not self.selection_defined:
            self.publish_selected_surface_with_temps(points, temperatures, self.current_frame_id)

        if self.selection_defined:
            self.continuous_surface_selection()
        elif not self.vis_running:
            self.start_interactive_selection()

    # --------------------------------------------------------
    # Selection and cache state
    # --------------------------------------------------------

    def reset_selection_state(self, clear_cache_loaded: bool = False):
        self.selection_defined = False
        self.boundary_points = []
        self.boundary_mask = None
        self.boundary_msg = None

        self.rect_center = None
        self.rect_proj_matrix = None
        self.rect_axes_2d = None
        self.rect_points_uv = None
        self.rect_corners_snapped_3d = None
        self.rect_uv_min = None
        self.rect_uv_max = None
        self.rect_pts2_center = None
        self.rect_margin = 0.0

        self.roi_rects = []
        self.roi_masks = []
        self.roi_boundary_msgs = []
        self.roi_complete = []
        self.active_roi_index = 0
        self.latest_percent_in_band = float("nan")
        self.last_switch_time = -1e9
        self.ignore_percent_until_sec = -1e9

        if clear_cache_loaded:
            self.cache_loaded = False
            self.cache_load_attempted = True

    def _points_to_pose_array(self, points_xyz, frame_id):
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = frame_id
        for x, y, z in np.asarray(points_xyz, dtype=np.float64):
            p = Pose()
            p.position.x = float(x)
            p.position.y = float(y)
            p.position.z = float(z)
            p.orientation.w = 1.0
            pa.poses.append(p)
        return pa

    def _copy_rect_to_single_state(self, rect: dict[str, Any]):
        self.rect_center = rect["center"]
        self.rect_proj_matrix = rect["proj_matrix"]
        self.rect_axes_2d = rect["axes_2d"]
        self.rect_points_uv = rect.get("points_uv")
        self.rect_corners_snapped_3d = np.asarray(rect["corners_xyz"], dtype=np.float64).tolist()
        self.rect_uv_min = rect["uv_min"]
        self.rect_uv_max = rect["uv_max"]
        self.rect_pts2_center = rect["pts2_center"]
        self.rect_margin = float(rect.get("margin", 0.0))
        self.boundary_mask = rect.get("mask")
        self.boundary_points = list(self.rect_corners_snapped_3d)
        self.boundary_msg = self._points_to_pose_array(self.rect_corners_snapped_3d, self.current_frame_id)

    def order_rectangle_corners_2d(self, corners_2d: np.ndarray) -> np.ndarray:
        c = corners_2d.mean(axis=0)
        angles = np.arctan2(corners_2d[:, 1] - c[1], corners_2d[:, 0] - c[0])
        ordered = corners_2d[np.argsort(angles)]

        # Rotate so the first point is consistently one of the lower-left style corners.
        start = int(np.argmin(ordered[:, 0] + ordered[:, 1]))
        ordered = np.roll(ordered, -start, axis=0)
        return ordered

    def build_rectangle_from_four_corners(self, picked_indices) -> dict[str, Any] | None:
        if self.current_points is None or len(picked_indices) != 4:
            return None

        corners = self.current_points[np.asarray(picked_indices, dtype=np.int32)].astype(np.float64)
        return self.build_rectangle_from_corners_xyz(corners)

    def build_rectangle_from_corners_xyz(self, corners_xyz: np.ndarray) -> dict[str, Any] | None:
        if self.current_points is None:
            return None

        corners = np.asarray(corners_xyz, dtype=np.float64)
        if corners.shape != (4, 3):
            return None

        center = corners.mean(axis=0)
        centered = corners - center
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        proj_matrix = vt[:2, :]

        points_2d = (self.current_points - center) @ proj_matrix.T
        corners_2d = (corners - center) @ proj_matrix.T

        # pts2_center = points_2d.mean(axis=0)
        # pts2_centered = points_2d - pts2_center
        # cov = pts2_centered.T @ pts2_centered
        # evals, evecs = np.linalg.eigh(cov)
        # order = np.argsort(evals)[::-1]
        # axes_2d = evecs[:, order]

        # points_uv = pts2_centered @ axes_2d
        # corners_uv = (corners_2d - pts2_center) @ axes_2d
        ordered_corners_2d = self.order_rectangle_corners_2d(corners_2d)

        edge01 = ordered_corners_2d[1] - ordered_corners_2d[0]
        edge03 = ordered_corners_2d[3] - ordered_corners_2d[0]

        if np.linalg.norm(edge03) > np.linalg.norm(edge01):
            axis_u_2d = edge03 / max(np.linalg.norm(edge03), 1e-12)
            axis_v_2d = edge01 / max(np.linalg.norm(edge01), 1e-12)
        else:
            axis_u_2d = edge01 / max(np.linalg.norm(edge01), 1e-12)
            axis_v_2d = edge03 / max(np.linalg.norm(edge03), 1e-12)

        # Re-orthogonalize so the rectangle test is stable.
        axis_v_2d = axis_v_2d - np.dot(axis_v_2d, axis_u_2d) * axis_u_2d
        axis_v_2d = axis_v_2d / max(np.linalg.norm(axis_v_2d), 1e-12)

        axes_2d = np.column_stack([axis_u_2d, axis_v_2d])

        pts2_center = ordered_corners_2d.mean(axis=0)
        points_uv = (points_2d - pts2_center) @ axes_2d
        corners_uv = (corners_2d - pts2_center) @ axes_2d

        tree = cKDTree(points_uv)
        _, nn_idx = tree.query(corners_uv, k=1)
        snapped_xyz = self.current_points[nn_idx]

        uv_min = corners_uv.min(axis=0)
        uv_max = corners_uv.max(axis=0)

        sample_count = min(len(points_uv), 2000)
        if sample_count >= 2:
            sample_idx = np.linspace(0, len(points_uv) - 1, sample_count, dtype=np.int32)
            sample_uv = points_uv[sample_idx]
            sample_tree = cKDTree(sample_uv)
            dists, _ = sample_tree.query(sample_uv, k=2)
            nn_d = dists[:, 1]
            spacing = float(np.median(nn_d[np.isfinite(nn_d)])) if np.any(np.isfinite(nn_d)) else 0.0
        else:
            spacing = 0.0

        margin = max(0.75 * spacing, 0.015)
        # margin = max(1.25 * spacing, 0.5 * spacing)
        
        uv_min_m = uv_min - margin
        uv_max_m = uv_max + margin

        mask = (
            (points_uv[:, 0] >= uv_min_m[0]) &
            (points_uv[:, 0] <= uv_max_m[0]) &
            (points_uv[:, 1] >= uv_min_m[1]) &
            (points_uv[:, 1] <= uv_max_m[1])
        )

        return {
            "center": center.astype(np.float64),
            "proj_matrix": proj_matrix.astype(np.float64),
            "axes_2d": axes_2d.astype(np.float64),
            "points_uv": points_uv.astype(np.float64),
            "corners_xyz": snapped_xyz.astype(np.float64),
            "uv_min": uv_min_m.astype(np.float64),
            "uv_max": uv_max_m.astype(np.float64),
            "pts2_center": pts2_center.astype(np.float64),
            "margin": float(margin),
            "mask": mask.astype(bool),
        }

    def build_rectangle_selection_from_four_corners(self, picked_indices):
        rect = self.build_rectangle_from_four_corners(picked_indices)
        if rect is None:
            return False
        self._copy_rect_to_single_state(rect)
        return True

    def update_rect_mask(self, rect: dict[str, Any]) -> np.ndarray | None:
        if self.current_points is None:
            return None

        required = ["center", "proj_matrix", "axes_2d", "pts2_center", "uv_min", "uv_max"]
        if any(rect.get(k) is None for k in required):
            return None

        points_2d = (self.current_points - rect["center"]) @ rect["proj_matrix"].T
        points_uv = (points_2d - rect["pts2_center"]) @ rect["axes_2d"]
        uv_min = rect["uv_min"]
        uv_max = rect["uv_max"]

        mask = (
            (points_uv[:, 0] >= uv_min[0]) &
            (points_uv[:, 0] <= uv_max[0]) &
            (points_uv[:, 1] >= uv_min[1]) &
            (points_uv[:, 1] <= uv_max[1])
        )
        rect["points_uv"] = points_uv
        rect["mask"] = mask
        return mask

    def update_cached_rectangle_mask(self):
        if ENABLE_MULTI_ROI_MODE:
            return self.get_active_roi_mask()

        if self.rect_center is None:
            return None

        rect = {
            "center": self.rect_center,
            "proj_matrix": self.rect_proj_matrix,
            "axes_2d": self.rect_axes_2d,
            "pts2_center": self.rect_pts2_center,
            "uv_min": self.rect_uv_min,
            "uv_max": self.rect_uv_max,
        }
        mask = self.update_rect_mask(rect)
        self.rect_points_uv = rect.get("points_uv")
        self.boundary_mask = mask
        return mask

    # --------------------------------------------------------
    # Interactive selection
    # --------------------------------------------------------

    def start_interactive_selection(self):
        self.vis_running = True
        self.vis_thread = threading.Thread(target=self.run_interactive_visualization)
        self.vis_thread.daemon = True
        self.vis_thread.start()

    def run_interactive_visualization(self):
        try:
            if ENABLE_MULTI_ROI_MODE:
                ok = self.run_multi_roi_selection()
            else:
                ok = self.run_single_roi_selection()

            if ok:
                self.selection_defined = True
                if SAVE_SELECTION_CACHE:
                    self.save_selection_cache()
                self.show_initial_selection()
        except Exception as exc:
            self.get_logger().error(f"Visualization error: {exc}")
        finally:
            self.vis_running = False

    def pick_rectangle_window(self, title: str) -> list[int]:
        self.get_logger().info(title)
        self.get_logger().info("Hold Shift + Left Click to pick exactly 4 rectangle corners, then close the window.")

        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window(window_name=title, width=1200, height=800)

        if self.current_pcd is not None:
            vis.add_geometry(self.current_pcd)

        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        vis.add_geometry(coord_frame)
        vis.run()
        picked_indices = list(vis.get_picked_points())
        vis.destroy_window()
        return picked_indices

    def run_single_roi_selection(self) -> bool:
        picked_indices = self.pick_rectangle_window("Define Main ROI")
        if len(picked_indices) != 4:
            self.get_logger().warn("Please pick exactly 4 corner points for rectangle mode.")
            return False

        ok = self.build_rectangle_selection_from_four_corners(picked_indices)
        if not ok:
            self.get_logger().warn("Failed to build rectangle selection.")
            return False

        self.get_logger().info("Rectangle boundary defined with 4 corners.")
        for i, point in enumerate(self.rect_corners_snapped_3d):
            self.get_logger().info(f"  Corner {i + 1}: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")
        return True

    def run_multi_roi_selection(self) -> bool:
        self.roi_rects = []
        self.roi_masks = []
        self.roi_boundary_msgs = []
        self.roi_complete = []
        self.active_roi_index = 0

        count = max(1, int(MULTI_ROI_COUNT))
        for roi_idx in range(count):
            title = f"Define ROI {roi_idx + 1} of {count}"
            picked_indices = self.pick_rectangle_window(title)
            if len(picked_indices) != 4:
                self.get_logger().warn(f"ROI {roi_idx + 1}: please pick exactly 4 corner points.")
                return False

            rect = self.build_rectangle_from_four_corners(picked_indices)
            if rect is None or rect.get("mask") is None or not np.any(rect["mask"]):
                self.get_logger().warn(f"ROI {roi_idx + 1}: failed to build non-empty rectangle selection.")
                return False

            self.roi_rects.append(rect)
            self.roi_masks.append(rect["mask"])
            self.roi_boundary_msgs.append(self._points_to_pose_array(rect["corners_xyz"], self.current_frame_id))
            self.roi_complete.append(False)

            self.get_logger().info(
                f"ROI {roi_idx + 1} defined with {int(np.sum(rect['mask']))} points."
            )

        self.boundary_mask = self.roi_masks[0]
        self.boundary_msg = self.roi_boundary_msgs[0]
        self._copy_rect_to_single_state(self.roi_rects[0])
        self.ignore_percent_until_sec = self.get_clock().now().nanoseconds * 1e-9 + MULTI_ROI_SWITCH_COOLDOWN_S
        return True

    # --------------------------------------------------------
    # Multi-ROI scheduler
    # --------------------------------------------------------

    def get_active_roi_mask(self) -> np.ndarray | None:
        if not self.roi_rects:
            return None

        self.active_roi_index = int(np.clip(self.active_roi_index, 0, len(self.roi_rects) - 1))
        mask = self.update_rect_mask(self.roi_rects[self.active_roi_index])
        if mask is not None:
            while len(self.roi_masks) < len(self.roi_rects):
                self.roi_masks.append(mask.copy())
            self.roi_masks[self.active_roi_index] = mask
            self.boundary_mask = mask
            self.boundary_msg = self.roi_boundary_msgs[self.active_roi_index]
        return mask

    def switch_to_next_roi(self):
        if not ENABLE_MULTI_ROI_MODE or not self.roi_rects:
            return

        next_idx = None
        for i, complete in enumerate(self.roi_complete):
            if not complete:
                next_idx = i
                break

        if next_idx is None:
            self.get_logger().info("All selected ROIs have reached the target percentage.")
            self.publish_roi_status()
            return

        if next_idx == self.active_roi_index:
            return

        prev_idx = self.active_roi_index
        self.active_roi_index = next_idx
        self.boundary_msg = self.roi_boundary_msgs[self.active_roi_index]
        self.boundary_mask = self.get_active_roi_mask()

        now_sec = self.get_clock().now().nanoseconds * 1e-9
        self.last_switch_time = now_sec
        self.ignore_percent_until_sec = now_sec + MULTI_ROI_SWITCH_COOLDOWN_S
        self.latest_percent_in_band = float("nan")

        self.get_logger().info(
            f"Switching active ROI from {prev_idx + 1} to {self.active_roi_index + 1}."
        )

        if MULTI_ROI_REPLAN_ON_SWITCH:
            trigger = Bool()
            trigger.data = True
            self.replan_trigger_pub.publish(trigger)
            self.get_logger().info(f"Published replan trigger on {REPLAN_TRIGGER_TOPIC}")

        self.publish_roi_status()

    def publish_roi_status(self):
        if not ENABLE_MULTI_ROI_MODE:
            return

        status = {
            "enabled": True,
            "active_roi_index": int(self.active_roi_index),
            "active_roi_number": int(self.active_roi_index + 1),
            "roi_count": int(len(self.roi_rects)),
            "roi_complete": [bool(x) for x in self.roi_complete],
            "latest_percent_in_band": None if not np.isfinite(self.latest_percent_in_band) else float(self.latest_percent_in_band),
            "target_percent": float(MULTI_ROI_TARGET_PERCENT),
        }
        msg = String()
        msg.data = json.dumps(status)
        self.roi_status_pub.publish(msg)

    # --------------------------------------------------------
    # Continuous publication
    # --------------------------------------------------------

    def continuous_surface_selection(self):
        if self.current_points is None or not self.selection_defined:
            return

        if ENABLE_MULTI_ROI_MODE:
            surface_mask = self.get_active_roi_mask()
        else:
            surface_mask = self.update_cached_rectangle_mask()

        if self.boundary_msg is not None:
            self.boundary_pub.publish(self.boundary_msg)

        if surface_mask is None or not np.any(surface_mask):
            self.get_logger().warn("No surface points selected")
            return

        selected_points = self.current_points[surface_mask]
        selected_temps = self.current_temperatures[surface_mask]
        if len(selected_temps) == 0:
            return

        temps_C = safe_ktoc_array(selected_temps)
        max_temp = float(np.max(temps_C))
        median_temp = float(np.median(temps_C))

        metrics_msg = Float64MultiArray()
        metrics_msg.data = [max_temp, median_temp]
        self.metrics_pub.publish(metrics_msg)

        self.publish_selected_surface_with_temps(selected_points, selected_temps, self.current_frame_id)
        self.publish_roi_status()

        self.log_counter += 1
        if self.log_counter % LOG_EVERY_N_FRAMES == 0:
            label = "surface"
            if ENABLE_MULTI_ROI_MODE:
                label = f"ROI {self.active_roi_index + 1}/{len(self.roi_rects)}"
            self.get_logger().info(
                f"Monitoring {label}: {len(selected_points)} points, "
                f"temp {float(np.min(temps_C)):.1f}-{float(np.max(temps_C)):.1f}°C "
                f"(avg {float(np.mean(temps_C)):.1f}°C)"
            )

    # --------------------------------------------------------
    # Cache
    # --------------------------------------------------------

    def serialize_rect(self, rect: dict[str, Any], prefix: str, payload: dict[str, Any]):
        payload[f"{prefix}_center"] = np.asarray(rect["center"], dtype=np.float64)
        payload[f"{prefix}_proj_matrix"] = np.asarray(rect["proj_matrix"], dtype=np.float64)
        payload[f"{prefix}_axes_2d"] = np.asarray(rect["axes_2d"], dtype=np.float64)
        payload[f"{prefix}_corners_xyz"] = np.asarray(rect["corners_xyz"], dtype=np.float64)
        payload[f"{prefix}_uv_min"] = np.asarray(rect["uv_min"], dtype=np.float64)
        payload[f"{prefix}_uv_max"] = np.asarray(rect["uv_max"], dtype=np.float64)
        payload[f"{prefix}_pts2_center"] = np.asarray(rect["pts2_center"], dtype=np.float64)
        payload[f"{prefix}_margin"] = np.asarray([float(rect.get("margin", 0.0))], dtype=np.float64)

    def deserialize_rect(self, data, prefix: str) -> dict[str, Any] | None:
        keys = [
            f"{prefix}_center",
            f"{prefix}_proj_matrix",
            f"{prefix}_axes_2d",
            f"{prefix}_corners_xyz",
            f"{prefix}_uv_min",
            f"{prefix}_uv_max",
            f"{prefix}_pts2_center",
        ]
        if any(k not in data.files for k in keys):
            return None

        rect = {
            "center": data[f"{prefix}_center"].astype(np.float64),
            "proj_matrix": data[f"{prefix}_proj_matrix"].astype(np.float64),
            "axes_2d": data[f"{prefix}_axes_2d"].astype(np.float64),
            "corners_xyz": data[f"{prefix}_corners_xyz"].astype(np.float64),
            "uv_min": data[f"{prefix}_uv_min"].astype(np.float64),
            "uv_max": data[f"{prefix}_uv_max"].astype(np.float64),
            "pts2_center": data[f"{prefix}_pts2_center"].astype(np.float64),
            "margin": float(data[f"{prefix}_margin"][0]) if f"{prefix}_margin" in data.files else 0.0,
            "mask": None,
            "points_uv": None,
        }
        self.update_rect_mask(rect)
        return rect

    def save_selection_cache(self):
        try:
            os.makedirs(os.path.dirname(SELECTION_CACHE_PATH), exist_ok=True)
            payload: dict[str, Any] = {
                "cache_version": np.asarray([2], dtype=np.int32),
                "multi_roi_mode": np.asarray([1 if ENABLE_MULTI_ROI_MODE else 0], dtype=np.int32),
                "frame_id": np.asarray([self.current_frame_id or ""], dtype=object),
            }

            if ENABLE_MULTI_ROI_MODE:
                payload["roi_count"] = np.asarray([len(self.roi_rects)], dtype=np.int32)
                for i, rect in enumerate(self.roi_rects):
                    self.serialize_rect(rect, f"roi_{i}", payload)
            else:
                rect = {
                    "center": self.rect_center,
                    "proj_matrix": self.rect_proj_matrix,
                    "axes_2d": self.rect_axes_2d,
                    "corners_xyz": np.asarray(self.rect_corners_snapped_3d, dtype=np.float64),
                    "uv_min": self.rect_uv_min,
                    "uv_max": self.rect_uv_max,
                    "pts2_center": self.rect_pts2_center,
                    "margin": self.rect_margin,
                }
                self.serialize_rect(rect, "main", payload)

            np.savez(SELECTION_CACHE_PATH, **payload)
            self.get_logger().info(f"Saved selection cache to {SELECTION_CACHE_PATH}")
        except Exception as exc:
            self.get_logger().warn(f"Failed to save selection cache: {exc}")

    def load_selection_cache(self) -> bool:
        if self.current_points is None:
            return False
        if not os.path.exists(SELECTION_CACHE_PATH):
            return False

        try:
            data = np.load(SELECTION_CACHE_PATH, allow_pickle=True)
            cache_multi = bool(int(data["multi_roi_mode"][0])) if "multi_roi_mode" in data.files else False
            if cache_multi != bool(ENABLE_MULTI_ROI_MODE):
                self.get_logger().info("Selection cache mode does not match current mode; ignoring cache.")
                return False

            if ENABLE_MULTI_ROI_MODE:
                roi_count = int(data["roi_count"][0]) if "roi_count" in data.files else 0
                if roi_count <= 0:
                    return False
                self.roi_rects = []
                self.roi_masks = []
                self.roi_boundary_msgs = []
                self.roi_complete = []

                for i in range(roi_count):
                    rect = self.deserialize_rect(data, f"roi_{i}")
                    if rect is None:
                        return False
                    mask = rect.get("mask")
                    if mask is None or not np.any(mask):
                        return False
                    self.roi_rects.append(rect)
                    self.roi_masks.append(mask)
                    self.roi_boundary_msgs.append(self._points_to_pose_array(rect["corners_xyz"], self.current_frame_id))
                    self.roi_complete.append(False)

                self.active_roi_index = 0
                self.boundary_mask = self.roi_masks[0]
                self.boundary_msg = self.roi_boundary_msgs[0]
                self._copy_rect_to_single_state(self.roi_rects[0])
                self.ignore_percent_until_sec = self.get_clock().now().nanoseconds * 1e-9 + MULTI_ROI_SWITCH_COOLDOWN_S
                return True

            rect = self.deserialize_rect(data, "main")
            if rect is None:
                return False
            mask = rect.get("mask")
            if mask is None or not np.any(mask):
                return False
            self._copy_rect_to_single_state(rect)
            return True
        except Exception as exc:
            self.get_logger().warn(f"Failed to load selection cache: {exc}")
            return False

    # --------------------------------------------------------
    # Visualization and publishing helpers
    # --------------------------------------------------------

    def temperature_to_colors(self, temperatures):
        temp_min = np.nanmin(temperatures)
        temp_max = np.nanmax(temperatures)
        temp_range = temp_max - temp_min

        if temp_range == 0:
            return np.ones((len(temperatures), 3)) * 0.5

        normalized = (temperatures - temp_min) / temp_range
        colors = np.zeros((len(temperatures), 3))
        colors[:, 2] = 1.0 - normalized
        colors[:, 0] = normalized
        colors[:, 1] = 1.0 - np.abs(normalized - 0.5) * 2
        return colors

    def publish_selected_surface_with_temps(self, points, temps, frame_id):
        if len(points) == 0:
            return

        try:
            header = Header()
            header.frame_id = frame_id
            header.stamp = self.get_clock().now().to_msg()

            fields = [
                PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name="thermal", offset=12, datatype=PointField.FLOAT32, count=1),
            ]

            cloud_data = [struct.pack("ffff", float(p[0]), float(p[1]), float(p[2]), float(t)) for p, t in zip(points, temps)]

            surface_cloud = PointCloud2()
            surface_cloud.header = header
            surface_cloud.height = 1
            surface_cloud.width = len(points)
            surface_cloud.fields = fields
            surface_cloud.is_bigendian = False
            surface_cloud.point_step = 16
            surface_cloud.row_step = surface_cloud.point_step * len(points)
            surface_cloud.data = b"".join(cloud_data)
            surface_cloud.is_dense = True

            self.pub_selected_surface.publish(surface_cloud)
        except Exception as exc:
            self.get_logger().error(f"Publishing error: {exc}")

    def show_initial_selection(self):
        if self.current_pcd is None:
            return

        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Initial Selection Result", width=1200, height=800)

            result_pcd = o3d.geometry.PointCloud()
            result_pcd.points = self.current_pcd.points
            colors = self.temperature_to_colors(self.current_temperatures)

            if ENABLE_MULTI_ROI_MODE and self.roi_rects:
                for i, rect in enumerate(self.roi_rects):
                    mask = rect.get("mask")
                    if mask is None:
                        mask = self.update_rect_mask(rect)
                    if mask is not None:
                        if i == self.active_roi_index:
                            colors[mask] = [0, 1, 0]
                        else:
                            colors[mask] = [0.1, 0.3, 1.0]
                    self.add_rect_markers(vis, rect["corners_xyz"], active=(i == self.active_roi_index))
            else:
                mask = self.boundary_mask
                if mask is not None:
                    colors[mask] = [0, 1, 0]
                corners = self.rect_corners_snapped_3d if self.rect_corners_snapped_3d is not None else self.boundary_points
                self.add_rect_markers(vis, corners, active=True)

            result_pcd.colors = o3d.utility.Vector3dVector(colors)
            vis.add_geometry(result_pcd)
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            vis.add_geometry(coord_frame)

            if ENABLE_MULTI_ROI_MODE:
                self.get_logger().info(
                    f"Initial selection: {len(self.roi_rects)} ROIs, active ROI {self.active_roi_index + 1}."
                )
            elif self.boundary_mask is not None:
                self.get_logger().info(f"Initial selection: {int(np.sum(self.boundary_mask))} points (Green)")
            self.get_logger().info("Close window to continue monitoring...")

            vis.run()
            vis.destroy_window()
        except Exception as exc:
            self.get_logger().error(f"Initial selection visualization error: {exc}")

    def add_rect_markers(self, vis, corners_xyz, active: bool):
        corners = np.asarray(corners_xyz, dtype=np.float64)
        if corners.shape[0] == 0:
            return

        sphere_color = [1, 0, 0] if active else [0.2, 0.2, 1.0]
        line_color = [1, 1, 0] if active else [0.2, 0.8, 1.0]

        for point in corners:
            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
            sphere.paint_uniform_color(sphere_color)
            sphere.translate(point)
            vis.add_geometry(sphere)

        if corners.shape[0] == 4:
            points = np.vstack([corners, corners[0]])
            lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
            boundary_lines = o3d.geometry.LineSet()
            boundary_lines.points = o3d.utility.Vector3dVector(points)
            boundary_lines.lines = o3d.utility.Vector2iVector(lines)
            boundary_lines.colors = o3d.utility.Vector3dVector([line_color] * len(lines))
            vis.add_geometry(boundary_lines)


def main(args=None):
    rclpy.init(args=args)
    node = ContinuousSurfaceSelector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

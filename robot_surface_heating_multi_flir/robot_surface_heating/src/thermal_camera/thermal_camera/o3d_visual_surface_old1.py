#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, PointField
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header, Float64MultiArray
from std_srvs.srv import Trigger

from matplotlib.path import Path
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import threading
import time
import struct

from geometry_msgs.msg import Pose, PoseArray


ENABLE_LOCAL_HEATING_MODE = False
USE_SELECTION_CACHE = True
SAVE_SELECTION_CACHE = True
LOCAL_BUFFER_NODES = 3
SELECTION_CACHE_PATH = "/tmp/surface_selection_cache.npz"


def ktoc(val):
    return (val - 27315) / 100.0

class ContinuousSurfaceSelector(Node):
    def __init__(self):
        super().__init__('continuous_surface_selector')

        # Subscribe to thermal pointcloud
        self.subscription = self.create_subscription(
            PointCloud2,
            '/raw_thermal_pointcloud',
            self.pointcloud_callback,
            10
        )

        # Service for reselection trigger
        self.reselection_service = self.create_service(
            Trigger,
            '/reselect_surface',
            self.reselection_service_callback
        )

        # Publish selected surface
        self.pub_selected_surface = self.create_publisher(PointCloud2, '/selected_surface_points', 10)

        # Store current pointcloud data
        self.current_points = None
        self.current_temperatures = None
        self.current_frame_id = None
        self.current_normals = None
        self.current_pcd = None
        
        # Selection state - persistent across callbacks
        self.boundary_points = []
        self.selection_defined = False
        self.reselection_requested = False
        self.boundary_mask = None

        # Rectangle-mode cached geometry
        self.rect_center = None
        self.rect_proj_matrix = None          # (2,3)
        self.rect_axes_2d = None              # (2,2) local in-plane axes
        self.rect_points_uv = None            # (N,2) cached point coords in rectangle basis
        self.rect_corners_snapped_3d = None   # snapped 3D corners for visualization
        self.rect_uv_min = None               # [u_min, v_min]
        self.rect_uv_max = None               # [u_max, v_max]

        self.rect_pts2_center = None
        self.rect_margin = 0.0
        self.boundary_msg = None

        # Surface selection parameters
        self.use_region_growing = False  # Disabled - use pure boundary selection
        # No need for normal_threshold or height_tolerance since we're not using region growing

        self.hotspot_mask = None
        self.buffer_mask = None
        self.protected_mask = None

        self.hotspot_points = []
        self.hotspot_msg = None

        self.cache_loaded = False
        
        # Threading for visualization
        self.vis_thread = None
        self.vis_running = False

        self.boundary_frame_id = 'base_link'

        self.boundary_msg = None

        self.boundary_pub = self.create_publisher(
            PoseArray, 
            'selected_boundary_points', 
            10,
            # qos
            )        
        

        self.metrics_pub = self.create_publisher(
            Float64MultiArray,
            '/thermal_roi_metrics',
            10
        )
        
        self.get_logger().info('Continuous Surface Selector running - Flexible Polygon Mode')
        self.get_logger().info('Select any number of boundary points (minimum 3)')
        self.get_logger().info('Use: ros2 service call /reselect_surface std_srvs/srv/Trigger')
        self.get_logger().info('Waiting for pointcloud data...')

    def _points_to_pose_array(self, points_xyz, frame_id):
        pa = PoseArray()
        pa.header.stamp = self.get_clock().now().to_msg()
        pa.header.frame_id = frame_id
        for x, y, z in points_xyz:
            p = Pose()
            p.position.x = float(x); p.position.y = float(y); p.position.z = float(z)
            # identity orientation; consumer can override if needed
            p.orientation.w = 1.0
            pa.poses.append(p)
        return pa

    def reselection_service_callback(self, request, response):
        """Service callback to trigger reselection of boundary points"""
        self.get_logger().info("Reselection requested via service call")
        self.reselection_requested = True
        
        response.success = True
        response.message = "Reselection triggered. Please define new boundary in the upcoming window."
        return response

    def pointcloud_callback(self, msg: PointCloud2):
        # Extract points and temperatures
        points = []
        temperatures = []

        for p in point_cloud2.read_points(msg, field_names=("x", "y", "z", "thermal"), skip_nans=True):
            x, y, z, temp = p
            points.append([x, y, z])
            # temperatures.append(ktoc(temp))
            temperatures.append(temp)

        if not points:
            self.get_logger().warn("Empty thermal point cloud")
            return

        points = np.array(points)
        temperatures = np.array(temperatures)

        # Create Open3D pointcloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Color by temperature for visualization
        colors = self.temperature_to_colors(temperatures)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
        normals = np.asarray(pcd.normals)
        
        # Force normals to point upward
        normals[normals[:, 2] < 0] *= -1
        pcd.normals = o3d.utility.Vector3dVector(normals)

        # Store current data
        self.current_points = points
        self.current_temperatures = temperatures
        self.current_frame_id = msg.header.frame_id
        self.current_normals = normals
        self.current_pcd = pcd

        # Directly publish the entire pointcloud with temperatures
        # self.publish_selected_surface_with_temps(points, temperatures, self.current_frame_id)
        if not self.selection_defined:
            # Only publish full cloud BEFORE selection
            self.publish_selected_surface_with_temps(
                points, temperatures, self.current_frame_id
            )


        # Check if reselection is requested
        if self.reselection_requested:
            self.reselection_requested = False
            self.selection_defined = False
            self.boundary_points = []

            self.rect_center = None
            self.rect_proj_matrix = None
            self.rect_axes_2d = None
            self.rect_points_uv = None
            self.rect_corners_snapped_3d = None
            self.rect_uv_min = None
            self.rect_uv_max = None
            self.get_logger().info("Reselection triggered - please define new boundary")

        # If boundary is defined, continuously select and publish surface
        if self.selection_defined and len(self.boundary_points) >= 4:
            self.continuous_surface_selection()
        
            # self.get_logger().info("before conditional")
        # Start interactive visualization if not already running and no selection defined
        elif not self.vis_running and not self.selection_defined:
            self.get_logger().info("Here")
            self.start_interactive_selection()

    def temperature_to_colors(self, temperatures):
        """Convert temperatures to RGB colors (blue=cold, red=hot)"""
        temp_min = np.min(temperatures)
        temp_max = np.max(temperatures)
        temp_range = temp_max - temp_min
        
        if temp_range == 0:
            return np.ones((len(temperatures), 3)) * 0.5
        
        normalized = (temperatures - temp_min) / temp_range
        
        colors = np.zeros((len(temperatures), 3))
        colors[:, 2] = 1.0 - normalized  # Blue for cold
        colors[:, 0] = normalized        # Red for hot
        colors[:, 1] = 1.0 - np.abs(normalized - 0.5) * 2  # Green in middle
        
        return colors
    
    def build_rectangle_selection_from_four_corners(self, picked_indices):
        """
        Build a clean rectangle selection once from 4 clicked corner points.
        The rectangle is defined in a local snapped 2D basis fitted to the
        selected surface points, then cached as a simple min/max box test.
        """
        if self.current_points is None or len(picked_indices) != 4:
            return False

        corners = self.current_points[np.asarray(picked_indices, dtype=np.int32)].astype(np.float64)

        # Best-fit plane from the 4 clicked corners
        center = corners.mean(axis=0)
        centered = corners - center
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        proj_matrix = vt[:2, :]  # (2,3)

        # Project all current points and the clicked corners to 2D plane
        points_2d = (self.current_points - center) @ proj_matrix.T
        corners_2d = (corners - center) @ proj_matrix.T

        # Estimate dominant in-plane axes from the whole visible point set, not just 4 corners
        pts2_center = points_2d.mean(axis=0)
        pts2_centered = points_2d - pts2_center
        cov = pts2_centered.T @ pts2_centered
        evals, evecs = np.linalg.eigh(cov)
        order = np.argsort(evals)[::-1]
        axes_2d = evecs[:, order]   # columns are principal directions in the fitted plane

        # Express all points and corners in this rectangle basis
        points_uv = pts2_centered @ axes_2d
        corners_uv = (corners_2d - pts2_center) @ axes_2d

        # Snap each clicked corner to the nearest actual point in uv-space
        tree = cKDTree(points_uv)
        _, nn_idx = tree.query(corners_uv, k=1)
        snapped_uv = points_uv[nn_idx]
        snapped_xyz = self.current_points[nn_idx]

        # Rectangle bounds in snapped uv coordinates
        # uv_min = snapped_uv.min(axis=0)
        # uv_max = snapped_uv.max(axis=0)
        uv_min = corners_uv.min(axis=0)
        uv_max = corners_uv.max(axis=0)

        # Small margin so boundary points are not accidentally excluded
        # Estimate point spacing directly from local nearest-neighbor distances in uv
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

        # margin = 0.35 * spacing
        margin = max(0.75 * spacing, 0.015)

        mask = (
            (points_uv[:, 0] >= uv_min[0] - margin) &
            (points_uv[:, 0] <= uv_max[0] + margin) &
            (points_uv[:, 1] >= uv_min[1] - margin) &
            (points_uv[:, 1] <= uv_max[1] + margin)
        )

        self.rect_center = center
        self.rect_proj_matrix = proj_matrix
        self.rect_axes_2d = axes_2d
        self.rect_points_uv = points_uv
        self.rect_corners_snapped_3d = snapped_xyz.tolist()
        # self.rect_uv_min = uv_min
        # self.rect_uv_max = uv_max
        self.rect_pts2_center = pts2_center
        self.rect_margin = margin
        self.rect_uv_min = uv_min - margin
        self.rect_uv_max = uv_max + margin
        self.boundary_mask = mask

        return True


    def update_cached_rectangle_mask(self):
        """
        Fast path for later frames.
        Recompute only the projected uv coordinates using cached rectangle basis,
        then apply a simple axis-aligned box test.
        """
        if (
            self.rect_pts2_center is None or
            self.current_points is None or
            self.rect_center is None or
            self.rect_proj_matrix is None or
            self.rect_axes_2d is None or
            self.rect_uv_min is None or
            self.rect_uv_max is None
        ):
            return None

        points_2d = (self.current_points - self.rect_center) @ self.rect_proj_matrix.T
        # pts2_center = points_2d.mean(axis=0)
        # points_uv = (points_2d - pts2_center) @ self.rect_axes_2d
        points_uv = (points_2d - self.rect_pts2_center) @ self.rect_axes_2d

        self.rect_points_uv = points_uv

        uv_min = self.rect_uv_min
        uv_max = self.rect_uv_max

        mask = (
            (points_uv[:, 0] >= uv_min[0]) &
            (points_uv[:, 0] <= uv_max[0]) &
            (points_uv[:, 1] >= uv_min[1]) &
            (points_uv[:, 1] <= uv_max[1])
        )

        self.boundary_mask = mask
        return mask

    def start_interactive_selection(self):
        """Start the interactive visualization for one-time boundary definition"""
        self.vis_running = True
        self.vis_thread = threading.Thread(target=self.run_interactive_visualization)
        self.vis_thread.daemon = True
        self.vis_thread.start()

    def run_interactive_visualization(self):
        """Run the interactive visualization for boundary definition"""
        self.get_logger().info("Starting boundary definition...")
        self.get_logger().info("Instructions:")
        self.get_logger().info("1. Hold Shift + Left Click to pick boundary points IN ORDER")
        self.get_logger().info("2. Pick at least 3 points, as many as you want")
        self.get_logger().info("3. Go around your region's perimeter")
        self.get_logger().info("4. Close window when done - surface will be monitored continuously")
        
        try:
            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window(window_name="Define Boundary - Pick Points Around Perimeter", width=1200, height=800)
            
            if self.current_pcd is not None:
                vis.add_geometry(self.current_pcd)
            
            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            vis.add_geometry(coord_frame)
            
            vis.run()
            
            # picked_indices = vis.get_picked_points()
            
            # if len(picked_indices) >= 3:
            #     self.boundary_points = [self.current_points[i] for i in picked_indices]  # Use ALL picked points
            #     self.selection_defined = True
                
            #     self.boundary_msg = self._points_to_pose_array(self.boundary_points, self.current_frame_id)

            #     self.get_logger().info(f"Boundary defined with {len(self.boundary_points)} points! Now monitoring continuously...")
            #     for i, point in enumerate(self.boundary_points):
            #         self.get_logger().info(f"  Point {i+1}: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]")
                
            #     # # Simple boundary selection - get ALL points within the polygon
            #     # self.boundary_mask = np.array([
            #     #     self.point_in_polygon(point, self.boundary_points) 
            #     #     for point in self.current_points
            #     # ])

            #     # --- Vectorized projection and Path setup ---
            #     self.poly_center = np.mean(self.boundary_points, axis=0)
            #     centered = self.boundary_points - self.poly_center

            #     _, _, Vt = np.linalg.svd(centered)
            #     self.proj_matrix = Vt[:2, :]  # shape (2, 3)

            #     # Project boundary points to 2D
            #     boundary_2d = (self.boundary_points - self.poly_center) @ self.proj_matrix.T
            #     self.polygon_path = Path(boundary_2d)

            #     # Optional: initial boundary mask for current points
            #     points_2d = (self.current_points - self.poly_center) @ self.proj_matrix.T
            #     self.boundary_mask = self.polygon_path.contains_points(points_2d)            

            #     # Show initial selection result
            #     self.show_initial_selection()
                
            # else:
            #     self.get_logger().warn("Need at least 3 points for a polygon. Restart node to try again.")
            picked_indices = vis.get_picked_points()

            if len(picked_indices) == 4:
                self.boundary_points = [self.current_points[i] for i in picked_indices]
                self.selection_defined = True

                ok = self.build_rectangle_selection_from_four_corners(picked_indices)
                if not ok:
                    self.selection_defined = False
                    self.get_logger().warn("Failed to build rectangle selection.")
                    vis.destroy_window()
                    self.vis_running = False
                    return

                # Publish snapped corners instead of raw clicked corners
                self.boundary_msg = self._points_to_pose_array(
                    self.rect_corners_snapped_3d,
                    self.current_frame_id
                )

                self.get_logger().info("Rectangle boundary defined with 4 corners.")
                for i, point in enumerate(self.rect_corners_snapped_3d):
                    self.get_logger().info(
                        f"  Corner {i+1}: [{point[0]:.3f}, {point[1]:.3f}, {point[2]:.3f}]"
                    )

                self.show_initial_selection()

            else:
                self.get_logger().warn("Please pick exactly 4 corner points for rectangle mode.")
            
            vis.destroy_window()
            
        except Exception as e:
            self.get_logger().error(f"Visualization error: {e}")
        
        self.vis_running = False

    def point_in_polygon(self, point, corners):
        """Improved point-in-polygon test for N-sided polygons"""
        if len(corners) < 3:
            return False
        
        try:
            corners_arr = np.array(corners)
            
            # Project to best-fit plane using SVD
            center = np.mean(corners_arr, axis=0)
            centered = corners_arr - center
            
            U, S, Vt = np.linalg.svd(centered)
            proj_matrix = Vt[:2, :]
            
            # Project corners and point to 2D
            corners_2d = np.dot(corners_arr - center, proj_matrix.T)
            point_2d = np.dot(point - center, proj_matrix.T)
            
            # Ray casting algorithm for N-sided polygon
            x, y = point_2d
            n = len(corners_2d)
            inside = False
            
            p1x, p1y = corners_2d[0]
            for i in range(1, n + 1):
                p2x, p2y = corners_2d[i % n]
                if y > min(p1y, p2y):
                    if y <= max(p1y, p2y):
                        if x <= max(p1x, p2x):
                            if p1y != p2y:
                                xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
                p1x, p1y = p2x, p2y
            
            return inside
            
        except Exception as e:
            return False

    # def continuous_surface_selection(self):
    #     """Continuously select ALL points within defined polygon boundary"""

    #     self.boundary_pub.publish(self.boundary_msg)
    #     # self.get_logger().info(f"Published boundary with {len(self.boundary_msg.poses)} points to 'selected_boundary_points'")

    #     if len(self.boundary_points) < 3 or self.current_points is None:
    #         return
        
    #     # --- Vectorized mask update ---
    #     points_2d = (self.current_points - self.poly_center) @ self.proj_matrix.T
    #     self.boundary_mask = self.polygon_path.contains_points(points_2d)
    #     surface_mask = self.boundary_mask  # No region growing, no height filter - just pure boundary

    #     # surface_mask = np.array([
    #     #     self.point_in_polygon(point, self.boundary_points)
    #     #     for point in self.current_points
    #     # ])

    #     # # 1. Convert boundary to 2D path (XY only)
    #     # boundary_points_2d = np.array(self.boundary_points)[:, :2]
    #     # boundary_path = Path(boundary_points_2d)

    #     # # 2. Extract current point cloud in XY plane
    #     # points_2d = self.current_points[:, :2]

    #     # # 3. Perform fast vectorized point-in-polygon test
    #     # surface_mask = boundary_path.contains_points(points_2d)
        
    #     if not np.any(surface_mask):
    #         self.get_logger().warn("No surface points selected")
    #         return
        
    #     # Get selected points and temperatures
    #     selected_points = self.current_points[surface_mask]
    #     selected_temps = self.current_temperatures[surface_mask]

    #     # selected_points = self.current_points
    #     # selected_temps = self.current_temperatures
        
    #     # Log selection info
    #     min_temp = np.min(selected_temps)
    #     max_temp = np.max(selected_temps)
    #     avg_temp = np.mean(selected_temps)
        
    #     # Publish every frame
    #     self.publish_selected_surface_with_temps(selected_points, selected_temps, self.current_frame_id)
    #     # self.publish_selected_surface_with_temps(self.current_points, self.current_temperatures, self.current_frame_id)


    #     # Log periodically (every 30 frames ≈ 1Hz if camera is 30fps)
    #     if hasattr(self, 'log_counter'):
    #         self.log_counter += 1
    #     else:
    #         self.log_counter = 0
            
    #     if self.log_counter % 30 == 0:
    #         self.get_logger().info(f"Monitoring surface: {len(selected_points)} points, "
    #                              f"temp {ktoc(min_temp):.1f}-{ktoc(max_temp):.1f}°C (avg {ktoc(avg_temp):.1f}°C)")
    def continuous_surface_selection(self):
        """Continuously publish points inside cached rectangle selection."""
        # self.boundary_pub.publish(self.boundary_msg)
        if self.boundary_msg is not None:
            self.boundary_pub.publish(self.boundary_msg)

        if self.current_points is None or not self.selection_defined:
            return

        surface_mask = self.update_cached_rectangle_mask()
        if surface_mask is None or not np.any(surface_mask):
            self.get_logger().warn("No surface points selected")
            return

        selected_points = self.current_points[surface_mask]
        selected_temps = self.current_temperatures[surface_mask]

        temps_C = ktoc(selected_temps)

        if len(selected_temps) == 0:
            return

        # Convert to Celsius
        temps_C = ktoc(selected_temps)

        max_temp = float(np.max(temps_C))
        median_temp = float(np.median(temps_C))

        msg = Float64MultiArray()
        msg.data = [max_temp, median_temp]

        self.metrics_pub.publish(msg)

        min_temp = np.min(selected_temps)
        max_temp = np.max(selected_temps)
        avg_temp = np.mean(selected_temps)

        self.publish_selected_surface_with_temps(
            selected_points,
            selected_temps,
            self.current_frame_id
        )

        if hasattr(self, 'log_counter'):
            self.log_counter += 1
        else:
            self.log_counter = 0

        if self.log_counter % 30 == 0:
            self.get_logger().info(
                f"Monitoring surface: {len(selected_points)} points, "
                f"temp {ktoc(min_temp):.1f}-{ktoc(max_temp):.1f}°C "
                f"(avg {ktoc(avg_temp):.1f}°C)"
            )

    def publish_selected_surface_with_temps(self, points, temps, frame_id):
        """Publish selected surface with thermal data"""
        if len(points) == 0:
            return
        
        try:
            header = Header()
            header.frame_id = frame_id
            header.stamp = self.get_clock().now().to_msg()
            
            # Create pointcloud with thermal field
            fields = [
                PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
                PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
                PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
                PointField(name='thermal', offset=12, datatype=PointField.FLOAT32, count=1),
            ]
            
            cloud_data = [
                struct.pack('ffff', *p, t)
                for p, t in zip(points, temps)
            ]

            surface_cloud = PointCloud2()
            surface_cloud.header = header
            surface_cloud.height = 1
            surface_cloud.width = len(points)
            surface_cloud.fields = fields
            surface_cloud.is_bigendian = False
            surface_cloud.point_step = 16
            surface_cloud.row_step = surface_cloud.point_step * len(points)
            surface_cloud.data = b''.join(cloud_data)
            surface_cloud.is_dense = True
            
            self.pub_selected_surface.publish(surface_cloud)
        except Exception as e:
            self.get_logger().error(f"Publishing error: {e}")
            
    # def show_initial_selection(self):
    #     """Show the initial selection result after boundary definition"""
    #     if self.current_pcd is None or len(self.boundary_points) < 3:
    #         return
        
    #     # Get initial selection
    #     boundary_mask = np.array([
    #         self.point_in_polygon(point, self.boundary_points) 
    #         for point in self.current_points
    #     ])
        
    #     if not np.any(self.boundary_mask):
    #         self.get_logger().warn("No points in boundary for initial display")
    #         return
        
    #     try:
    #         vis = o3d.visualization.Visualizer()
    #         vis.create_window(window_name="Initial Selection Result", width=1200, height=800)
            
    #         # Create colored pointcloud
    #         result_pcd = o3d.geometry.PointCloud()
    #         result_pcd.points = self.current_pcd.points
            
    #         # Color: green for selected, original temperature colors for unselected
    #         colors = self.temperature_to_colors(self.current_temperatures)
    #         colors[boundary_mask] = [0, 1, 0]  # Green for selected points
    #         result_pcd.colors = o3d.utility.Vector3dVector(colors)
            
    #         # Add boundary markers
    #         for i, point in enumerate(self.boundary_points):
    #             sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
    #             sphere.paint_uniform_color([1, 0, 0])  # Red
    #             sphere.translate(point)
    #             vis.add_geometry(sphere)
            
    #         # Add boundary lines (connect all points in sequence, then close the loop)
    #         if len(self.boundary_points) >= 3:
    #             points = self.boundary_points + [self.boundary_points[0]]  # Close the loop
    #             lines = [[i, i+1] for i in range(len(self.boundary_points))]
                
    #             boundary_lines = o3d.geometry.LineSet()
    #             boundary_lines.points = o3d.utility.Vector3dVector(points)
    #             boundary_lines.lines = o3d.utility.Vector2iVector(lines)
    #             boundary_lines.colors = o3d.utility.Vector3dVector([[1, 1, 0]] * len(lines))  # Yellow
    #             vis.add_geometry(boundary_lines)
            
    #         vis.add_geometry(result_pcd)
            
    #         # Add coordinate frame
    #         coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    #         vis.add_geometry(coord_frame)
            
    #         selected_count = np.sum(boundary_mask)
    #         self.get_logger().info(f"Initial selection: {selected_count} points (Green)")
    #         self.get_logger().info("Close window to continue monitoring...")
            
    #         vis.run()
    #         vis.destroy_window()
            
    #     except Exception as e:
    #         self.get_logger().error(f"Initial selection visualization error: {e}")
    def show_initial_selection(self):
        """Show the initial rectangle selection result."""
        if self.current_pcd is None or self.boundary_mask is None:
            return

        boundary_mask = self.boundary_mask

        if not np.any(boundary_mask):
            self.get_logger().warn("No points in boundary for initial display")
            return

        try:
            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Initial Selection Result", width=1200, height=800)

            result_pcd = o3d.geometry.PointCloud()
            result_pcd.points = self.current_pcd.points

            colors = self.temperature_to_colors(self.current_temperatures)
            colors[boundary_mask] = [0, 1, 0]
            result_pcd.colors = o3d.utility.Vector3dVector(colors)

            corners_to_draw = self.rect_corners_snapped_3d if self.rect_corners_snapped_3d is not None else self.boundary_points

            for point in corners_to_draw:
                sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
                sphere.paint_uniform_color([1, 0, 0])
                sphere.translate(point)
                vis.add_geometry(sphere)

            if len(corners_to_draw) == 4:
                points = corners_to_draw + [corners_to_draw[0]]
                lines = [[0, 1], [1, 2], [2, 3], [3, 0]]

                boundary_lines = o3d.geometry.LineSet()
                boundary_lines.points = o3d.utility.Vector3dVector(points)
                boundary_lines.lines = o3d.utility.Vector2iVector(lines)
                boundary_lines.colors = o3d.utility.Vector3dVector([[1, 1, 0]] * len(lines))
                vis.add_geometry(boundary_lines)

            vis.add_geometry(result_pcd)

            coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            vis.add_geometry(coord_frame)

            selected_count = int(np.sum(boundary_mask))
            self.get_logger().info(f"Initial selection: {selected_count} points (Green)")
            self.get_logger().info("Close window to continue monitoring...")

            vis.run()
            vis.destroy_window()

        except Exception as e:
            self.get_logger().error(f"Initial selection visualization error: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ContinuousSurfaceSelector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
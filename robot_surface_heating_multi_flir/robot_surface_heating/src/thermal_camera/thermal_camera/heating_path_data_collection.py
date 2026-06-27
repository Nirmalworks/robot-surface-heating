#!/usr/bin/env python3

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import PoseArray

import threading
import time
import numpy as np
import open3d as o3d

import rclpy
from rclpy.node import Node
from std_srvs.srv import Trigger

from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2

from geometry_msgs.msg import Pose, Point, Quaternion
from thermal_camera_interfaces.msg import Extrema

import tf2_ros
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R


def safe_unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return v / n


def yaw_from_quat_xyzw(q: np.ndarray) -> float:
    return float(R.from_quat(q).as_euler("zyx")[0])


def quat_from_z_and_yaw(z_world: np.ndarray, yaw_world: float) -> np.ndarray:
    z = safe_unit(np.asarray(z_world, dtype=np.float64))

    yaw_rot = R.from_euler("z", yaw_world)
    x_ref = yaw_rot.apply([1.0, 0.0, 0.0])

    x = x_ref - np.dot(x_ref, z) * z
    if np.linalg.norm(x) < 1e-6:
        alt = np.array([0.0, 1.0, 0.0], dtype=np.float64) if abs(z[1]) < 0.9 else np.array([1.0, 0.0, 0.0], dtype=np.float64)
        x = alt - np.dot(alt, z) * z
    x = safe_unit(x)

    y = safe_unit(np.cross(z, x))

    rot_m = np.column_stack((x, y, z))
    return R.from_matrix(rot_m).as_quat()


def densify_polyline(points_xyz: np.ndarray, spacing: float) -> np.ndarray:
    if points_xyz.shape[0] < 2:
        return points_xyz.copy()

    out = [points_xyz[0].copy()]
    for i in range(points_xyz.shape[0] - 1):
        a = points_xyz[i]
        b = points_xyz[i + 1]
        d = b - a
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            continue
        nseg = max(1, int(np.ceil(L / spacing)))
        for k in range(1, nseg + 1):
            t = float(k) / float(nseg)
            out.append((1.0 - t) * a + t * b)
    return np.asarray(out, dtype=np.float64)


class HeatingPathDataCollection(Node):
    def __init__(self):
        super().__init__("heating_path_data_collection")

        self.cloud_sub = self.create_subscription(
            PointCloud2,
            "/selected_surface_points",
            self.cloud_callback,
            10
        )

        self.path_pub = self.create_publisher(Extrema, "/greedy_policy", 10)

        self.debug_posearray_pub = self.create_publisher(PoseArray, "/debug_heating_path_poses", 10)
        self.debug_path_pub = self.create_publisher(Path, "/debug_heating_path", 10)
        self.debug_line_pub = self.create_publisher(Marker, "/debug_heating_path_line", 10)

        self.select_srv = self.create_service(
            Trigger,
            "/select_heating_path",
            self.select_service_cb
        )

        self.publish_srv = self.create_service(
            Trigger,
            "/publish_heating_path",
            self.publish_service_cb
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.cloud_lock = threading.Lock()
        self.points = None
        self.normals = None
        self.frame_id = None
        self.pcd = None
        self.kdtree = None

        self.path_lock = threading.Lock()
        self.selected_points = None
        self.path_waypoints = None
        self.path_normals = None
        self.yaw_ref = None

        self.vis_thread = None
        self.vis_running = False
        self.select_requested = True

        self.waypoint_spacing = float(self.declare_parameter("waypoint_spacing_m", 0.01).value)
        self.normal_radius = float(self.declare_parameter("normal_radius_m", 0.04).value)
        self.normal_max_nn = int(self.declare_parameter("normal_max_nn", 30).value)
        self.tool_frame = str(self.declare_parameter("tool_frame", "heat_gun").value)
        self.world_frame = str(self.declare_parameter("world_frame", "world").value)

        self.get_logger().info("heating_path_data_collection running")
        self.get_logger().info("Waiting for /selected_surface_points")


    def select_service_cb(self, request, response):
        self.select_requested = True
        response.success = True
        response.message = "Selection requested"
        return response


    def publish_service_cb(self, request, response):
        ok = self.publish_current_path()
        response.success = bool(ok)
        response.message = "Published" if ok else "No path available"
        return response

    def flip_normals_world_up(self, frame_id: str, normals_local: np.ndarray) -> np.ndarray:
        try:
            t = self.tf_buffer.lookup_transform(
                self.world_frame,
                frame_id,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
            q = t.transform.rotation
            rot = R.from_quat([q.x, q.y, q.z, q.w])
            n_world = rot.apply(normals_local)

            flip = (n_world[:, 2] < 0.0)
            normals_local = normals_local.copy()
            normals_local[flip] *= -1.0
            return normals_local
        except Exception as e:
            self.get_logger().warn(f"Normal flip using world_up failed: {e}")
            return normals_local

    def cloud_callback(self, msg: PointCloud2):
        pts = []
        for p in point_cloud2.read_points(msg, field_names=("x", "y", "z", "thermal"), skip_nans=True):
            x, y, z, _ = p
            pts.append([x, y, z])

        if len(pts) == 0:
            return

        pts = np.asarray(pts, dtype=np.float64)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius,
                max_nn=self.normal_max_nn
            )
        )

        pcd.orient_normals_consistent_tangent_plane(50)

        normals = np.asarray(pcd.normals, dtype=np.float64)
        normals[normals[:, 2] < 0.0] *= -1.0
        normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-12)
        pcd.normals = o3d.utility.Vector3dVector(normals)

        with self.cloud_lock:
            self.points = pts
            self.normals = normals
            self.frame_id = msg.header.frame_id
            self.pcd = pcd
            self.kdtree = cKDTree(pts)

        if self.select_requested and (not self.vis_running):
            self.select_requested = False
            self.start_picker_thread()


    def start_picker_thread(self):
        self.vis_running = True
        self.vis_thread = threading.Thread(target=self.run_picker, daemon=True)
        self.vis_thread.start()


    def run_picker(self):
        try:
            with self.cloud_lock:
                pcd = self.pcd
                pts = None if self.points is None else self.points.copy()

            if pcd is None or pts is None:
                self.get_logger().warn("No pointcloud available for selection")
                return

            self.get_logger().info("Picker instructions")
            self.get_logger().info("Shift + Left Click points in order")
            self.get_logger().info("Close window when done")

            vis = o3d.visualization.VisualizerWithEditing()
            vis.create_window(window_name="Select heating path points (ordered)", width=1200, height=800)
            vis.add_geometry(pcd)
            vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
            vis.run()
            picked = vis.get_picked_points()
            vis.destroy_window()

            if len(picked) < 2:
                self.get_logger().warn("Pick at least 2 points")
                return

            selected = np.asarray([pts[i] for i in picked], dtype=np.float64)

            waypoints = densify_polyline(selected, spacing=self.waypoint_spacing)

            with self.cloud_lock:
                tree = self.kdtree
                normals = self.normals

            if tree is None or normals is None:
                self.get_logger().warn("Normals not ready")
                return

            _, nn_idx = tree.query(waypoints, k=1)
            wp_normals = normals[nn_idx]
            wp_normals = wp_normals / (np.linalg.norm(wp_normals, axis=1, keepdims=True) + 1e-12)
            wp_normals = self.flip_normals_world_up(self.frame_id, wp_normals)

            yaw_ref = self.get_or_latch_yaw_ref()

            with self.path_lock:
                self.selected_points = selected
                self.path_waypoints = waypoints
                self.path_normals = wp_normals
                self.yaw_ref = yaw_ref

            self.get_logger().info(f"Selected {len(selected)} points, generated {len(waypoints)} waypoints")
            self.publish_current_path()

            self.show_path_preview(selected, waypoints)

        except Exception as e:
            self.get_logger().error(f"Picker error: {e}")
        finally:
            self.vis_running = False


    def show_path_preview(self, selected: np.ndarray, waypoints: np.ndarray):
        try:
            with self.cloud_lock:
                pcd = self.pcd
            if pcd is None:
                return

            lines = [[i, i + 1] for i in range(len(waypoints) - 1)]
            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(waypoints)
            line_set.lines = o3d.utility.Vector2iVector(lines)
            line_set.colors = o3d.utility.Vector3dVector(np.tile(np.array([[1.0, 1.0, 0.0]]), (len(lines), 1)))

            spheres = []
            for p in selected:
                s = o3d.geometry.TriangleMesh.create_sphere(radius=0.01)
                s.paint_uniform_color([1.0, 0.0, 0.0])
                s.translate(p)
                spheres.append(s)

            vis = o3d.visualization.Visualizer()
            vis.create_window(window_name="Heating path preview", width=1200, height=800)
            vis.add_geometry(pcd)
            vis.add_geometry(line_set)
            for s in spheres:
                vis.add_geometry(s)
            vis.add_geometry(o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1))
            vis.run()
            vis.destroy_window()
        except Exception:
            return


    def get_or_latch_yaw_ref(self) -> float:
        try:
            tf_tool = self.tf_buffer.lookup_transform(
                self.world_frame,
                self.tool_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.2),
            )
            q = tf_tool.transform.rotation
            return yaw_from_quat_xyzw(np.array([q.x, q.y, q.z, q.w], dtype=np.float64))
        except Exception as e:
            self.get_logger().warn(f"Yaw reference lookup failed, using 0: {e}")
            return 0.0


    def publish_current_path(self) -> bool:
        with self.path_lock:
            waypoints = None if self.path_waypoints is None else self.path_waypoints.copy()
            wp_normals = None if self.path_normals is None else self.path_normals.copy()
            yaw_ref = self.yaw_ref

        with self.cloud_lock:
            frame_id = self.frame_id

        if waypoints is None or wp_normals is None or frame_id is None:
            return False

        msg = Extrema()
        msg.header.frame_id = frame_id
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.value = float("nan")

        poses = []
        for p, n in zip(waypoints, wp_normals):
            q = quat_from_z_and_yaw(n, yaw_ref)
            pose = Pose()
            pose.position = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
            pose.orientation = Quaternion(x=float(q[0]), y=float(q[1]), z=float(q[2]), w=float(q[3]))
            poses.append(pose)

        msg.poses = poses
        self.path_pub.publish(msg)

        # PoseArray (same frame as Extrema)
        pa = PoseArray()
        pa.header = msg.header
        pa.poses = poses
        self.debug_posearray_pub.publish(pa)

        # Path (PoseStamped list)
        path = Path()
        path.header = msg.header
        for pose in poses:
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose = pose
            path.poses.append(ps)
        self.debug_path_pub.publish(path)

        # Line strip marker for a clean polyline
        line = Marker()
        line.header = msg.header
        line.ns = "heating_path"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.pose.orientation.w = 1.0
        line.scale.x = 0.003
        line.color = ColorRGBA(r=1.0, g=1.0, b=0.0, a=1.0)
        line.points = [Point(x=float(p.position.x), y=float(p.position.y), z=float(p.position.z)) for p in poses]
        line.lifetime.sec = 0
        self.debug_line_pub.publish(line)

        self.get_logger().info(f"Published Extrema path with {len(poses)} poses on /greedy_policy")
        return True


def main(args=None):
    rclpy.init(args=args)
    node = HeatingPathDataCollection()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
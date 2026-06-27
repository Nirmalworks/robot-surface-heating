#!/usr/bin/env python3

import os
import math
import struct
import numpy as np
import trimesh

import rclpy
from rclpy.node import Node
from scipy.spatial.transform import Rotation as R

from visualization_msgs.msg import Marker
from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header
from geometry_msgs.msg import TransformStamped
from tf2_ros.static_transform_broadcaster import StaticTransformBroadcaster


# ============================================================
# Hard-coded defaults
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Options:
#   "stl_lattice"       -> STL lattice through stl_to_lattice.py
#   "txt_lattice"   -> TXT lattice through txt_to_lattice.py
#   "legacy_grid"   -> old STL grid method
#   "legacy_sample" -> old STL random surface sample
POINTCLOUD_SOURCE_DEFAULT = "txt_lattice"
# POINTCLOUD_SOURCE_DEFAULT = "stl_lattice"

# STL_FILE_NAME = "RSS_v3.stl"
# STL_FILE_NAME = "saddle_mold.STL"
# TXT_FILE_NAME = "nd-composite-core.txt"
# TXT_FILE_NAME = "nd-metal-curved.txt"

# STL_FILE_PATH = os.path.join(BASE_DIR, STL_FILE_NAME)
# TXT_FILE_PATH = os.path.join(BASE_DIR, TXT_FILE_NAME)
TXT_FILE_PATH = "/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/nd-composite-core.txt"
# TXT_FILE_PATH = "/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/nd-metal-curved.txt"
# STL_FILE_PATH = "/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/saddle_mold.STL"
STL_FILE_PATH = "/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/RSS_v3.stl"

STL_LATTICE_NPZ_PATH = os.path.join(BASE_DIR, "..", "npz", "cad_lattice_cache.npz")
TXT_LATTICE_NPZ_PATH = os.path.join(BASE_DIR, "..", "npz", "txt_lattice_cache.npz")

FRAME_ID_DEFAULT = "base_link"
POINTCLOUD_FRAME_ID = "cad_pointcloud_frame"

MARKER_X_DEFAULT = 0.8
MARKER_Y_DEFAULT = 0.0
MARKER_Z_DEFAULT = 0.2

MARKER_SCALE_DEFAULT = 1.0

LATTICE_FORCE_REBUILD_DEFAULT = True
LATTICE_SAVE_CACHE_DEFAULT = True


try:
    import thermal_camera.stl_to_lattice as stl_lattice
except Exception as e:
    stl_lattice = None
    print(f"Failed to import stl_to_lattice: {e}")

try:
    import thermal_camera.txt_to_lattice as txt_lattice
except Exception as e:
    txt_lattice = None
    print(f"Failed to import txt_to_lattice: {e}")


def convert_cloud_from_numpy(points: np.ndarray, normals: np.ndarray, frame_id: str) -> PointCloud2:
    fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="normal_x", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="normal_y", offset=16, datatype=PointField.FLOAT32, count=1),
        PointField(name="normal_z", offset=20, datatype=PointField.FLOAT32, count=1),
    ]

    cloud_data = []
    for p, n in zip(points, normals):
        cloud_data.append(
            struct.pack(
                "ffffff",
                float(p[0]), float(p[1]), float(p[2]),
                float(n[0]), float(n[1]), float(n[2]),
            )
        )
    cloud_data = b"".join(cloud_data)

    header = Header()
    header.stamp = rclpy.time.Time().to_msg()
    header.frame_id = frame_id

    cloud_msg = PointCloud2()
    cloud_msg.header = header
    cloud_msg.height = 1
    cloud_msg.width = int(points.shape[0])
    cloud_msg.fields = fields
    cloud_msg.is_bigendian = False
    cloud_msg.point_step = 24
    cloud_msg.row_step = cloud_msg.point_step * cloud_msg.width
    cloud_msg.data = cloud_data
    cloud_msg.is_dense = True
    return cloud_msg


class SimpleCADVisualizer(Node):
    def __init__(self):
        super().__init__("simple_cad_visualizer")

        # self.get_logger().info("TESTTTTTT")

        self.declare_parameter("cad_file", self._default_geometry_file())
        self.declare_parameter("frame_id", FRAME_ID_DEFAULT)
        self.declare_parameter("x", MARKER_X_DEFAULT)
        self.declare_parameter("y", MARKER_Y_DEFAULT)
        self.declare_parameter("z", MARKER_Z_DEFAULT)
        self.declare_parameter("scale", MARKER_SCALE_DEFAULT)

        self.declare_parameter("lattice_npz_path", self._default_lattice_npz_path())
        self.declare_parameter("lattice_force_rebuild", LATTICE_FORCE_REBUILD_DEFAULT)
        self.declare_parameter("lattice_save_cache", LATTICE_SAVE_CACHE_DEFAULT)
        self.declare_parameter("pointcloud_source", POINTCLOUD_SOURCE_DEFAULT)

        self.cad_file = str(self.get_parameter("cad_file").value)
        self.frame_id = str(self.get_parameter("frame_id").value)

        self.marker_pub = self.create_publisher(Marker, "cad_model", 10)
        self.pointcloud_pub = self.create_publisher(PointCloud2, "cad_pointcloud", 10)

        self.marker_timer = self.create_timer(0.1, self.publish_visualization)
        self.cloud_timer = self.create_timer(1.0 / 30.0, self.publish_pointcloud)

        self.calculated_cad_to_world_frame: np.ndarray | None = None
        self.broadcaster = StaticTransformBroadcaster(self)

        self._points = np.zeros((0, 3), dtype=np.float32)
        self._normals = np.zeros((0, 3), dtype=np.float32)
        self._lattice_meta: dict = {}

        self._validate_geometry_file()
        self._load_pointcloud_geometry()

        self.get_logger().info("CAD Visualizer is running")

    def _default_geometry_file(self) -> str:
        if POINTCLOUD_SOURCE_DEFAULT == "txt_lattice":
            return TXT_FILE_PATH
        return STL_FILE_PATH

    def _default_lattice_npz_path(self) -> str:
        if POINTCLOUD_SOURCE_DEFAULT == "txt_lattice":
            return TXT_LATTICE_NPZ_PATH
        return STL_LATTICE_NPZ_PATH

    def _validate_geometry_file(self) -> None:
        source = str(self.get_parameter("pointcloud_source").value).strip().lower()

        if source == "txt_lattice":
            self.cad_file = TXT_FILE_PATH
        else:
            self.cad_file = STL_FILE_PATH

        if os.path.exists(self.cad_file):
            self.get_logger().info(f"Using geometry file: {self.cad_file}")
        else:
            self.get_logger().warning(f"Geometry file not found: {self.cad_file}")
            self.get_logger().warning("Pointcloud will be empty.")

    def _load_pointcloud_geometry(self) -> None:
        source = str(self.get_parameter("pointcloud_source").value).strip().lower()

        if not self.cad_file or not os.path.exists(self.cad_file):
            self.get_logger().warning("No geometry file available, pointcloud will be empty")
            self._points = np.zeros((0, 3), dtype=np.float32)
            self._normals = np.zeros((0, 3), dtype=np.float32)
            return

        if source == "txt_lattice":
            self._load_txt_lattice()
            return

        if source == "stl_lattice":
            self._load_stl_lattice()
            return

        if source == "legacy_grid":
            self.get_logger().info("Loading mesh and generating XZ grid with legacy path")
            self._points, self._normals = self.generate_xz_grid(self.cad_file, spacing=0.01)
            self.get_logger().info(f"Loaded pointcloud: {self._points.shape[0]} points")
            return

        if source == "legacy_sample":
            self.get_logger().info("Loading mesh and sampling surface points with legacy path")
            self._points, self._normals = self.load_and_sample(self.cad_file, 20000)
            self.get_logger().info(f"Loaded pointcloud: {self._points.shape[0]} points")
            return

        self.get_logger().warning(f"Unknown pointcloud_source '{source}', publishing empty cloud")
        self._points = np.zeros((0, 3), dtype=np.float32)
        self._normals = np.zeros((0, 3), dtype=np.float32)

    def _load_txt_lattice(self) -> None:
        if txt_lattice is None:
            self.get_logger().error("txt_to_lattice could not be imported")
            self._points = np.zeros((0, 3), dtype=np.float32)
            self._normals = np.zeros((0, 3), dtype=np.float32)
            return

        npz_path = str(self.get_parameter("lattice_npz_path").value).strip()
        if not npz_path:
            npz_path = TXT_LATTICE_NPZ_PATH

        force_rebuild = bool(self.get_parameter("lattice_force_rebuild").value)
        save_cache = bool(self.get_parameter("lattice_save_cache").value)

        self.get_logger().info("Loading TXT lattice pointcloud")
        self.get_logger().info(f"TXT path: {self.cad_file}")
        self.get_logger().info(f"Lattice cache path: {npz_path}")

        pts, nrm, meta = txt_lattice.load_or_build_lattice_points_normals(
            txt_path=self.cad_file,
            npz_path=npz_path,
            force_rebuild=force_rebuild,
            save_cache=save_cache,
        )

        self._points = np.asarray(pts, dtype=np.float32)
        self._normals = np.asarray(nrm, dtype=np.float32)
        self._lattice_meta = meta if isinstance(meta, dict) else {}

        self._validate_loaded_cloud("TXT lattice")

    def _load_stl_lattice(self) -> None:
        if stl_lattice is None:
            self.get_logger().error("stl_to_lattice could not be imported")
            self._points = np.zeros((0, 3), dtype=np.float32)
            self._normals = np.zeros((0, 3), dtype=np.float32)
            return

        npz_path = str(self.get_parameter("lattice_npz_path").value).strip()
        if not npz_path:
            npz_path = STL_LATTICE_NPZ_PATH

        force_rebuild = bool(self.get_parameter("lattice_force_rebuild").value)
        save_cache = bool(self.get_parameter("lattice_save_cache").value)

        self.get_logger().info("Loading STL lattice pointcloud")
        self.get_logger().info(f"STL path: {self.cad_file}")
        self.get_logger().info(f"Lattice cache path: {npz_path}")

        pts, nrm, meta = stl_lattice.load_or_build_lattice_points_normals(
            stl_path=self.cad_file,
            npz_path=npz_path,
            force_rebuild=force_rebuild,
            save_cache=save_cache,
        )

        self._points = np.asarray(pts, dtype=np.float32)
        self._normals = np.asarray(nrm, dtype=np.float32)
        self._lattice_meta = meta if isinstance(meta, dict) else {}

        self._validate_loaded_cloud("STL lattice")

    def _validate_loaded_cloud(self, label: str) -> None:
        if self._points.shape[0] != self._normals.shape[0]:
            self.get_logger().error(f"{label} points and normals length mismatch, publishing empty cloud")
            self._points = np.zeros((0, 3), dtype=np.float32)
            self._normals = np.zeros((0, 3), dtype=np.float32)
            return

        if self._points.shape[0] == 0:
            self.get_logger().warning(f"{label} loaded zero points")
            return

        mins = self._points.min(axis=0)
        maxs = self._points.max(axis=0)
        ranges = maxs - mins

        self.get_logger().info(f"{label} loaded: {self._points.shape[0]} points")
        self.get_logger().info(
            "Pointcloud bounds: "
            f"x=[{mins[0]:.4f}, {maxs[0]:.4f}] "
            f"y=[{mins[1]:.4f}, {maxs[1]:.4f}] "
            f"z=[{mins[2]:.4f}, {maxs[2]:.4f}]"
        )
        self.get_logger().info(
            "Pointcloud size: "
            f"dx={ranges[0]:.4f} m "
            f"dy={ranges[1]:.4f} m "
            f"dz={ranges[2]:.4f} m"
        )

    def publish_pointcloud_transform(self) -> None:
        if self.calculated_cad_to_world_frame is None:
            return

        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id = "base_link"
        t.child_frame_id = POINTCLOUD_FRAME_ID

        t.transform.translation.x = float(self.calculated_cad_to_world_frame[0])
        t.transform.translation.y = float(self.calculated_cad_to_world_frame[1])
        t.transform.translation.z = float(self.calculated_cad_to_world_frame[2])

        t.transform.rotation.x = float(self.calculated_cad_to_world_frame[3])
        t.transform.rotation.y = float(self.calculated_cad_to_world_frame[4])
        t.transform.rotation.z = float(self.calculated_cad_to_world_frame[5])
        t.transform.rotation.w = float(self.calculated_cad_to_world_frame[6])

        self.broadcaster.sendTransform(t)

    def load_and_sample(self, mesh_path: str, n_points: int = 5000) -> tuple[np.ndarray, np.ndarray]:
        mesh = trimesh.load_mesh(mesh_path)
        points, face_indices = trimesh.sample.sample_surface(mesh, n_points)
        normals = mesh.face_normals[face_indices]
        return points.astype(np.float32), normals.astype(np.float32)

    def generate_xz_grid(self, mesh_path: str, spacing: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        mesh = trimesh.load_mesh(mesh_path)
        bounds = mesh.bounds
        xmin, ymin, zmin = bounds[0]
        xmax, _, zmax = bounds[1]

        xs = np.arange(xmin, xmax, spacing)
        zs = np.arange(zmin, zmax, spacing)
        xx, zz = np.meshgrid(xs, zs)
        yy = np.full_like(xx, ymin)

        grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])

        closest, _, face_id = mesh.nearest.on_surface(grid_points)
        valid = face_id != -1
        closest = closest[valid]
        normals = mesh.face_normals[face_id[valid]]

        return closest.astype(np.float32), normals.astype(np.float32)

    def publish_pointcloud(self) -> None:
        if self._points.shape[0] == 0:
            return

        scale = 1.0
        scaled_points = self._points * scale

        cloud_msg = convert_cloud_from_numpy(
            scaled_points,
            self._normals,
            POINTCLOUD_FRAME_ID,
        )
        cloud_msg.header.stamp = self.get_clock().now().to_msg()
        self.pointcloud_pub.publish(cloud_msg)

    def publish_visualization(self) -> None:
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "cad_visualization"
        marker.id = 0
        marker.action = Marker.ADD

        if self.calculated_cad_to_world_frame is None:
            marker.pose.position.x = float(self.get_parameter("x").value)
            marker.pose.position.y = float(self.get_parameter("y").value)
            marker.pose.position.z = float(self.get_parameter("z").value)
            marker.pose.orientation.w = 1.0
        else:
            marker.pose.position.x = float(self.calculated_cad_to_world_frame[0])
            marker.pose.position.y = float(self.calculated_cad_to_world_frame[1])
            marker.pose.position.z = float(self.calculated_cad_to_world_frame[2])
            marker.pose.orientation.x = float(self.calculated_cad_to_world_frame[3])
            marker.pose.orientation.y = float(self.calculated_cad_to_world_frame[4])
            marker.pose.orientation.z = float(self.calculated_cad_to_world_frame[5])
            marker.pose.orientation.w = float(self.calculated_cad_to_world_frame[6])

        source = str(self.get_parameter("pointcloud_source").value).strip().lower()

        if source in ("lattice", "legacy_grid", "legacy_sample") and self.cad_file:
            marker.type = Marker.MESH_RESOURCE
            marker.mesh_resource = f"file://{self.cad_file}"
            marker.mesh_use_embedded_materials = False

            scale = float(self.get_parameter("scale").value)
            marker.scale.x = scale
            marker.scale.y = scale
            marker.scale.z = scale

            marker.color.r = 0.3
            marker.color.g = 0.9
            marker.color.b = 0.4
            marker.color.a = 0.6
        else:
            marker.type = Marker.CUBE
            marker.scale.x = 0.03
            marker.scale.y = 0.03
            marker.scale.z = 0.03

            marker.color.r = 0.2
            marker.color.g = 0.3
            marker.color.b = 1.0
            marker.color.a = 0.4

        self.marker_pub.publish(marker)

    def model_point_frame_calculation(self, mdl_pts, raw_wld_pts, eef_offset) -> None:
        wld_pts = []
        for T_point in raw_wld_pts:
            wld_pts.append(T_point[:3, 3] + T_point[:3, :3] @ eef_offset)

        A = mdl_pts if isinstance(mdl_pts, np.ndarray) else np.array(mdl_pts)
        B = wld_pts if isinstance(wld_pts, np.ndarray) else np.array(wld_pts)

        centroid_A = np.mean(A, axis=0)
        centroid_B = np.mean(B, axis=0)

        AA = A - centroid_A
        BB = B - centroid_B

        H = AA.T @ BB
        U, _, Vt = np.linalg.svd(H)
        R_mat = Vt.T @ U.T

        if np.linalg.det(R_mat) < 0:
            Vt[-1, :] *= -1
            R_mat = Vt.T @ U.T

        t = centroid_B - R_mat @ centroid_A
        q = R.from_matrix(R_mat).as_quat()

        self.calculated_cad_to_world_frame = np.concatenate([t.flatten(), q.flatten()], axis=0)


def main(args=None):
    rclpy.init(args=args)

    try:
        node = SimpleCADVisualizer()

        # Replace these with reference points from the active geometry source.
        # For txt_lattice, these must be scanner/TXT-frame points after scaling.
        # # ====== RSS_V3 ======
        # model_pts = np.array([
        #     [0.0, 0.0, 0.0],
        #     [0.566184, 0.036625, 0.0],
        #     [0.434884, 0.028132, 0.824438],
        #     [0.131299, 0.008494, 0.824438],
        # ], dtype=np.float64)

        # raw_world_pts = np.array([
        #     [[-0.50285053,  0.83220158, -0.23362762,  0.37286872],
        #      [ 0.79615431,  0.55116913,  0.24970166, -0.59651353],
        #      [ 0.33657044, -0.06044102, -0.93971656,  0.11252270],
        #      [ 0.0,         0.0,         0.0,         1.0       ]],

        #     [[-0.68702274,  0.65417952, -0.31630510,  0.38965210],
        #      [ 0.71371420,  0.68924454, -0.12471573,  0.00648864],
        #      [ 0.13642509, -0.31143398, -0.94042388,  0.10766228],
        #      [ 0.0,         0.0,         0.0,         1.0       ]],

        #     [[-0.40509465,  0.90951455,  0.09317513,  1.16361879],
        #      [ 0.83992716,  0.41047027, -0.35501624, -0.11501234],
        #      [-0.36113806, -0.06555486, -0.93020528,  0.11261413],
        #      [ 0.0,         0.0,         0.0,         1.0       ]],

        #     [[-0.46870058,  0.78253980,  0.40981854,  1.12083144],
        #      [ 0.80482642,  0.18706913,  0.56325800, -0.52292733],
        #      [ 0.36410741,  0.59383214, -0.71748811,  0.09046926],
        #      [ 0.0,         0.0,         0.0,         1.0       ]],
        # ], dtype=np.float64)
        # ====== RSS_v3 ======


        # ====== 5/4/26 - nd-composite-core ======
        model_pts = np.array([
            [3.2950446164,  0.3623536314, -0.7178479420],
            [3.2953461398, -0.5889201074, -0.7428639956],
            [2.9772347364, -0.5807347034, -0.7247596376],
            [2.9839366534, -0.0246957342, -0.7334778066],
        ], dtype=np.float64)

        raw_world_pts = np.array([
            [
                [-0.46850533,  0.74054825,  0.48175828,  1.08229889],
                [ 0.86926757,  0.48376204,  0.10172602, -0.50459995],
                [-0.15772335,  0.46643603, -0.87037967,  0.13803347],
                [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
            ],

            [
                [-0.01855688,  0.96736633,  0.25270146,  0.15040232],
                [ 0.98278031, -0.02882530,  0.18251563, -0.47969434],
                [ 0.18384367,  0.25173694, -0.95017368,  0.12326173],
                [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
            ],

            [
                [-0.67909689,  0.47708853, -0.55786552,  0.27177563],
                [ 0.70828119,  0.62547370, -0.32729254, -0.11054010],
                [ 0.19278270, -0.61738900, -0.76267008,  0.08819744],
                [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
            ],

            [
                [-0.66089862,  0.74844450, -0.05517110,  0.76318849],
                [ 0.69919484,  0.58736449, -0.40758990, -0.12964803],
                [-0.27265287, -0.30795095, -0.91149691,  0.10703047],
                [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
            ],
        ], dtype=np.float64)
        # ====== 5/4/26 - nd-composite-core ======
    

        # ====== 5/8/2026 - metallic bumper ======
        # model_pts = np.array([
        #     [ 2.98496083, -0.45075358, -0.67443975],
        #     [ 3.08503960, -0.51633549, -0.64641098],
        #     [ 3.40341862, -0.50227822, -0.69705441],
        #     [ 3.42778573, -0.15561348, -0.58761660],
        # ], dtype=np.float64)

        # raw_world_pts = np.array([
        #     [
        #         [ 0.03261550,  0.99465377,  0.09798015,  0.50748649],
        #         [ 0.99657041, -0.03982365,  0.07253610, -0.13266116],
        #         [ 0.07605023,  0.09527832, -0.99254139,  0.18956655],
        #         [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
        #     ],

        #     [
        #         [ 0.03314553,  0.99511695,  0.09297116,  0.44479219],
        #         [ 0.99677799, -0.03971175,  0.06968940, -0.21542828],
        #         [ 0.07304115,  0.09036171, -0.99322694,  0.22021154],
        #         [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
        #     ],

        #     [
        #         [ 0.23680893,  0.92625230, -0.29322040,  0.51048264],
        #         [ 0.80370338, -0.01718968,  0.59478180, -0.60755023],
        #         [ 0.54587765, -0.37651187, -0.74850278,  0.14500380],
        #         [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
        #     ],

        #     [
        #         [ 0.06090890,  0.97385805, -0.21883923,  0.84611630],
        #         [ 0.79865320,  0.08395738,  0.59590621, -0.62603092],
        #         [ 0.59870123, -0.21107265, -0.77266110,  0.25802705],
        #         [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
        #     ],
        # ], dtype=np.float64)
        # ====== 5/8/2026 - metallic bumper ======


        # ===== 5/12/26 saddle_mold =====
        # model_pts = np.array([
        #     [ 0.63999999,  0.52134365,  0.04701440],
        #     [ 0.53999114,  0.02249115,  0.06264786],
        #     [ 0.02000847,  0.02134394,  0.03373664],
        #     [ 0.16028135,  0.56040597,  0.07143027],
        # ], dtype=np.float64)


        # raw_world_pts = np.array([
        #     [
        #         [-0.04665562,  0.99881910, -0.01355225,  0.91629551],
        #         [ 0.99259383,  0.04787969,  0.11164690, -0.65115599],
        #         [ 0.11216393, -0.00824293, -0.99365553,  0.18104249],
        #         [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
        #     ],

        #     [
        #         [ 0.18761593,  0.91976111,  0.34473145,  0.34793364],
        #         [ 0.98204452, -0.16860038, -0.08463144, -0.52302896],
        #         [-0.01971885,  0.35441983, -0.93487847,  0.17109212],
        #         [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
        #     ],

        #     [
        #         [ 0.00946937,  0.99727816,  0.07312044,  0.36409055],
        #         [ 0.88986645, -0.04175919,  0.45430593, -0.05495848],
        #         [ 0.45612283,  0.06076544, -0.88783981,  0.13907436],
        #         [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
        #     ],

        #     [
        #         [ 0.19985015,  0.92749848, -0.31592165,  0.99014369],
        #         [ 0.90172617, -0.04795438,  0.42963973, -0.18146892],
        #         [ 0.38334037, -0.37073838, -0.84593334,  0.18422947],
        #         [ 0.00000000,  0.00000000,  0.00000000,  1.00000000],
        #     ],
        # ], dtype=np.float64)
        # ====== 5/12/26 saddle_mold =====


        eef_pointy_t = np.array([0.0, 0.0, 0.111125], dtype=np.float64)

        node.model_point_frame_calculation(model_pts, raw_world_pts, eef_pointy_t)
        node.publish_pointcloud_transform()

        rclpy.spin(node)

    except KeyboardInterrupt:
        print("\nShutting down CAD visualizer...")
    finally:
        rclpy.shutdown()


if __name__ == "__main__":
    main()
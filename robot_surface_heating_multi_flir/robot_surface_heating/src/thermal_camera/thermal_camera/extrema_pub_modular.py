#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from std_msgs.msg import Header, Float64
from sensor_msgs.msg import PointCloud2
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker
from thermal_camera_interfaces.msg import Extrema
import time

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Vector3Stamped

from thermal_camera.common_motionplan_utilities import get_best_ik, get_fk
from moveit_msgs.srv import GetPositionIK, GetPositionFK
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

# verbosity setting
verbose = False
# verbose = True
def loginfo(node: Node, msg: str):
    node.get_logger().info(msg)
def logwarn(node: Node, msg: str):
    node.get_logger().warn(msg)
def logerr(node: Node, msg: str):
    node.get_logger().error(msg)

if not verbose:
    loginfo = lambda *args: None  # noqa
    logwarn = lambda *args: None  # noqa
    logerr = lambda *args: None  # noqa

def safe_normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])

def ktoc(val):
  return (val - 27315) / 100.0

def ktof(val):
  return (1.8 * ktoc(val) + 32.0)

from sklearn.neighbors import NearestNeighbors

def compute_pca_normals(points: np.ndarray, k: int = 30) -> np.ndarray:
    """
    Compute normals using PCA and orient them consistently to +Z.
    
    Args:
        points: (N, 3) array of point cloud coordinates
        k: Number of neighbors to use for local PCA

    Returns:
        normals: (N, 3) array of unit normal vectors
    """
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto').fit(points)
    _, indices = nbrs.kneighbors(points)

    normals = np.zeros_like(points)

    for i in range(len(points)):
        neighbor_pts = points[indices[i]]
        cov = np.cov(neighbor_pts.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]  # eigenvector with smallest eigenvalue

        # Enforce normal to face upward
        if normal[2] < 0:
            normal *= -1

        normals[i] = normal

    return normals

def pose_to_position(pose: Pose):
    """Converts Pose object into a numpy array with shape (3,) for x,y,z"""
    pos = np.array([pose.position.x,
                    pose.position.y,
                    pose.position.z])
    return pos

def within_radius(p1: np.ndarray, p2: np.ndarray, radius: float) -> bool:
    # fast: compare squared distance to avoid sqrt
    return np.sum((p2 - p1)**2) <= radius**2

def normal_to_quaternion_batched(normals: np.ndarray) -> np.ndarray:
    """
    Convert a (N, 3) array of normals to (N, 4) array of quaternions aligning +Z to each normal.
    
    Args:
        normals: (N, 3) array of surface normal vectors.
    
    Returns:
        quaternions: (N, 4) array of quaternions (x, y, z, w).
    """
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)  # Normalize Z axes
    z_axis = normals

    # Use x_axis = [1, 0, 0] unless z is too close to x (then use [0, 1, 0])
    x_ref = np.tile(np.array([[1.0, 0.0, 0.0]]), (normals.shape[0], 1))
    alt_x_ref = np.tile(np.array([[0.0, 1.0, 0.0]]), (normals.shape[0], 1))
    dot_products = np.abs(np.sum(z_axis * x_ref, axis=1))
    switch_mask = dot_products > 0.999  # nearly parallel
    x_axis = np.where(switch_mask[:, None], alt_x_ref, x_ref)

    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis, axis=1, keepdims=True)
    x_axis = np.cross(y_axis, z_axis)

    # Rotation matrix: stack x, y, z as columns (shape: N, 3, 3)
    rot_matrices = np.stack((x_axis, y_axis, z_axis), axis=2)

    # Convert to quaternions (requires loop in scipy)
    quaternions = R.from_matrix(rot_matrices).as_quat()  # (N, 4)

    return quaternions

class CurrentPose:
    """Container class for best extrema point and value."""
    def __init__(self, frame_id, stamp):
        # initialize starting best_val to a safe default pose
        self.best_val: Extrema = Extrema()
        self.best_val.header.frame_id = frame_id
        self.best_val.header.stamp = stamp
        # self.best_val.pose.position = Point(
        #     x=0.6183806415480197, y=-0.49666659881919156, z=0.0525086918362738
        # )
        # self.best_val.pose.orientation = Quaternion(
        #     x=-0.0021976900015181536, y=0.001907758224241607, z=0.6985991810938093, w=0.7155073128852526
        # )
    
        default_pose = Pose()
        default_pose.position = Point(
            x=0.6183806415480197, y=-0.49666659881919156, z=0.0525086918362738
        )
        default_pose.orientation = Quaternion(
            x=-0.0021976900015181536, y=0.001907758224241607, z=0.6985991810938093, w=0.7155073128852526
        )

        self.best_val.poses = [default_pose]

class RawThermalExtremaPublisher(Node):
    timeout_sec_ = 5.0
    move_group_name_ = "ur10e"
    joint_state_topic_ = "joint_states"
    ik_srv_name_ = "compute_ik"
    fk_srv_name_ = "compute_fk"
    base_ = "base_link"
    end_effector_ = "tool0"

    def __init__(self):
        super().__init__('raw_thermal_extrema_publisher')

        self.pub_debug_hot = self.create_publisher(PoseStamped, '/debug_hot_pose', 10)
        self.pub_debug_cold = self.create_publisher(PoseStamped, '/debug_cold_pose', 10)

        self.pub_hottest = self.create_publisher(Extrema, '/hottest_pose', 10)
        self.pub_coldest = self.create_publisher(Extrema, '/coldest_pose', 10)
        self.greedy_pub = self.create_publisher(Extrema, '/greedy_policy', 10)

        start_time = self.get_clock().now().to_msg()
        self.cur_coldest_pose: Extrema = CurrentPose('cad_pointcloud_frame', start_time)
        self.cur_hottest_pose: Extrema = CurrentPose('cad_pointcloud_frame', start_time)
        self.marker_pub = self.create_publisher(Marker, "/heatmap_markers", 10)

        self.ema_hottest_point = None
        self.ema_coldest_point = None
        self.ema_hottest_normal = None
        self.ema_coldest_normal = None
        self.ema_alpha = 0.3

        self.latest_coldest_pub = None
        self.latest_point_pub = None
        self.latest_pub_time = time.perf_counter() - 5.0
        self.latest_temp_pub = None
        
        self.z_offset = 0.35

        # TF
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # plotting utilities
        self.desired_temperature = 33.0 # Celsius
        self.average_temps = []
        self.mean_error_from_desired = []
        self.std_dev_error_from_desired = []

        # FK/IK clients
        self.debug_pose_pub = self.create_publisher(PoseStamped, '/robot_target_pose', 10)
        self.ik_client_ = self.create_client(GetPositionIK, self.ik_srv_name_)
        if not self.ik_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            # self.get_logger().error("IK service not available.")
            logerr(self,"IK service not available.")
            exit(1)

        self.fk_client_ = self.create_client(GetPositionFK, self.fk_srv_name_)
        if not self.fk_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            # self.get_logger().error("FK service not available.")
            logerr(self,"FK service not available.")
            exit(1)
        # self.get_logger().info("Created FK/IK service clients.")
        loginfo(self, "Created FK/IK service clients.")

        # main callback subscriber is allowed to process
        # messages once the posemap callback finishes
        # self.pose_map_ready = False
        self.pose_map_ready = True
        self.subscription = self.create_subscription(
            PointCloud2,
            # '/raw_thermal_pointcloud',
            '/selected_surface_points',
            self.pointcloud_callback,
            10
        )

        # # get map of IK-valid poses in the pointcloud
        # self.point_pose_map: dict[tuple, Pose|None] = {}
        # self.point_mask: np.ndarray = None
        # self.pointcloud_setup = self.create_subscription(
        #     PointCloud2,
        #     '/raw_thermal_pointcloud',
        #     self.posemap_callback,
        #     # self.pointcloud_callback,
        #     10,
        #     callback_group=MutuallyExclusiveCallbackGroup()
        # )


        self.band_low_C = 40.0
        self.band_high_C = 50.0
        self.pub_percent_in_band = self.create_publisher(Float64, '/selected_surface_percent_in_band', 10)
        self.band_log_counter = 0

        self.target_percent = 90.0

        self.heating_start_time = None
        self.reached_target = False
        self.time_to_target = None

    def get_tool_offset_pose(
        self, 
        original_pose: Pose, 
        z_offset: float,
        frame_id: str,
        verbose: bool = False
    ) -> Pose:
        """
        Gets the robot pose offset along the inverted surface normal,
        flipping the tool 180 deg around X and offsetting along the tool Z.
        """
        quat = [
            original_pose.orientation.x,
            original_pose.orientation.y,
            original_pose.orientation.z,
            original_pose.orientation.w
        ]

        rot_orig = R.from_quat(quat)
        flip_x = R.from_euler('x', 180, degrees=True)
        rot_flipped = rot_orig * flip_x

        # Compute offset vector along inverted surface normal (tool's new Z axis)
        offset_vec = -z_offset * rot_flipped.apply([0, 0, 1])

        offset_position = np.array([
            original_pose.position.x,
            original_pose.position.y,
            original_pose.position.z
        ]) + offset_vec

        new_quat = rot_flipped.as_quat()

        if verbose:
            # self.get_logger().info(f"Original position: {original_pose.position}")
            loginfo(self,f"Original position: {original_pose.position}")
            # self.get_logger().info(f"Offset vec: {offset_vec}")
            loginfo(self,f"Offset vec: {offset_vec}")
            # self.get_logger().info(f"New position: {offset_position}")
            loginfo(self,f"New position: {offset_position}")
            # self.get_logger().info(f"New orientation: {new_quat}")
            loginfo(self,f"New orientation: {new_quat}")

        target_pose = PoseStamped()
        target_pose.header.frame_id = frame_id
        target_pose.header.stamp = self.get_clock().now().to_msg()
        target_pose.pose.position = Point(
            x = offset_position[0], y = offset_position[1], z = offset_position[2]
        )
        target_pose.pose.orientation = Quaternion(
            x = new_quat[0], y = new_quat[1], z = new_quat[2], w = new_quat[3],
        )

        # transform CAD local frame pose to world frame pose
        for _ in range(10):
            try:
                target_pose = self.tf_buffer.transform(target_pose, "world", timeout=rclpy.duration.Duration(seconds=0.1))
                return target_pose.pose
            except Exception as e:
                # self.get_logger().warn(f"TF transform failed: {e}")
                logwarn(self,f"TF transform failed: {e}")
                continue

        return None

    def posemap_callback(self, msg: PointCloud2):
        """Offline computation of a map of points in the pointcloud with valid
        IK configurations. Points with invalid configurations are included
        in a no-go mask."""
        start_time = self.get_clock().now()

        # get the pointcloud normals
        points = point_cloud2.read_points(msg, field_names=("x", "y", "z", "thermal"), skip_nans=True)
        self.get_logger().info(f"points shape {points.shape}")
        points = np.array([[*list(point)[:3]] for point in points], dtype=np.float64)

        # Open3D point cloud + normal estimation
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(50)
        pcd.normalize_normals()

        # Always flip if Z is negative — works for horizontal and slanted surfaces
        normals = np.asarray(pcd.normals, dtype=np.float64) 
        normals[normals[:, 2] < 0] *= -1

        # Normalize and get quaternions from normals
        normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
        qs = normal_to_quaternion_batched(normals)

        # determine if each point's IK is reachable
        num_fail_success = [0,0]
        import time
        total_pts = 0
        for p, q in zip(points, qs):
            total_pts += 1
            # self.get_logger().info(f"point: {total_pts}")
            target_pose = self.get_tool_offset_pose(
                Pose(
                    position=Point(x=p[0],y=p[1],z=p[2]),
                    orientation=Quaternion(x=q[0],y=q[1],z=q[2],w=q[3])
                ),
                self.z_offset,
                msg.header.frame_id
            )
            ik_check = get_best_ik(
                self, target_pose, self.joint_state_topic_,
                self.move_group_name_, self.base_,
                self.ik_client_, attempts=1
            )
            if ik_check is None:
                self.point_pose_map[tuple(p)] = None
                num_fail_success[0] += 1
            else:
                self.point_pose_map[tuple(p)] = target_pose
                num_fail_success[1] += 1

        self.destroy_subscription(self.pointcloud_setup)
        self.get_logger().info(f"Valid poses: {num_fail_success[1]}, Invalid poses: {num_fail_success[0]}")
        self.get_logger().info(f"Duration: {(self.get_clock().now()-start_time).seconds_nanoseconds()[0]} seconds")
        self.get_logger().info("Pointcloud-pose mapping complete.")
        
        valid_keys = {k for k, v in self.point_pose_map.items() if v is not None}
        self.point_mask = np.array([tuple(p) in valid_keys for p in points])
        self.pose_map_ready = True
        self.get_logger().info('Raw thermal extrema node with normal estimation running.')

    def pointcloud_callback(self, msg: PointCloud2):
        """Greedy Coldest Node analysis policy."""

        if self.heating_start_time is None:
            self.heating_start_time = time.time()
            self.get_logger().info("Started timing heating process")

        if not self.pose_map_ready:
            return

        points = []
        temperatures = []

        # try:
        for p in point_cloud2.read_points(msg, field_names=("x", "y", "z", "thermal"), skip_nans=True):
            x, y, z, temp = p
            points.append([x, y, z])
            temperatures.append(ktoc(temp))
            # temperatures.append(temp)
        # except AssertionError:
        #     logwarn(self,"Pointcloud published with wrong fields.")
        #     return

        if not points:
            # self.get_logger().warn("Empty raw thermal point cloud.")
            logwarn(self,"Empty raw thermal point cloud.")
            return

        points = np.array(points)
        temperatures = np.array(temperatures)
        # self.get_logger().info(f"points shape {points.shape}")
        loginfo(self,f"points shape {points.shape}")

        # Open3D point cloud + normal estimation
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.04, max_nn=30))

        # pcd.orient_normals_consistent_tangent_plane(50)
        # pcd.normalize_normals()

        normals = np.asarray(pcd.normals)

        # # Always flip if Z is negative — works for horizontal and slanted surfaces
        normals[normals[:, 2] < 0] *= -1
        pcd.normals = o3d.utility.Vector3dVector(normals)

        # Temperature smoothing
        tree = cKDTree(points)
        smoothed_temp = np.zeros(len(points))
        for i, pt in enumerate(points):
            idxs = tree.query_ball_point(pt, r=0.01)
            smoothed_temp[i] = np.mean(temperatures[idxs]) if idxs else temperatures[i]

        # Percent of selected_surface_points within target band
        in_band_mask = (
            (smoothed_temp >= self.band_low_C) &
            (smoothed_temp <= self.band_high_C)
        )
        percent_in_band = 100.0 * np.mean(in_band_mask)

        if (
            not self.reached_target and
            percent_in_band >= self.target_percent
        ):
            self.reached_target = True
            self.time_to_target = time.time() - self.heating_start_time

            self.get_logger().info(
                f"Reached {self.target_percent:.1f}% in band "
                f"in {self.time_to_target:.2f} seconds"
            )

        # Publish for experimentation
        band_msg = Float64()
        band_msg.data = float(percent_in_band)
        self.pub_percent_in_band.publish(band_msg)

        if self.heating_start_time is not None:
            elapsed = time.time() - self.heating_start_time

        # Print periodically
        self.band_log_counter += 1
        if self.band_log_counter % 10 == 0 and self.heating_start_time is not None:
            self.get_logger().info(
                f"Selected surface in-band: {percent_in_band:.1f}% "
                f"({np.count_nonzero(in_band_mask)}/{len(smoothed_temp)}) "
                f"for band [{self.band_low_C:.1f}, {self.band_high_C:.1f}] C [{elapsed:.2f}]"
            )

        # # store data for future plotting
        # self.average_temps.append(np.mean(temperatures))
        # error = np.square(temperatures - self.desired_temperature)
        # self.mean_error_from_desired.append(np.mean(error))
        # self.std_dev_error_from_desired.append(np.std(error))
        # np.savez('/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/temp_data_33C.npz',
        # # np.savez('/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/temp_data_29C.npz',
        # # np.savez('/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/temp_data_40C.npz',
        #          avg_temps = np.array(self.average_temps), mean_error = np.array(self.mean_error_from_desired),
        #          std_dev_error = np.array(self.std_dev_error_from_desired), desired_temp = np.array([self.desired_temperature])
        # )

        # use point mask to filter out invalid points and temperatures
        # points[~self.point_mask] = np.nan
        # smoothed_temp[~self.point_mask] = np.nan

        # find hottest and coldest nodes
        hottest_idx = np.nanargmax(smoothed_temp)
        coldest_idx = np.nanargmin(smoothed_temp)   # coldest point
        # coldest_idx = np.random.randint(0, smoothed_temp.shape[0])  # random point

        hottest_point = points[hottest_idx]
        coldest_point = points[coldest_idx]
        hottest_normal = normals[hottest_idx]
        coldest_normal = normals[coldest_idx]

        # # Debug coldest
        # self.get_logger().info(f"coldest_point={coldest_point}")
        # if self.latest_point_pub is not None:
        #     self.get_logger().info(f"latest_point={self.latest_point_pub}")

        # def ema(prev, new, alpha):
        #     return new if prev is None else alpha * new + (1 - alpha) * prev

        # self.ema_hottest_point = ema(self.ema_hottest_point, hottest_point, self.ema_alpha)
        # self.ema_coldest_point = ema(self.ema_coldest_point, coldest_point, self.ema_alpha)
        # self.ema_hottest_normal = safe_normalize(
        #     ema(self.ema_hottest_normal, hottest_normal, self.ema_alpha)
        # )
        # self.ema_coldest_normal = safe_normalize(
        #     ema(self.ema_coldest_normal, coldest_normal, self.ema_alpha)
        # )

        self.ema_hottest_point = hottest_point
        self.ema_coldest_point = coldest_point
        self.ema_hottest_normal = safe_normalize(hottest_normal)
        self.ema_coldest_normal = safe_normalize(coldest_normal)
        # self.get_logger().info(f"Coldest normal final: {self.ema_coldest_normal}, Z = {self.ema_coldest_normal[2]:.3f}, temp = {temperatures[coldest_idx]:.1f}C")
        loginfo(self,f"Coldest normal final: {self.ema_coldest_normal}, Z = {self.ema_coldest_normal[2]:.3f}, temp = {temperatures[coldest_idx]:.1f}C")

        do_cold = True
        # do_hot = True
    

        current_time = time.perf_counter()
        # Don't publish if the coldest point is within certain spherical radius of last point pubished
        if self.latest_point_pub is not None:
            # if within_radius(coldest_point, self.latest_point_pub, radius=0.03):
            #     # self.get_logger().info("Within radius. Not publishing.")
            #     do_cold = False
            # # elif abs(self.latest_temp_pub - temperatures[coldest_idx]) < 0.5:
            # #     do_cold = False
            # elif current_time - self.latest_pub_time < 1.0:
            #     do_cold = False
            if current_time - self.latest_pub_time < 1.0:
                do_cold = False

        # Publish coldest on condition
        if do_cold:
            # self.get_logger().info("Publishing new coldest pose...")
            self.latest_pub_time = time.perf_counter()
            self.latest_point_pub = coldest_point
            # self.latest_temp_pub = temperatures[coldest_idx]
            self.publish_pose(self.greedy_pub, self.ema_coldest_point, self.ema_coldest_normal, 
                msg.header.frame_id, temperatures[coldest_idx], self.pub_debug_cold, self.cur_coldest_pose
            )
        # For visualization only
        # self.get_logger().info(str(temperatures[coldest_idx]))
        self.latest_temp_pub = temperatures[coldest_idx]
        self.publish_marker(self.ema_coldest_point, msg.header.frame_id, 1, (0.0, 0.0, 1.0))
        self.publish_text_marker(f"{temperatures[coldest_idx]:.1f}C", coldest_point, msg.header.frame_id, 11, (0.8, 0.8, 1.0))
        
        # Always publish to pose topics
        self.publish_pose(self.pub_coldest, self.ema_coldest_point, self.ema_coldest_normal, 
                msg.header.frame_id, temperatures[coldest_idx], self.pub_debug_cold, self.cur_coldest_pose
            )

        self.publish_pose(self.pub_hottest, self.ema_hottest_point, self.ema_hottest_normal, 
            msg.header.frame_id, temperatures[hottest_idx], self.pub_debug_hot, self.cur_hottest_pose
        )
        self.publish_marker(self.ema_hottest_point, msg.header.frame_id, 0, (1.0, 0.0, 0.0))
        self.publish_text_marker(f"{temperatures[hottest_idx]:.1f}C", hottest_point, msg.header.frame_id, 10, (1.0, 0.8, 0.8))



    def normal_to_quaternion(self, normal):
        z_axis = np.array(normal, dtype=np.float64)
        z_axis /= np.linalg.norm(z_axis)
        x_axis = np.array([1.0, 0.0, 0.0])
        if np.allclose(z_axis, x_axis):
            x_axis = np.array([0.0, 1.0, 0.0])
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)
        rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
        return R.from_matrix(rot_matrix).as_quat()

    def normal_with_input_yaw(self, normal, input_quat):
        # Normalize input normal (desired Z direction)
        z_axis = np.array(normal, dtype=np.float64)
        z_axis /= np.linalg.norm(z_axis)

        # Extract yaw from input quaternion
        input_r = R.from_quat(input_quat)
        yaw = input_r.as_euler('zyx')[0]  # extract yaw (Z rotation)

        # Construct a yaw-only rotation around global Z
        yaw_rot = R.from_euler('z', yaw)
        x_ref = yaw_rot.apply([1, 0, 0])  # rotated X axis

        # Create orthogonal frame with Z = normal, X = x_ref projected onto plane normal to Z
        x_axis = x_ref - np.dot(x_ref, z_axis) * z_axis  # remove component along Z
        if np.linalg.norm(x_axis) < 1e-6:
            # If projection is too small (e.g., x_ref is parallel to z_axis), pick arbitrary X
            x_axis = np.array([1, 0, 0]) if abs(z_axis[0]) < 0.9 else np.array([0, 1, 0])
            x_axis -= np.dot(x_axis, z_axis) * z_axis  # make orthogonal to Z
        x_axis /= np.linalg.norm(x_axis)

        y_axis = np.cross(z_axis, x_axis)

        # Build rotation matrix and convert to quaternion
        rot_matrix = np.column_stack((x_axis, y_axis, z_axis))
        return R.from_matrix(rot_matrix).as_quat()

    def force_z_up_preserve_xy(self, input_quat, target_z=np.array([0, 0, 1])):
        # Convert input quaternion to rotation matrix
        R_input = R.from_quat(input_quat)
        x_input = R_input.apply([1, 0, 0])
        y_input = R_input.apply([0, 1, 0])
        # Normalize Z
        z_axis = np.array(target_z, dtype=np.float64)
        z_axis /= np.linalg.norm(z_axis)

        # Project original X onto plane orthogonal to Z
        x_axis = x_input - np.dot(x_input, z_axis) * z_axis
        if np.linalg.norm(x_axis) < 1e-6:
            # If too close to parallel, use projected Y instead
            x_axis = y_input - np.dot(y_input, z_axis) * z_axis

        x_axis /= np.linalg.norm(x_axis)

        # Compute Y to complete orthonormal basis
        y_axis = np.cross(z_axis, x_axis)

        # Reconstruct rotation matrix
        R_new = np.column_stack((x_axis, y_axis, z_axis))
        return R.from_matrix(R_new).as_quat()

    def publish_pose(self, pub, point, normal, frame_id, temperature, debug_pub, cur_pose: CurrentPose) -> Pose:
        cur_time = self.get_clock().now().to_msg()
        extrema = Extrema()
        pose = Pose()
        poses = []
        extrema.header.frame_id = frame_id
        extrema.header.stamp = cur_time
        pose.position.x, pose.position.y, pose.position.z = map(float, point)

        # Normalize and get quaternion from normal
        normal = normal / np.linalg.norm(normal)
        q = self.normal_to_quaternion(normal)

        # Build a vector message for transforming
        normal_vec = Vector3Stamped()
        normal_vec.header.frame_id = frame_id
        normal_vec.header.stamp = self.get_clock().now().to_msg()
        normal_vec.vector.x = normal[0]
        normal_vec.vector.y = normal[1]
        normal_vec.vector.z = normal[2]

        try:
            # Transform normal to world frame
            world_vec = self.tf_buffer.transform(normal_vec, "world", timeout=rclpy.duration.Duration(seconds=0.1))
            world_z = np.array([world_vec.vector.x, world_vec.vector.y, world_vec.vector.z], dtype=np.float64)
            world_z /= np.linalg.norm(world_z)

            angle_to_up = np.arccos(np.clip(np.dot(world_z, [0, 0, 1]), -1.0, 1.0))
            angle_deg = np.degrees(angle_to_up)

            if angle_deg > 30.0:
                # self.get_logger().warn(f"Pose is {angle_deg:.1f}° from up — forcing up-facing pose.")
                logwarn(self,f"Pose is {angle_deg:.1f}° from up — forcing up-facing pose.")
                q = self.normal_to_quaternion([0, 0, 1])  # Force up

        except Exception as e:
            # self.get_logger().warn(f"TF transform failed: {e}")
            logwarn(self,f"TF transform failed: {e}")

        pose.orientation.x = q[0]
        pose.orientation.y = q[1]
        pose.orientation.z = q[2]
        pose.orientation.w = q[3]


        # confirm validity of pose
        check_pose = PoseStamped()
        check_pose.header = normal_vec.header
        check_pose.pose = pose
        try:
            # Transform normal to world frame
            world_pose = self.tf_buffer.transform(check_pose, "world", timeout=rclpy.duration.Duration(seconds=0.1))
            # self.get_logger().info(f"check pose {check_pose}")
            loginfo(self,f"check pose {check_pose}")
            # self.get_logger().info(f"World pose pos {world_pose.pose.position} ort {world_pose.pose.orientation}")
            loginfo(self,f"World pose pos {world_pose.pose.position} ort {world_pose.pose.orientation}")

            q = world_pose.pose.orientation
            rot = R.from_quat([q.x, q.y, q.z, q.w])

            # Step 2: Get the Z-axis (the third column of the rotation matrix)
            z_axis = rot.as_matrix()[:, 2]

            # Step 3: Check angle to global Z axis
            cos_angle = np.dot(z_axis, [0, 0, 1])
            angle_rad = np.arccos(np.clip(cos_angle, -1.0, 1.0))
            angle_deg = np.degrees(angle_rad)

            # Step 4: Reject or accept based on threshold (e.g. 30 degrees)
            if angle_deg > 50.0 or world_pose.pose.position.z < 0.04:
                # self.get_logger().warn(f"Adjustment needed pose: Z axis is {angle_deg:.1f}° from vertical, position at {world_pose.pose.position}.")
                logwarn(self,f"Adjustment needed pose: Z axis is {angle_deg:.1f}° from vertical, position at {world_pose.pose.position}.")
                # if cur_pose.best_val is not None: 
                #     pose = cur_pose.best_val.pose
                # else:
                #     return
                # q = self.normal_to_quaternion([0, 0, 1])  # Force up
                q = self.force_z_up_preserve_xy(
                    np.array([
                        world_pose.pose.orientation.x,
                        world_pose.pose.orientation.y,
                        world_pose.pose.orientation.z,
                        world_pose.pose.orientation.w,
                    ]),
                )   # Force up but keep original Yaw intact
                world_pose.pose.orientation.x = q[0]
                world_pose.pose.orientation.y = q[1]
                world_pose.pose.orientation.z = q[2]
                world_pose.pose.orientation.w = q[3]
                pose = self.tf_buffer.transform(world_pose, frame_id, timeout=rclpy.duration.Duration(seconds=0.1)).pose

            else:
                # self.get_logger().info(f"Accepted pose: Z axis is {angle_deg:.1f}° from vertical, position at {world_pose.pose.position}.")
                loginfo(self,f"Accepted pose: Z axis is {angle_deg:.1f}° from vertical, position at {world_pose.pose.position}.")
                # cur_pose.best_val.pose = pose

        except Exception as e:
            # self.get_logger().warn(f"TF transform failed: {e}")
            logwarn(self,f"TF transform failed: {e}")
            # if cur_pose.best_val is not None: 
            #     pose = cur_pose.best_val.pose
            # else:
            #     return
            return

        # Rotate 180 about Z
        q_rot_z_180 = R.from_euler('y', 180, degrees=True).as_quat()
        q_current = [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
        q_combined = R.from_quat(q_rot_z_180) * R.from_quat(q_current)
        q_combined = q_combined.as_quat()
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = q_combined

        # assemble and publish temperature message
        poses.append(pose)
        extrema.poses = poses
        extrema.value = float(temperature)
        pub.publish(extrema)

        # debug publish
        debug_pose = PoseStamped()
        debug_pose.header = extrema.header
        debug_pose.pose = pose
        self.latest_coldest_pub = pose # store the latest coldest pose published
        debug_pub.publish(debug_pose)

    def publish_marker(self, point, frame_id, marker_id, color):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "heatmap"
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = map(float, point)
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.025
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        marker.lifetime.sec = 1
        self.marker_pub.publish(marker)

    def publish_text_marker(self, text, position, frame_id, marker_id, color):
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "labels"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = float(position[0])
        marker.pose.position.y = float(position[1])
        marker.pose.position.z = float(position[2]) + 0.06
        marker.pose.orientation.w = 1.0
        marker.scale.z = 0.05
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0
        marker.text = text
        marker.lifetime.sec = 1
        self.marker_pub.publish(marker)


def main(args=None):
    rclpy.init(args=args)
    node = RawThermalExtremaPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from sensor_msgs.msg import JointState
from sensor_msgs_py import point_cloud2
import numpy as np

import rosbag2_py
from rclpy.serialization import serialize_message

import os
from ament_index_python.packages import get_package_share_directory
import shutil

# verbosity setting
# verbose = False
verbose = True
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

class ThermalBagWriter(Node):
    """Node for writing thermal pointcloud and robot pose data
    to a RosBag."""

    def __init__(self):
        super().__init__('thermal_data_rosbag_writer')
        self.z_offset = 0.35

        # desired temperatures
        self.desired_temperature = 60.0 # Celsius, = 140F
        self.room_temperature = 25.5 # Celsius, ~= 79F

        # callbakc to set up 
        # self.selection_points: np.ndarray | None = None
        self.selection_points = None
        self.selected_surface_sub = self.create_subscription(
            PointCloud2,
            '/selected_surface_points',
            self.selected_surface_mask_callback,
            10
        )

        # make sure previous ros bag is deleted before running
        # (assuming that relevant data has been copied elsewhere)
        pkg_path = os.path.join('/'.join(get_package_share_directory("thermal_camera").split('/')[:-4]+['thermal_data_bag']))
        if os.path.exists(pkg_path) and os.path.isdir(pkg_path):
            logwarn(self, "Deleting previous rosbag instance!")
            shutil.rmtree(pkg_path)

        # initialize and open rosbag writer
        self.writer = rosbag2_py.SequentialWriter()
        storage_options = rosbag2_py.StorageOptions(
            uri='thermal_data_bag',
            storage_id='sqlite3')
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format='cdr',
            output_serialization_format='cdr',
        )
        self.writer.open(storage_options, converter_options)

        # set up writing topics and callbacks ->
        #   thermal pointcloud of selected surface points
        #   thermal pointcloud of non-selected points
        #   robot FK pose, with tool offset
        self.thermal_topic_str = '/raw_thermal_pointcloud'
        self.select_thermal_topic_str = '/raw_thermal_selected'
        self.rest_thermal_topic_str = '/raw_thermal_rest'
        self.robot_pose_topic_str = '/robot_target_pose'
        self.waypoints_topic_str = '/debug_cartesian_waypoints'
        self.joint_states_topic_str = '/joint_states'
        self.full_thermal_topic_str = '/raw_thermal_pointcloud'
        self.selected_surface_topic_str = '/selected_surface_points'
        
        self.writer.create_topic(rosbag2_py.TopicMetadata(
            name=self.select_thermal_topic_str,
            type='sensor_msgs/msg/PointCloud2',
            serialization_format='cdr'
        ))
        self.writer.create_topic(rosbag2_py.TopicMetadata(
            name=self.rest_thermal_topic_str,
            type='sensor_msgs/msg/PointCloud2',
            serialization_format='cdr'
        ))
        self.writer.create_topic(rosbag2_py.TopicMetadata(
            name=self.robot_pose_topic_str,
            type='geometry_msgs/msg/PoseStamped',
            serialization_format='cdr'
        ))
        self.writer.create_topic(rosbag2_py.TopicMetadata(
            name=self.waypoints_topic_str,
            type='visualization_msgs/msg/MarkerArray',
            serialization_format='cdr'
        ))
        self.writer.create_topic(rosbag2_py.TopicMetadata(
            name=self.joint_states_topic_str,
            type='sensor_msgs/msg/JointState',
            serialization_format='cdr'
        ))

        self.writer.create_topic(rosbag2_py.TopicMetadata(
            name=self.full_thermal_topic_str,
            type='sensor_msgs/msg/PointCloud2',
            serialization_format='cdr'
        ))
        self.writer.create_topic(rosbag2_py.TopicMetadata(
            name=self.selected_surface_topic_str,
            type='sensor_msgs/msg/PointCloud2',
            serialization_format='cdr'
        ))
        
        
        self.thermal_sub = self.create_subscription(
            PointCloud2,
            self.thermal_topic_str,
            self.thermal_topic_callback,
            10)
        self.thermal_sub

        self.robot_pose_sub = self.create_subscription(
            PoseStamped,
            self.robot_pose_topic_str,
            self.robot_topic_callback,
            10)
        self.robot_pose_sub

        self.debug_waypoints_sub = self.create_subscription(
            MarkerArray,
            self.waypoints_topic_str,
            self.waypoints_callback,
            10
        )

        self.joint_states_sub = self.create_subscription(
            JointState,
            self.joint_states_topic_str,
            self.joint_states_callback,
            10
        )

        self._last_js_write_ns = 0
        self._js_write_period_ns = int(1e9 / 100)  # keep ~100 Hz


    # def selected_surface_mask_callback(self, msg: PointCloud2):
    #     """Creates mask using points in the selected surface."""
    #     if self.selection_points is not None:
    #         return

    #     self.selection_points = np.array(
    #         list(point_cloud2.read_points(msg, field_names=("x", "y", "z", "thermal"), skip_nans=True))
    #     )
    #     loginfo(self, f"Selection received: {self.selection_points.shape[0]} points")
    def selected_surface_mask_callback(self, msg: PointCloud2):
        """Store that the selected surface was received and write it once to bag."""
        if self.selection_points is not None:
            return

        self.selection_points = True
        n_pts = sum(1 for _ in point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z", "thermal"),
            skip_nans=True
        ))
        loginfo(self, f"Selection received: {n_pts} points")

        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.writer.write(
            self.selected_surface_topic_str,
            serialize_message(msg),
            stamp_ns
        )

    def robot_topic_callback(self, msg: PoseStamped):
        """Writes robot EEF offset poses to the bag."""
        self.writer.write(
            self.robot_pose_topic_str,
            serialize_message(msg),
            msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        )

    def waypoints_callback(self, msg: MarkerArray):
        """Writes waypoints from motion planner to the bag."""
        stamp_ns = max(m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec for m in msg.markers)
        self.writer.write(
            self.waypoints_topic_str,
            serialize_message(msg),
            stamp_ns
        )

    def joint_states_callback(self, msg: JointState):
        now_ns = (msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec) \
                if (msg.header.stamp.sec or msg.header.stamp.nanosec) \
                else (self.get_clock().now().nanoseconds)
        if now_ns - self._last_js_write_ns < self._js_write_period_ns:
            return
        self._last_js_write_ns = now_ns
        self.writer.write(self.joint_states_topic_str, serialize_message(msg), now_ns)

    # def thermal_topic_callback(self, msg: PointCloud2):
    #     """Writes the pointcloud temperatures to a ROS bag. Splits
    #     into one topic for the selected surface and another for
    #     the rest of the pointcloud."""

    #     if self.selection_points is None:
    #         logwarn(self, "Selectable surface has not been received.")
    #         return

    #     # -------------------------------------------------------------
    #     # NEW SECTION: Transform incoming pointcloud → base_link frame
    #     # -------------------------------------------------------------
    #     try:
    #         # Make sure TF buffer is initialized once
    #         if not hasattr(self, "tf_buffer"):
    #             import tf2_ros
    #             self.tf_buffer = tf2_ros.Buffer()
    #             self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    #         # Lookup transform from cloud frame → base_link
    #         transform = self.tf_buffer.lookup_transform(
    #             "base_link", msg.header.frame_id, rclpy.time.Time())

    #         # Convert transform to 4×4 matrix
    #         import numpy as np
    #         from scipy.spatial.transform import Rotation as R

    #         t = transform.transform.translation
    #         q = transform.transform.rotation
    #         T = np.eye(4, dtype=np.float32)
    #         T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    #         T[:3, 3] = [t.x, t.y, t.z]

    #         # Read original pointcloud data
    #         all_pts = np.array(
    #             list(point_cloud2.read_points(msg, field_names=("x", "y", "z", "thermal"), skip_nans=True))
    #         )

    #         # Apply transform to (x, y, z)
    #         pts_h = np.hstack([all_pts[:, :3], np.ones((all_pts.shape[0], 1))])
    #         all_pts[:, :3] = (T @ pts_h.T).T[:, :3]

    #         # (Optionally) update the header frame to base_link
    #         msg.header.frame_id = "base_link"

    #     except Exception as e:
    #         logwarn(self, f"TF transform failed ({msg.header.frame_id} → base_link): {e}")
    #         all_pts = np.array(
    #             list(point_cloud2.read_points(msg, field_names=("x", "y", "z", "thermal"), skip_nans=True))
    #         )
    #     # -------------------------------------------------------------
    #     # END NEW SECTION
    #     # -------------------------------------------------------------

    #     # Apply selection mask to get two new pointcloud messages  
    #     selection_xyz = np.stack(   # shape (M, 3)
    #         [self.selection_points['x'], self.selection_points['y'], self.selection_points['z']],
    #         axis=-1,
    #         dtype=np.float32
    #     )
    #     all_xyz = np.stack(
    #         [all_pts['x'], all_pts['y'], all_pts['z']],
    #         axis=-1,
    #         dtype=np.float32
    #     )

    #     selection_struct = selection_xyz.view([('', np.float32)] * 3)
    #     full_struct = all_xyz.view([('', np.float32)] * 3)

    #     in_mask = np.isin(full_struct, selection_struct).flatten()

    #     selected_points = all_pts[in_mask]
    #     other_points = all_pts[~in_mask]

    #     # Reconstruct clouds with full data (x, y, z, thermal)
    #     fields = [
    #         PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    #         PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    #         PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    #         PointField(name='thermal', offset=12, datatype=PointField.FLOAT32, count=1),
    #     ]
    #     cloud_selected = point_cloud2.create_cloud(msg.header, fields, selected_points)
    #     cloud_other = point_cloud2.create_cloud(msg.header, fields, other_points)

    #     # write pointclouds to bag
    #     self.writer.write(
    #         self.select_thermal_topic_str,
    #         serialize_message(cloud_selected),
    #         msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
    #     )
    #     self.writer.write(
    #         self.rest_thermal_topic_str,
    #         serialize_message(cloud_other),
    #         msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
    #     )


    # def thermal_topic_callback(self, msg: PointCloud2):
    #     """Writes the pointcloud temperatures to a ROS bag. Splits
    #     into one topic for the selected surface and another for
    #     the rest of the pointcloud."""
    #     if self.selection_points is None:
    #         logwarn(self, "Selectable surface has not been received.")
    #         return

    #     all_pts = np.array(
    #         list(point_cloud2.read_points(msg, field_names=("x", "y", "z", "thermal"), skip_nans=True))
    #     )

    #     # apply selection mask to get two new pointcloud messages  
    #     selection_xyz = np.stack(   # shape (M, 3)
    #         [self.selection_points['x'], self.selection_points['y'], self.selection_points['z']],
    #         axis=-1,
    #         dtype=np.float32
    #     )
    #     all_xyz = np.stack(
    #         [all_pts['x'], all_pts['y'], all_pts['z']],
    #         axis=-1,
    #         dtype=np.float32
    #     )

    #     selection_struct = selection_xyz.view([('', np.float32)] * 3)
    #     full_struct = all_xyz.view([('', np.float32)] * 3)

    #     in_mask = np.isin(full_struct, selection_struct).flatten()

    #     selected_points = all_pts[in_mask]
    #     other_points = all_pts[~in_mask]

    #     # Reconstruct clouds with full data (x, y, z, thermal)
    #     fields = [
    #         PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    #         PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    #         PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    #         PointField(name='thermal', offset=12, datatype=PointField.FLOAT32, count=1),
    #     ]
    #     cloud_selected = point_cloud2.create_cloud(msg.header, fields, selected_points)
    #     cloud_other = point_cloud2.create_cloud(msg.header, fields, other_points)

    #     # write pointclouds to bag
    #     self.writer.write(
    #         self.select_thermal_topic_str,
    #         serialize_message(cloud_selected),
    #         msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
    #     )
    #     self.writer.write(
    #         self.rest_thermal_topic_str,
    #         serialize_message(cloud_other),
    #         msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
    #     )
    def thermal_topic_callback(self, msg: PointCloud2):
        """Write the full thermal pointcloud to the bag."""
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.writer.write(
            self.full_thermal_topic_str,
            serialize_message(msg),
            stamp_ns
        )

def main(args=None):
    rclpy.init(args=args)
    node = ThermalBagWriter()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

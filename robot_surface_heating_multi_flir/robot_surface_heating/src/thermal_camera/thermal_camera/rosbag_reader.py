#!/usr/bin/env python3

import os
from enum import Enum

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2


class StorageIndex(Enum):
    THERMAL = 0
    ROBOT = 1
    SELECTION = 2


# verbosity setting
verbose = True


def loginfo(node: Node, msg: str):
    node.get_logger().info(msg)


def logwarn(node: Node, msg: str):
    node.get_logger().warn(msg)


def logerr(node: Node, msg: str):
    node.get_logger().error(msg)


if not verbose:
    loginfo = lambda *args: None
    logwarn = lambda *args: None
    logerr = lambda *args: None


class ThermalBagReader(Node):
    """Read thermal pointcloud and robot pose data from bag playback topics
    and save them to an NPZ file."""

    def __init__(self):
        super().__init__('thermal_data_reader')

        self.z_offset = 0.35

        # desired temperatures
        self.desired_temperature = 60.0  # Celsius
        self.room_temperature = 25.5     # Celsius

        # destination file
        self.save_file_path = os.path.join(
            '/'.join(
                get_package_share_directory("thermal_camera").split('/')[:-4]
                + ['src', 'thermal_camera']
            ),
            'coldest_node_data',
            # '03_06_26_composite_real.npz'
            '04_03_26_mpc_test_1.npz'
        )

        self.get_logger().info(f"Saving data to {self.save_file_path}")

        # stored data
        self.thermal_pointclouds: list[np.ndarray] = []
        self.robot_poses: list[np.ndarray] = []
        self.selection_points: np.ndarray | None = None

        self.timestamps = {
            StorageIndex.THERMAL: [],
            StorageIndex.ROBOT: [],
            StorageIndex.SELECTION: [],
        }

        self.frames = {
            "thermal_frame_id": None,
            "robot_frame_id": None,
            "selection_frame_id": None,
        }

        # topic strings matching revised writer
        self.thermal_topic_str = '/raw_thermal_pointcloud'
        self.selected_surface_topic_str = '/selected_surface_points'
        self.robot_pose_topic_str = '/robot_target_pose'

        # subscribers
        self.thermal_sub = self.create_subscription(
            PointCloud2,
            self.thermal_topic_str,
            self.thermal_callback,
            10
        )

        self.selected_surface_sub = self.create_subscription(
            PointCloud2,
            self.selected_surface_topic_str,
            self.selected_surface_callback,
            10
        )

        self.robot_sub = self.create_subscription(
            PoseStamped,
            self.robot_pose_topic_str,
            self.robot_pose_callback,
            10
        )

    @staticmethod
    def stamp_to_ns(msg) -> int:
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    @staticmethod
    def pointcloud2_to_xyzt(msg: PointCloud2) -> np.ndarray:
        pts = np.array(
            [
                [p[0], p[1], p[2], p[3]]
                for p in point_cloud2.read_points(
                    msg,
                    field_names=("x", "y", "z", "thermal"),
                    skip_nans=True
                )
            ],
            dtype=np.float64
        )
        return pts

    def thermal_callback(self, msg: PointCloud2):
        pts = self.pointcloud2_to_xyzt(msg)
        self.thermal_pointclouds.append(pts)
        self.timestamps[StorageIndex.THERMAL].append(self.stamp_to_ns(msg))

        if self.frames["thermal_frame_id"] is None:
            self.frames["thermal_frame_id"] = msg.header.frame_id
            loginfo(self, f"Thermal frame: {msg.header.frame_id}")

        if len(self.thermal_pointclouds) % 50 == 0:
            loginfo(self, f"Stored {len(self.thermal_pointclouds)} thermal clouds")

    def selected_surface_callback(self, msg: PointCloud2):
        if self.selection_points is not None:
            return

        self.selection_points = self.pointcloud2_to_xyzt(msg)
        self.timestamps[StorageIndex.SELECTION].append(self.stamp_to_ns(msg))

        if self.frames["selection_frame_id"] is None:
            self.frames["selection_frame_id"] = msg.header.frame_id

        loginfo(
            self,
            f"Stored selected surface with {self.selection_points.shape[0]} points "
            f"in frame {msg.header.frame_id}"
        )

    def robot_pose_callback(self, msg: PoseStamped):
        pose = msg.pose
        self.robot_poses.append(
            np.array(
                [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ],
                dtype=np.float64
            )
        )
        self.timestamps[StorageIndex.ROBOT].append(self.stamp_to_ns(msg))

        if self.frames["robot_frame_id"] is None:
            self.frames["robot_frame_id"] = msg.header.frame_id
            loginfo(self, f"Robot pose frame: {msg.header.frame_id}")

    def save_data(self):
        """Save stored data to NPZ file."""
        thermal_count = len(self.thermal_pointclouds)
        robot_count = len(self.robot_poses)
        selection_count = 0 if self.selection_points is None else self.selection_points.shape[0]

        loginfo(
            self,
            f"Saving: thermal={thermal_count}, robot={robot_count}, "
            f"selection_points={selection_count}"
        )

        os.makedirs(os.path.dirname(self.save_file_path), exist_ok=True)

        np.savez(
            self.save_file_path,
            thermal_pointclouds=np.array(self.thermal_pointclouds, dtype=object),
            selected_surface_points=(
                self.selection_points
                if self.selection_points is not None
                else np.empty((0, 4), dtype=np.float64)
            ),
            robot_poses=np.array(self.robot_poses, dtype=np.float64),
            thermal_timestamps_ns=np.array(
                self.timestamps[StorageIndex.THERMAL],
                dtype=np.int64
            ),
            robot_timestamps_ns=np.array(
                self.timestamps[StorageIndex.ROBOT],
                dtype=np.int64
            ),
            selection_timestamps_ns=np.array(
                self.timestamps[StorageIndex.SELECTION],
                dtype=np.int64
            ),
            thermal_frame_id=np.array(
                self.frames["thermal_frame_id"] or "",
                dtype='<U256'
            ),
            robot_frame_id=np.array(
                self.frames["robot_frame_id"] or "",
                dtype='<U256'
            ),
            selection_frame_id=np.array(
                self.frames["selection_frame_id"] or "",
                dtype='<U256'
            ),
        )

        self.get_logger().info(f"Wrote data to {self.save_file_path}")


def main(args=None):
    rclpy.init(args=args)
    node = ThermalBagReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.save_data()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

# #!/usr/bin/env python3

# import rclpy
# from rclpy.node import Node

# from sensor_msgs.msg import PointCloud2
# from geometry_msgs.msg import PoseStamped
# from sensor_msgs_py import point_cloud2
# import numpy as np

# import os
# from ament_index_python.packages import get_package_share_directory

# from enum import Enum
# class StorageIndex(Enum):
#     SELECTED=0
#     REST=1
#     ROBOT=2

# # verbosity setting
# # verbose = False
# verbose = True
# def loginfo(node: Node, msg: str):
#     node.get_logger().info(msg)
# def logwarn(node: Node, msg: str):
#     node.get_logger().warn(msg)
# def logerr(node: Node, msg: str):
#     node.get_logger().error(msg)

# if not verbose:
#     loginfo = lambda *args: None  # noqa
#     logwarn = lambda *args: None  # noqa
#     logerr = lambda *args: None  # noqa

# class ThermalBagReader(Node):
#     """Node for reading thermal pointcloud and robot pose data
#     from a topic/rosbag and writing it to a local storage file."""

#     def __init__(self):
#         super().__init__('thermal_data_reader')
#         self.z_offset = 0.35

#         # desired temperatures
#         self.desired_temperature = 60.0 # Celsius, = 140F
#         # self.desired_temperature = 37.78 # Celsius, = 100F
#         self.room_temperature = 25.5 # Celsius, ~= 79F

#         # destination file
#         self.save_file_path = os.path.join(
#             '/'.join(get_package_share_directory("thermal_camera").split('/')[:-4]+['src','thermal_camera']),
#             'coldest_node_data',
#             # 'flat_surface_60C_bag_data.npz'
#             # 'slanted_surface_60C_bag_data.npz'
#             # 'steep_surface_60C_bag_data.npz'
#             # 'full_surface_60C_bag_data.npz'
#             # 'robot_segmentation_test_2.npz'
#             'code_optimize_test_1.npz'
#         )

#         self.get_logger().info(f"Saving data to {self.save_file_path}")

#         # storage of recorded data
#         self.select_thermal = []
#         self.rest_thermal = []
#         self.robot_poses = []
#         self.timestamps = [[],[],[]]

#         # callbacks to set up 
#         self.select_thermal_topic_str = '/raw_thermal_selected'
#         self.rest_thermal_topic_str = '/raw_thermal_rest'
#         self.robot_pose_topic_str = '/robot_target_pose'

#         self.selected_sub = self.create_subscription(
#             PointCloud2,
#             '/raw_thermal_selected',
#             self.selected_callback,
#             10
#         )
#         self.rest_sub = self.create_subscription(
#             PointCloud2,
#             '/raw_thermal_rest',
#             self.rest_callback,
#             10
#         )
#         self.robot_sub = self.create_subscription(
#             PoseStamped,
#             '/robot_target_pose',
#             self.robot_pose_callback,
#             10
#         )

#     def selected_callback(self, msg: PointCloud2):
#         """Stores selected portion of pointcloud as Mx4 array:
#             ```
#             [
#                 [X,Y,Z,<raw thermal value>], 
#                 ...
#             ]
#             ```
#         """
#         pts = np.array(
#             list(point_cloud2.read_points(msg, field_names=("x", "y", "z", "thermal"), skip_nans=True))
#         )

#         self.select_thermal.append(np.stack(   # shape (M, 4)
#             [pts['x'], pts['y'], pts['z'], pts['thermal']],
#             axis=-1,
#             dtype=np.float64
#         ))
#         self.timestamps[StorageIndex.SELECTED.value].append(
#             msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
#         )

#     def rest_callback(self, msg: PointCloud2):
#         """Stores non-selected portion of pointcloud as Mx4 array:
#             ```
#             [
#                 [X,Y,Z,<raw thermal value>], 
#                 ...
#             ]
#             ```
#         """
#         pts = np.array(
#             list(point_cloud2.read_points(msg, field_names=("x", "y", "z", "thermal"), skip_nans=True))
#         )

#         self.rest_thermal.append(np.stack(   # shape (M, 4)
#             [pts['x'], pts['y'], pts['z'], pts['thermal']],
#             axis=-1,
#             dtype=np.float64
#         ))
#         self.timestamps[StorageIndex.REST.value].append(
#             msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
#         )

#     def robot_pose_callback(self, msg: PoseStamped):
#         """Stores robot EEF pose as 1x7 array:
#             ```
#             [X,Y,Z,qX,qY,qZ,qW]
#             ```
#         """
#         pose = msg.pose
#         self.robot_poses.append(np.array(
#             [
#                 pose.position.x, pose.position.y, pose.position.z,
#                 pose.orientation.x, pose.orientation.y, pose.orientation.z,
#                 pose.orientation.w
#             ]
#         ))
#         self.timestamps[StorageIndex.ROBOT.value].append(
#             msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
#         )

#     def save_data(self):
#         """Saves stored data to NPZ file (runs just before shutdown)."""
#         max_arr_len = np.max([len(self.select_thermal), len(self.rest_thermal), len(self.robot_poses)])
#         for idx in range(len(StorageIndex)):
#             if len(self.timestamps[idx]) < max_arr_len:
#                 self.timestamps[idx] += [np.nan]*(max_arr_len-len(self.timestamps[idx]))
#         print(np.array(self.timestamps).shape)
#         np.savez(self.save_file_path,
#             selected_points = np.array(self.select_thermal),
#             other_points = np.array(self.rest_thermal),
#             robot_poses = np.array(self.robot_poses),
#             timestamps = np.array(self.timestamps)
#         )

#         self.get_logger().info(f"Wrote data to {self.save_file_path}")

# def main(args=None):
#     rclpy.init(args=args)
#     node = ThermalBagReader()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.save_data()
#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()

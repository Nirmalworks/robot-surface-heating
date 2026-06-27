#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from sensor_msgs.msg import PointCloud2, JointState
from geometry_msgs.msg import PoseStamped
from visualization_msgs.msg import MarkerArray
from sensor_msgs_py import point_cloud2
import numpy as np

import os
from ament_index_python.packages import get_package_share_directory
from enum import Enum
from datetime import datetime

# MoveIt FK service / messages
from moveit_msgs.srv import GetPositionFK
from moveit_msgs.msg import RobotState, MoveItErrorCodes


class StorageIndex(Enum):
    SELECTED  = 0
    REST      = 1
    ROBOT     = 2
    WAYPOINTS = 3
    JOINTS    = 4


class ThermalBagReader(Node):
    """Read thermal pointclouds, robot pose, waypoint markers, and joint states
    from topics/rosbag, compute EEF FK from joint states (asynchronously), and save to NPZ.
    """

    def __init__(self):
        super().__init__('thermal_data_reader')

        # ---- Config ----
        self.base_frame = "base_link"   # set to your robot's base frame
        # self.eef_link   = "butane_torch"       # set to your EEF link
        self.eef_link   = "heat_gun"   

        # Destination file (datestamped)
        out_name = datetime.now().strftime("%m_%d_%y_traj_test_tc=tmp.npz")
        self.save_file_path = os.path.join(
            '/'.join(get_package_share_directory("thermal_camera").split('/')[:-4] + ['src', 'thermal_camera']),
            'coldest_node_data',
            out_name
        )
        self.get_logger().info(f"Saving data to {self.save_file_path}")

        # ---- Storage ----
        self.select_thermal = []   # list of (Mi,4) arrays [x y z thermal]
        self.rest_thermal   = []   # list of (Ni,4) arrays [x y z thermal]
        self.robot_poses    = []   # list of (7,) [x y z qx qy qz qw]
        self.waypoint_sets  = []   # list of (Ki,7)
        self.waypoint_ids   = []   # list of (Ki,)
        self.joint_names      = []  # list[list[str]]
        self.joint_positions  = []  # list[np.ndarray]
        self.joint_velocities = []  # list[np.ndarray]
        self.joint_efforts    = []  # list[np.ndarray]
        # FK results (same length/order as completed FK calls, not necessarily all joints)
        self.eef_poses      = []   # list of (7,) arrays
        self.eef_timestamps = []   # list of ns
        self.timestamps = [[], [], [], [], []]  # SELECTED, REST, ROBOT, WAYPOINTS, JOINTS

        # Index & state for asynchronous FK processing
        self._fk_idx = 0
        self._fk_inflight = False
        self._fk_enabled = True

        # ---- Topic names ----
        self.select_thermal_topic_str = '/raw_thermal_selected'
        self.rest_thermal_topic_str   = '/raw_thermal_rest'
        self.robot_pose_topic_str     = '/robot_target_pose'
        self.waypoints_topic_str      = '/debug_cartesian_waypoints'
        self.joint_states_topic_str   = '/joint_states'

        # ---- Subscriptions ----
        self.selected_sub = self.create_subscription(
            PointCloud2, self.select_thermal_topic_str, self.selected_callback, 10
        )
        self.rest_sub = self.create_subscription(
            PointCloud2, self.rest_thermal_topic_str, self.rest_callback, 10
        )
        self.robot_sub = self.create_subscription(
            PoseStamped, self.robot_pose_topic_str, self.robot_pose_callback, 10
        )
        self.waypoints_sub = self.create_subscription(
            MarkerArray, self.waypoints_topic_str, self.waypoints_callback, 10
        )
        self.joint_states_sub = self.create_subscription(
            JointState, self.joint_states_topic_str, self.joint_states_callback, 10
        )

        # ---- MoveIt FK client ----
        self.fk_client = self.create_client(GetPositionFK, '/compute_fk')
        self.get_logger().info("Waiting for MoveIt FK service '/compute_fk' ...")
        if not self.fk_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().warn("FK service not available; proceeding without FK.")
            self._fk_enabled = False
        else:
            self.get_logger().info("FK service available")

        # Timer to trickle through FK requests asynchronously (non-blocking)
        # 50 Hz tick is fine; we only send a new request if none is in flight.
        self.create_timer(0.02, self._fk_tick)

    # ---------- Helpers ----------

    def pc2_to_xyzt_array(self, msg: PointCloud2) -> np.ndarray:
        """Return an (N,4) float64 array [x,y,z,thermal] from a PointCloud2."""
        pts_list = list(point_cloud2.read_points(
            msg, field_names=("x", "y", "z", "thermal"), skip_nans=True
        ))
        if not pts_list:
            return np.empty((0, 4), dtype=np.float64)

        arr = np.array(pts_list)  # may be (N,4) plain or structured (N,) with named fields
        if getattr(arr.dtype, "names", None):   # structured case
            out = np.stack([arr['x'], arr['y'], arr['z'], arr['thermal']], axis=-1)
        else:                                   # plain list-of-tuples -> (N,4)
            out = np.asarray(arr, dtype=np.float64).reshape(-1, 4)

        return out.astype(np.float64, copy=False)

    # ---------- Callbacks ----------

    def selected_callback(self, msg: PointCloud2):
        pts = self.pc2_to_xyzt_array(msg)
        self.select_thermal.append(pts)
        self.timestamps[StorageIndex.SELECTED.value].append(
            msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        )

    def rest_callback(self, msg: PointCloud2):
        pts = self.pc2_to_xyzt_array(msg)
        self.rest_thermal.append(pts)
        self.timestamps[StorageIndex.REST.value].append(
            msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        )

    def robot_pose_callback(self, msg: PoseStamped):
        p = msg.pose
        self.robot_poses.append(np.array(
            [p.position.x, p.position.y, p.position.z,
             p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w],
            dtype=np.float64
        ))
        self.timestamps[StorageIndex.ROBOT.value].append(
            msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        )

    def waypoints_callback(self, msg: MarkerArray):
        poses = []
        ids = []
        stamp_ns_candidates = []
        for m in msg.markers:
            pp = m.pose
            poses.append([
                pp.position.x, pp.position.y, pp.position.z,
                pp.orientation.x, pp.orientation.y, pp.orientation.z, pp.orientation.w
            ])
            ids.append(m.id)
            stamp_ns_candidates.append(
                m.header.stamp.sec * 1_000_000_000 + m.header.stamp.nanosec
            )
        self.waypoint_sets.append(np.array(poses, dtype=np.float64))
        self.waypoint_ids.append(np.array(ids, dtype=np.int32))
        stamp_ns = int(np.max(stamp_ns_candidates)) if stamp_ns_candidates else self.get_clock().now().nanoseconds
        self.timestamps[StorageIndex.WAYPOINTS.value].append(stamp_ns)

    def joint_states_callback(self, msg: JointState):
        self.joint_names.append(list(msg.name))
        self.joint_positions.append(np.array(msg.position, dtype=np.float64))
        self.joint_velocities.append(
            np.array(msg.velocity, dtype=np.float64) if len(msg.velocity) > 0 else np.array([], dtype=np.float64)
        )
        self.joint_efforts.append(
            np.array(msg.effort, dtype=np.float64) if len(msg.effort) > 0 else np.array([], dtype=np.float64)
        )
        self.timestamps[StorageIndex.JOINTS.value].append(
            msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        )

    # ---------- Asynchronous FK while running ----------

    def _fk_tick(self):
        """Send the next FK request if enabled and none is in flight."""
        if not self._fk_enabled or self._fk_inflight:
            return

        # If we've already requested FK for all currently buffered joint states, nothing to do.
        if self._fk_idx >= len(self.joint_names):
            return

        names = self.joint_names[self._fk_idx]
        positions = self.joint_positions[self._fk_idx]
        t_ns = self.timestamps[StorageIndex.JOINTS.value][self._fk_idx]

        rs = RobotState()
        rs.joint_state.name = list(names)
        rs.joint_state.position = [float(x) for x in positions]

        req = GetPositionFK.Request()
        req.header.frame_id = self.base_frame
        req.header.stamp = self.get_clock().now().to_msg()
        req.fk_link_names = [self.eef_link]
        req.robot_state = rs

        self._fk_inflight = True
        future = self.fk_client.call_async(req)
        future.add_done_callback(lambda fut, t_ns=t_ns: self._fk_done(fut, t_ns))

    def _fk_done(self, future, t_ns: int):
        """Store FK result and advance the index. Runs inside the executor thread."""
        try:
            res = future.result()
        except Exception as e:
            # Service exception; just skip this sample
            self.get_logger().warn(f"FK future exception: {e}")
            res = None

        if res and res.error_code.val == MoveItErrorCodes.SUCCESS and res.pose_stamped:
            pose = res.pose_stamped[0].pose
            self.eef_poses.append(np.array([
                pose.position.x, pose.position.y, pose.position.z,
                pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w
            ], dtype=np.float64))
            self.eef_timestamps.append(t_ns)
        else:
            # Optionally log at debug level to avoid spam
            # self.get_logger().debug("FK failed or empty pose; skipping")
            pass

        self._fk_idx += 1
        self._fk_inflight = False

    # ---------- Save ----------

    def save_data(self):
        """Save all streams to NPZ (whatever FK is done so far)."""
        # Determine max stream length to pad timestamps table
        stream_lengths = [
            len(self.select_thermal),
            len(self.rest_thermal),
            len(self.robot_poses),
            len(self.waypoint_sets),
            len(self.joint_names),
        ]
        max_arr_len = int(np.max(stream_lengths)) if stream_lengths else 0

        # Pad timestamps per-stream so we can form a rectangular 2D array
        for idx in range(len(StorageIndex)):
            if len(self.timestamps[idx]) < max_arr_len:
                self.timestamps[idx] += [np.nan] * (max_arr_len - len(self.timestamps[idx]))

        # Write NPZ (use object arrays for variable-length frames)
        np.savez(self.save_file_path,
            selected_points = np.array(self.select_thermal, dtype=object),  # each: (Mi,4)
            other_points    = np.array(self.rest_thermal,   dtype=object),  # each: (Ni,4)
            robot_poses     = np.array(self.robot_poses,    dtype=np.float64),  # (R,7)
            waypoint_sets   = np.array(self.waypoint_sets,  dtype=object),  # each: (Ki,7)
            waypoint_ids    = np.array(self.waypoint_ids,   dtype=object),  # each: (Ki,)
            joint_names      = np.array(self.joint_names,      dtype=object),
            joint_positions  = np.array(self.joint_positions,  dtype=object),
            joint_velocities = np.array(self.joint_velocities, dtype=object),
            joint_efforts    = np.array(self.joint_efforts,    dtype=object),
            eef_poses      = np.array(self.eef_poses,      dtype=np.float64),  # (K,7) for completed FK only
            eef_timestamps = np.array(self.eef_timestamps, dtype=np.float64),  # (K,)
            timestamps = np.array(self.timestamps, dtype=np.float64),          # (5, max_len)
        )

        self.get_logger().info(f"Wrote data to {self.save_file_path}")


def main(args=None):
    rclpy.init(args=args)
    node = ThermalBagReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_data()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

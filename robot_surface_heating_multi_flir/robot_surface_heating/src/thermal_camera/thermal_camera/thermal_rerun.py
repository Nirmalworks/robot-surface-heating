#!/usr/bin/env python3
"""
thermal_rerun.py

Reads playback topics from the latest rosbag_writer output, transforms full
thermal point clouds into a common frame, computes EEF poses from URDF FK
using joint states, and saves the same training-ready NPZ fields as before.

Timing behavior:
- Assumes frame timing may be non-uniform
- Saves timestamps using the exact thermal frame message header times
- Interpolates joint states to those exact thermal frame times
- Computes EEF poses at those exact thermal frame times

Saved NPZ fields:
  - thermal_frames:             (T,) object array; each entry is (Ni,4) [x,y,z,thermal] in target_frame
  - eef_pos_quat:               (T,7) float32 array; [x,y,z,qx,qy,qz,qw] in target_frame
  - joint_positions:            (T,) object array; each entry is (n_joints,) interpolated joints
  - joint_names:                (n_joints,) object array
  - timestamps:                 (T,) float64 seconds, exact thermal-frame timestamps
  - dt_target:                  float64 seconds, retained for compatibility
  - target_frame:               str

Original selected-region fields, preserved for compatibility:
  - selected_surface_points:    (Ns,4) float32 array; first selected ROI only
  - selected_surface_xyz:       (Ns,3) float32 array; first selected ROI only
  - selected_surface_time_ns:   int64 timestamp for first selected ROI

Additional multi-ROI fields:
  - selected_surface_points_all:      (R,) object array; each entry is (Ni,4)
  - selected_surface_xyz_all:         (R,) object array; each entry is (Ni,3)
  - selected_surface_time_ns_all:     (R,) int64 array
  - selected_surface_timestamps_all:  (R,) float64 array, seconds
  - selected_surface_roi_id_all:      (R,) int32 array
  - selected_surface_roi_index_all:   (R,) int32 array, from /multi_roi_status if available

Additional fields for optimizer-debug reconstruction:
  - planner_debug_json:         (P,) object array of JSON strings, one per plan
  - planner_debug_timestamps:   (P,) float64 array of planner timestamps in seconds
  - planner_debug_plan_index:   (P,) int32 array of plan indices
"""

import json
import os

from ament_index_python.packages import get_package_share_directory
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import String
import tf2_ros
from scipy.spatial.transform import Rotation as R
from urdfpy import URDF


URDF_PATH = "/tmp/my_robot_cell.urdf"
MESH_ROOT = (
    "/home/cam/robot_surface_heating_dev/"
    "robot_surface_heating_multi_flir/"
    "robot_surface_heating/"
    "src/Universal_Robots_ROS2_Description"
)

TARGET_FRAME = "base_link"
EEF_LINK = "heat_gun"

# NPZ_SAVE_NAME = "04_28_26_mpc_ramp_5.npz"
# NPZ_SAVE_NAME = "05_05_26_mpc_1.npz"
# NPZ_SAVE_NAME = "05_05_26_honeycomb_large_3.npz"
# NPZ_SAVE_NAME = "05_07_26_local_heating_2.npz"
# NPZ_SAVE_NAME = "05_13_26_ian_sample_test_1.npz"
NPZ_SAVE_NAME = "05_21_26_honey_comb_local_mpc_1.npz"

# Topic published by o3d_visual_surface when multi-ROI mode is enabled.
ROI_STATUS_TOPIC = "/multi_roi_status"

# ROI-change detection fallback. This is used even if ROI_STATUS_TOPIC is absent.
ROI_CENTER_CHANGE_M = 0.01
ROI_SIZE_CHANGE_FRAC = 0.05
ROI_SIGNATURE_MIN_INTERVAL_NS = 200_000_000


class ThermalBagReader(Node):
    def __init__(self):
        super().__init__("thermal_bag_reader")

        self.target_frame = TARGET_FRAME
        self.eef_link_name = EEF_LINK
        self.npz_save_name = NPZ_SAVE_NAME

        self.get_logger().info(f"Saving npz file name: {self.npz_save_name}")

        # Kept only for compatibility with downstream code that expects this field.
        self.dt_target = 0.111

        self.output_dir = os.path.join(
            os.path.expanduser("~"),
            "thermal_npz_datasets",
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.get_logger().info(
            f"[Reader] target_frame={self.target_frame}, dt_target={self.dt_target:.3f}s"
        )
        self.get_logger().info(
            f"[Reader] URDF={URDF_PATH}, eef_link={self.eef_link_name}"
        )

        try:
            self.robot = URDF.load(URDF_PATH)
            link_names = {lnk.name for lnk in self.robot.links}
            if self.eef_link_name not in link_names:
                self.get_logger().warn(
                    f"EEF link '{self.eef_link_name}' not found in URDF; FK may fail."
                )
        except Exception as e:
            self.get_logger().error(f"Failed to load URDF '{URDF_PATH}': {e}")
            raise

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Full thermal frames
        self.thermal_frames = []
        self.thermal_times_ns = []

        # Original static selected region fields.
        # These are preserved exactly for old downstream code.
        self.selected_surface_points = None
        self.selected_surface_time_ns = None

        # Multi-ROI selected region history.
        self.selected_surface_points_all = []
        self.selected_surface_time_ns_all = []
        self.selected_surface_roi_id_all = []
        self.selected_surface_roi_index_all = []

        self.last_selected_signature = None
        self.last_selected_store_time_ns = None
        self.latest_roi_status = {}
        self.latest_roi_status_time_ns = None
        self.last_status_roi_index = None

        # Joint stream
        self.joint_times_ns = []
        self.joint_names = None
        self.joint_series = []

        # Planner debug stream
        self.planner_debug_json = []
        self.planner_debug_timestamps = []
        self.planner_debug_plan_index = []

        self.pc_topic = "/raw_thermal_pointcloud"
        self.selection_topic = "/selected_surface_points"
        self.js_topic = "/joint_states"
        self.planner_debug_topic = "/planner_debug_json"
        self.roi_status_topic = ROI_STATUS_TOPIC

        self.create_subscription(PointCloud2, self.pc_topic, self.pc_cb, 10)
        self.create_subscription(PointCloud2, self.selection_topic, self.selection_cb, 10)
        self.create_subscription(JointState, self.js_topic, self.js_cb, 50)
        self.create_subscription(String, self.planner_debug_topic, self.planner_debug_cb, 50)
        self.create_subscription(String, self.roi_status_topic, self.roi_status_cb, 50)

        self.get_logger().info(
            "Listening:\n"
            f"  - {self.pc_topic}\n"
            f"  - {self.selection_topic}\n"
            f"  - {self.js_topic}\n"
            f"  - {self.planner_debug_topic}\n"
            f"  - {self.roi_status_topic}"
        )

    @staticmethod
    def stamp_to_ns(msg) -> int:
        return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

    @staticmethod
    def _ns_to_s(ns_array):
        return np.asarray(ns_array, dtype=np.float64) * 1e-9

    @staticmethod
    def pointcloud2_to_xyzt(msg: PointCloud2) -> np.ndarray:
        pts = np.array(
            [
                [p[0], p[1], p[2], p[3]]
                for p in point_cloud2.read_points(
                    msg,
                    field_names=("x", "y", "z", "thermal"),
                    skip_nans=True,
                )
            ],
            dtype=np.float32,
        )
        return pts

    @staticmethod
    def _roi_signature(arr: np.ndarray) -> dict:
        xyz = arr[:, :3]
        return {
            "count": int(arr.shape[0]),
            "center": np.mean(xyz, axis=0).astype(np.float64),
            "mins": np.min(xyz, axis=0).astype(np.float64),
            "maxs": np.max(xyz, axis=0).astype(np.float64),
        }

    def _latest_status_roi_index(self) -> int:
        value = self.latest_roi_status.get("active_roi_index", -1)
        try:
            return int(value)
        except Exception:
            return -1

    def _should_store_roi_record(self, arr: np.ndarray, stamp_ns: int) -> bool:
        if len(self.selected_surface_points_all) == 0:
            return True

        current_signature = self._roi_signature(arr)
        previous_signature = self.last_selected_signature

        status_roi_index = self._latest_status_roi_index()
        if status_roi_index >= 0 and status_roi_index != self.last_status_roi_index:
            return True

        if previous_signature is None:
            return True

        previous_count = max(int(previous_signature["count"]), 1)
        current_count = int(current_signature["count"])
        size_frac = abs(current_count - previous_count) / previous_count

        previous_center = np.asarray(previous_signature["center"], dtype=np.float64)
        current_center = np.asarray(current_signature["center"], dtype=np.float64)
        center_dist = float(np.linalg.norm(current_center - previous_center))

        if size_frac > ROI_SIZE_CHANGE_FRAC:
            return True

        if center_dist > ROI_CENTER_CHANGE_M:
            return True

        if self.last_selected_store_time_ns is None:
            return False

        if stamp_ns - self.last_selected_store_time_ns < ROI_SIGNATURE_MIN_INTERVAL_NS:
            return False

        return False

    def _store_roi_record(self, arr: np.ndarray, stamp_ns: int):
        roi_id = len(self.selected_surface_points_all)
        status_roi_index = self._latest_status_roi_index()

        self.selected_surface_points_all.append(arr.astype(np.float32, copy=False))
        self.selected_surface_time_ns_all.append(int(stamp_ns))
        self.selected_surface_roi_id_all.append(int(roi_id))
        self.selected_surface_roi_index_all.append(int(status_roi_index))

        self.last_selected_signature = self._roi_signature(arr)
        self.last_selected_store_time_ns = int(stamp_ns)
        self.last_status_roi_index = int(status_roi_index)

        if status_roi_index >= 0:
            label = f"status_index={status_roi_index}"
        else:
            label = "status_index=unavailable"

        self.get_logger().info(
            f"[Reader] Stored selected ROI record {roi_id} with {arr.shape[0]} points "
            f"at t={stamp_ns * 1e-9:.3f}s ({label})"
        )

    def transform_points_to_target_frame(
        self,
        pts_xyzt: np.ndarray,
        src_frame: str,
        timeout_sec: float = 2.0,
    ) -> np.ndarray | None:
        if pts_xyzt.size == 0:
            return pts_xyzt

        if src_frame == self.target_frame:
            return pts_xyzt

        try:
            transform = self.tf_buffer.lookup_transform(
                self.target_frame,
                src_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout_sec),
            )

            t = transform.transform.translation
            q = transform.transform.rotation

            rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix().astype(np.float32)
            transform_matrix = np.eye(4, dtype=np.float32)
            transform_matrix[:3, :3] = rot
            transform_matrix[:3, 3] = [t.x, t.y, t.z]

            pts_h = np.hstack(
                [pts_xyzt[:, :3], np.ones((pts_xyzt.shape[0], 1), dtype=np.float32)]
            )

            out = pts_xyzt.copy()
            out[:, :3] = (transform_matrix @ pts_h.T).T[:, :3]
            return out

        except Exception as e:
            self.get_logger().warn(
                f"TF not ready yet ({src_frame} -> {self.target_frame}): {e}. "
                "Skipping this message."
            )
            return None

    def pc_cb(self, msg: PointCloud2):
        """Ingest full thermal cloud and transform into target_frame."""
        arr = self.pointcloud2_to_xyzt(msg)
        if arr.size == 0:
            return

        arr = self.transform_points_to_target_frame(arr, msg.header.frame_id)
        if arr is None:
            return

        self.thermal_frames.append(arr)
        self.thermal_times_ns.append(self.stamp_to_ns(msg))

    def selection_cb(self, msg: PointCloud2):
        """
        Store first selected surface exactly as before, and also store a compact
        history of selected ROI changes.
        """
        arr = self.pointcloud2_to_xyzt(msg)
        if arr.size == 0:
            self.get_logger().warn("[Reader] Received empty selected surface pointcloud")
            return

        arr = self.transform_points_to_target_frame(arr, msg.header.frame_id)
        if arr is None:
            return

        arr = arr.astype(np.float32, copy=False)
        stamp_ns = int(self.stamp_to_ns(msg))

        # Original behavior preserved: first selected ROI only.
        if self.selected_surface_points is None:
            self.selected_surface_points = arr
            self.selected_surface_time_ns = stamp_ns
            self.get_logger().info(
                f"[Reader] Stored selected surface with {arr.shape[0]} points"
            )

        # New behavior: save ROI changes without affecting old fields.
        if self._should_store_roi_record(arr, stamp_ns):
            self._store_roi_record(arr, stamp_ns)

    def roi_status_cb(self, msg: String):
        raw = msg.data
        if not raw:
            return

        try:
            status = json.loads(raw)
        except Exception:
            return

        if not isinstance(status, dict):
            return

        self.latest_roi_status = status
        self.latest_roi_status_time_ns = self.get_clock().now().nanoseconds

    def js_cb(self, msg: JointState):
        """Buffer joint vectors; initialize joint_names on first message."""
        if not msg.name or not msg.position:
            return

        if self.joint_names is None:
            self.joint_names = list(msg.name)
            self.get_logger().info(f"[Reader] joint_names = {self.joint_names}")

        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        vec = np.zeros(len(self.joint_names), dtype=np.float32)

        for i, joint_name in enumerate(self.joint_names):
            if joint_name in name_to_idx:
                src_idx = name_to_idx[joint_name]
                if src_idx < len(msg.position):
                    vec[i] = float(msg.position[src_idx])

        stamp_ns = self.stamp_to_ns(msg)
        self.joint_times_ns.append(stamp_ns)
        self.joint_series.append(vec)

    def planner_debug_cb(self, msg: String):
        """
        Store raw planner-debug JSON strings so the exact optimizer outputs can be
        reconstructed offline later. Also cache a few parsed fields for indexing.
        """
        raw = msg.data
        if not raw:
            return

        self.planner_debug_json.append(raw)

        try:
            parsed = json.loads(raw)
            stamp_sec = float(parsed.get("stamp_sec", np.nan))
            plan_index = int(parsed.get("plan_index", -1))
        except Exception as e:
            self.get_logger().warn(f"[Reader] Failed to parse planner debug JSON: {e}")
            stamp_sec = np.nan
            plan_index = -1

        self.planner_debug_timestamps.append(stamp_sec)
        self.planner_debug_plan_index.append(plan_index)

    def _build_exact_thermal_timeline(self):
        """
        Return exact thermal-frame timestamps and matching frames, clipped to the
        temporal overlap with the joint stream so joint interpolation is valid.
        """
        if len(self.thermal_times_ns) < 1 or len(self.joint_times_ns) < 2:
            raise RuntimeError("Insufficient data to build exact thermal timeline.")

        t_pc = self._ns_to_s(self.thermal_times_ns)
        t_js = self._ns_to_s(self.joint_times_ns)

        order_pc = np.argsort(t_pc)
        t_pc = t_pc[order_pc]
        frames_pc = [self.thermal_frames[i] for i in order_pc]

        t0 = float(t_js.min())
        t1 = float(t_js.max())
        keep = (t_pc >= t0) & (t_pc <= t1)

        if not np.any(keep):
            raise RuntimeError("No temporal overlap between thermal and joint streams.")

        t_exact = t_pc[keep].astype(np.float64)
        frames_exact = [frames_pc[i] for i in np.nonzero(keep)[0]]

        return t_exact, frames_exact

    def _interp_joints(self, t_query):
        """Linearly interpolate each joint independently to arbitrary query times."""
        t_js = self._ns_to_s(self.joint_times_ns)
        order = np.argsort(t_js)
        t_js = t_js[order]
        joint_matrix = np.stack([self.joint_series[i] for i in order], axis=0)

        n_joints = joint_matrix.shape[1]
        interp = np.zeros((len(t_query), n_joints), dtype=np.float32)

        for j in range(n_joints):
            interp[:, j] = np.interp(t_query, t_js, joint_matrix[:, j])

        return [interp[i] for i in range(interp.shape[0])]

    def _fk_eef(self, joint_vec):
        """Compute EEF pose [x,y,z,qx,qy,qz,qw] from URDF FK."""
        cfg = {n: float(v) for n, v in zip(self.joint_names, joint_vec)}

        try:
            tf_dict = self.robot.link_fk(cfg=cfg)
        except Exception as e:
            self.get_logger().warn(f"URDF FK failed: {e}")
            return np.zeros(7, dtype=np.float32)

        eef_transform = None
        for link, transform_matrix in tf_dict.items():
            if link.name == self.eef_link_name:
                eef_transform = np.asarray(transform_matrix, dtype=np.float64)
                break

        if eef_transform is None:
            self.get_logger().warn(
                f"EEF link '{self.eef_link_name}' not found in FK result; zeros returned."
            )
            return np.zeros(7, dtype=np.float32)

        pos = eef_transform[:3, 3].astype(np.float64)
        quat = R.from_matrix(eef_transform[:3, :3]).as_quat()
        return np.hstack([pos, quat]).astype(np.float32)

    def save_npz(self):
        if not self.thermal_frames:
            self.get_logger().error("No thermal point clouds recorded; nothing to save.")
            return

        if not self.joint_series:
            self.get_logger().error("No joint states recorded; nothing to save.")
            return

        if self.joint_names is None:
            self.get_logger().error("Joint names were never initialized; nothing to save.")
            return

        timestamps_exact, thermal_frames_exact = self._build_exact_thermal_timeline()
        joints_interp = self._interp_joints(timestamps_exact)
        eef_pos_quat = np.stack([self._fk_eef(jv) for jv in joints_interp], axis=0)

        npz_save_path = os.path.join(
            "/".join(
                get_package_share_directory("thermal_camera").split("/")[:-4]
                + ["src", "thermal_camera"]
            ),
            "coldest_node_data",
            NPZ_SAVE_NAME,
        )

        save_dict = {
            "thermal_frames": np.array(thermal_frames_exact, dtype=object),
            "eef_pos_quat": eef_pos_quat.astype(np.float32),
            "joint_positions": np.array(joints_interp, dtype=object),
            "joint_names": np.array(self.joint_names, dtype=object),
            "timestamps": timestamps_exact.astype(np.float64),
            "dt_target": float(self.dt_target),
            "target_frame": self.target_frame,
        }

        if self.selected_surface_points is not None:
            save_dict["selected_surface_points"] = self.selected_surface_points.astype(
                np.float32, copy=False
            )
            save_dict["selected_surface_xyz"] = self.selected_surface_points[:, :3].astype(
                np.float32, copy=False
            )
            save_dict["selected_surface_time_ns"] = np.int64(self.selected_surface_time_ns)
        else:
            self.get_logger().warn(
                "[Reader] No selected surface was received. Saving NPZ without selected-surface fields."
            )

        if len(self.selected_surface_points_all) > 0:
            roi_times_ns = np.asarray(self.selected_surface_time_ns_all, dtype=np.int64)

            save_dict["selected_surface_points_all"] = np.array(
                self.selected_surface_points_all,
                dtype=object,
            )
            save_dict["selected_surface_xyz_all"] = np.array(
                [
                    roi[:, :3].astype(np.float32, copy=False)
                    for roi in self.selected_surface_points_all
                ],
                dtype=object,
            )
            save_dict["selected_surface_time_ns_all"] = roi_times_ns
            save_dict["selected_surface_timestamps_all"] = roi_times_ns.astype(np.float64) * 1e-9
            save_dict["selected_surface_roi_id_all"] = np.asarray(
                self.selected_surface_roi_id_all,
                dtype=np.int32,
            )
            save_dict["selected_surface_roi_index_all"] = np.asarray(
                self.selected_surface_roi_index_all,
                dtype=np.int32,
            )

            self.get_logger().info(
                f"[Reader] Saving {len(self.selected_surface_points_all)} selected ROI records"
            )
        else:
            self.get_logger().warn(
                "[Reader] No selected ROI history records were stored."
            )

        if len(self.planner_debug_json) > 0:
            save_dict["planner_debug_json"] = np.array(self.planner_debug_json, dtype=object)
            save_dict["planner_debug_timestamps"] = np.asarray(
                self.planner_debug_timestamps, dtype=np.float64
            )
            save_dict["planner_debug_plan_index"] = np.asarray(
                self.planner_debug_plan_index, dtype=np.int32
            )
            self.get_logger().info(
                f"[Reader] Saving {len(self.planner_debug_json)} planner debug records"
            )
        else:
            self.get_logger().warn(
                "[Reader] No planner debug records were received. Saving NPZ without planner debug fields."
            )

        np.savez(npz_save_path, **save_dict)
        self.get_logger().info(f"[Reader] Saved NPZ: {npz_save_path}")

    def on_shutdown(self):
        try:
            self.save_npz()
        except Exception as e:
            self.get_logger().error(f"Failed to save NPZ: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = ThermalBagReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.save_npz()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


# #!/usr/bin/env python3
# """
# thermal_rerun.py

# Reads playback topics from the latest rosbag_writer output, transforms full
# thermal point clouds into a common frame, computes EEF poses from URDF FK
# using joint states, and saves the same training-ready NPZ fields as before.

# Timing behavior:
# - Assumes frame timing may be non-uniform
# - Saves timestamps using the exact thermal frame message header times
# - Interpolates joint states to those exact thermal frame times
# - Computes EEF poses at those exact thermal frame times

# Saved NPZ fields:
#   - thermal_frames:             (T,) object array; each entry is (Ni,4) [x,y,z,thermal] in target_frame
#   - eef_pos_quat:               (T,7) float32 array; [x,y,z,qx,qy,qz,qw] in target_frame
#   - joint_positions:            (T,) object array; each entry is (n_joints,) interpolated joints
#   - joint_names:                (n_joints,) object array
#   - timestamps:                 (T,) float64 seconds, exact thermal-frame timestamps
#   - dt_target:                  float64 seconds, retained for compatibility
#   - target_frame:               str

# Additional fields for selected region support:
#   - selected_surface_points:    (Ns,4) float32 array; [x,y,z,thermal] in target_frame
#   - selected_surface_xyz:       (Ns,3) float32 array
#   - selected_surface_time_ns:   int64 timestamp for selected_surface_points

# Additional fields for optimizer-debug reconstruction:
#   - planner_debug_json:         (P,) object array of JSON strings, one per plan
#   - planner_debug_timestamps:   (P,) float64 array of planner timestamps in seconds
#   - planner_debug_plan_index:   (P,) int32 array of plan indices
# """

# import os
# import json

# from ament_index_python.packages import get_package_share_directory
# import numpy as np
# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import PointCloud2, JointState
# from sensor_msgs_py import point_cloud2
# from std_msgs.msg import String
# import tf2_ros
# from scipy.spatial.transform import Rotation as R
# from urdfpy import URDF


# URDF_PATH = "/tmp/my_robot_cell.urdf"
# MESH_ROOT = (
#     "/home/cam/robot_surface_heating_dev/"
#     "robot_surface_heating_multi_flir/"
#     "robot_surface_heating/"
#     "src/Universal_Robots_ROS2_Description"
# )
# TARGET_FRAME = "base_link"
# EEF_LINK = "heat_gun"
# # NPZ_SAVE_NAME = "04_28_26_mpc_ramp_5.npz"
# # NPZ_SAVE_NAME = "05_05_26_mpc_1.npz"
# # NPZ_SAVE_NAME = "05_05_26_honeycomb_large_3.npz"
# NPZ_SAVE_NAME = "05_07_26_local_heating_1.npz"


# class ThermalBagReader(Node):
#     def __init__(self):
#         super().__init__("thermal_bag_reader")

#         self.target_frame = TARGET_FRAME
#         self.eef_link_name = EEF_LINK
#         self.npz_save_name = NPZ_SAVE_NAME

#         self.get_logger().info(f"Saving npz file name: {self.npz_save_name}")

#         # Kept only for compatibility with downstream code that expects this field.
#         self.dt_target = 0.111

#         self.output_dir = os.path.join(
#             os.path.expanduser("~"),
#             "thermal_npz_datasets",
#         )
#         os.makedirs(self.output_dir, exist_ok=True)

#         self.get_logger().info(
#             f"[Reader] target_frame={self.target_frame}, dt_target={self.dt_target:.3f}s"
#         )
#         self.get_logger().info(
#             f"[Reader] URDF={URDF_PATH}, eef_link={self.eef_link_name}"
#         )

#         try:
#             self.robot = URDF.load(URDF_PATH)
#             link_names = {lnk.name for lnk in self.robot.links}
#             if self.eef_link_name not in link_names:
#                 self.get_logger().warn(
#                     f"EEF link '{self.eef_link_name}' not found in URDF; FK may fail."
#                 )
#         except Exception as e:
#             self.get_logger().error(f"Failed to load URDF '{URDF_PATH}': {e}")
#             raise

#         self.tf_buffer = tf2_ros.Buffer()
#         self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

#         # Full thermal frames
#         self.thermal_frames = []
#         self.thermal_times_ns = []

#         # Static selected region
#         self.selected_surface_points = None
#         self.selected_surface_time_ns = None

#         # Joint stream
#         self.joint_times_ns = []
#         self.joint_names = None
#         self.joint_series = []

#         # Planner debug stream
#         self.planner_debug_json = []
#         self.planner_debug_timestamps = []
#         self.planner_debug_plan_index = []

#         self.pc_topic = "/raw_thermal_pointcloud"
#         self.selection_topic = "/selected_surface_points"
#         self.js_topic = "/joint_states"
#         self.planner_debug_topic = "/planner_debug_json"

#         self.create_subscription(PointCloud2, self.pc_topic, self.pc_cb, 10)
#         self.create_subscription(PointCloud2, self.selection_topic, self.selection_cb, 10)
#         self.create_subscription(JointState, self.js_topic, self.js_cb, 50)
#         self.create_subscription(String, self.planner_debug_topic, self.planner_debug_cb, 50)

#         self.get_logger().info(
#             "Listening:\n"
#             f"  - {self.pc_topic}\n"
#             f"  - {self.selection_topic}\n"
#             f"  - {self.js_topic}\n"
#             f"  - {self.planner_debug_topic}"
#         )

#     @staticmethod
#     def stamp_to_ns(msg) -> int:
#         return msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec

#     @staticmethod
#     def _ns_to_s(ns_array):
#         return np.asarray(ns_array, dtype=np.float64) * 1e-9

#     @staticmethod
#     def pointcloud2_to_xyzt(msg: PointCloud2) -> np.ndarray:
#         pts = np.array(
#             [
#                 [p[0], p[1], p[2], p[3]]
#                 for p in point_cloud2.read_points(
#                     msg,
#                     field_names=("x", "y", "z", "thermal"),
#                     skip_nans=True,
#                 )
#             ],
#             dtype=np.float32,
#         )
#         return pts

#     # def transform_points_to_target_frame(
#     #     self,
#     #     pts_xyzt: np.ndarray,
#     #     src_frame: str,
#     # ) -> np.ndarray:
#     #     if pts_xyzt.size == 0:
#     #         return pts_xyzt

#     #     if src_frame == self.target_frame:
#     #         return pts_xyzt

#     #     try:
#     #         transform = self.tf_buffer.lookup_transform(
#     #             self.target_frame,
#     #             src_frame,
#     #             rclpy.time.Time(),
#     #         )

#     #         t = transform.transform.translation
#     #         q = transform.transform.rotation

#     #         rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix().astype(np.float32)
#     #         T = np.eye(4, dtype=np.float32)
#     #         T[:3, :3] = rot
#     #         T[:3, 3] = [t.x, t.y, t.z]

#     #         pts_h = np.hstack(
#     #             [pts_xyzt[:, :3], np.ones((pts_xyzt.shape[0], 1), dtype=np.float32)]
#     #         )
#     #         pts_xyzt = pts_xyzt.copy()
#     #         pts_xyzt[:, :3] = (T @ pts_h.T).T[:, :3]
#     #         return pts_xyzt

#     #     except Exception as e:
#     #         self.get_logger().warn(
#     #             f"TF lookup failed ({src_frame} -> {self.target_frame}): {e}. "
#     #             "Using raw coordinates."
#     #         )
#     #         return pts_xyzt

#     def transform_points_to_target_frame(
#         self,
#         pts_xyzt: np.ndarray,
#         src_frame: str,
#         timeout_sec: float = 2.0,
#     ) -> np.ndarray | None:
#         if pts_xyzt.size == 0:
#             return pts_xyzt

#         if src_frame == self.target_frame:
#             return pts_xyzt

#         try:
#             transform = self.tf_buffer.lookup_transform(
#                 self.target_frame,
#                 src_frame,
#                 rclpy.time.Time(),
#                 timeout=rclpy.duration.Duration(seconds=timeout_sec),
#             )

#             t = transform.transform.translation
#             q = transform.transform.rotation

#             rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix().astype(np.float32)
#             T = np.eye(4, dtype=np.float32)
#             T[:3, :3] = rot
#             T[:3, 3] = [t.x, t.y, t.z]

#             pts_h = np.hstack(
#                 [pts_xyzt[:, :3], np.ones((pts_xyzt.shape[0], 1), dtype=np.float32)]
#             )

#             out = pts_xyzt.copy()
#             out[:, :3] = (T @ pts_h.T).T[:, :3]
#             return out

#         except Exception as e:
#             self.get_logger().warn(
#                 f"TF not ready yet ({src_frame} -> {self.target_frame}): {e}. "
#                 "Skipping this message."
#             )
#             return None

#     def pc_cb(self, msg: PointCloud2):
#         """Ingest full thermal cloud and transform into target_frame."""
#         arr = self.pointcloud2_to_xyzt(msg)
#         if arr.size == 0:
#             return

#         arr = self.transform_points_to_target_frame(arr, msg.header.frame_id)
#         if arr is None:
#             return

#         self.thermal_frames.append(arr)
#         self.thermal_times_ns.append(self.stamp_to_ns(msg))

#     def selection_cb(self, msg: PointCloud2):
#         """Store selected surface once, if present."""
#         if self.selected_surface_points is not None:
#             return

#         arr = self.pointcloud2_to_xyzt(msg)
#         if arr.size == 0:
#             self.get_logger().warn("[Reader] Received empty selected surface pointcloud")
#             return

#         arr = self.transform_points_to_target_frame(arr, msg.header.frame_id)
#         if arr is None:
#             return

#         self.selected_surface_points = arr.astype(np.float32, copy=False)
#         self.selected_surface_time_ns = int(self.stamp_to_ns(msg))

#         self.get_logger().info(
#             f"[Reader] Stored selected surface with {arr.shape[0]} points"
#         )

#     def js_cb(self, msg: JointState):
#         """Buffer joint vectors; initialize joint_names on first message."""
#         if not msg.name or not msg.position:
#             return

#         if self.joint_names is None:
#             self.joint_names = list(msg.name)
#             self.get_logger().info(f"[Reader] joint_names = {self.joint_names}")

#         name_to_idx = {n: i for i, n in enumerate(msg.name)}
#         vec = np.zeros(len(self.joint_names), dtype=np.float32)

#         for i, joint_name in enumerate(self.joint_names):
#             if joint_name in name_to_idx:
#                 src_idx = name_to_idx[joint_name]
#                 if src_idx < len(msg.position):
#                     vec[i] = float(msg.position[src_idx])

#         stamp_ns = self.stamp_to_ns(msg)
#         self.joint_times_ns.append(stamp_ns)
#         self.joint_series.append(vec)

#     def planner_debug_cb(self, msg: String):
#         """
#         Store raw planner-debug JSON strings so the exact optimizer outputs can be
#         reconstructed offline later. Also cache a few parsed fields for indexing.
#         """
#         raw = msg.data
#         if not raw:
#             return

#         self.planner_debug_json.append(raw)

#         try:
#             parsed = json.loads(raw)
#             stamp_sec = float(parsed.get("stamp_sec", np.nan))
#             plan_index = int(parsed.get("plan_index", -1))
#         except Exception as e:
#             self.get_logger().warn(f"[Reader] Failed to parse planner debug JSON: {e}")
#             stamp_sec = np.nan
#             plan_index = -1

#         self.planner_debug_timestamps.append(stamp_sec)
#         self.planner_debug_plan_index.append(plan_index)

#     def _build_exact_thermal_timeline(self):
#         """
#         Return exact thermal-frame timestamps and matching frames, clipped to the
#         temporal overlap with the joint stream so joint interpolation is valid.
#         """
#         if len(self.thermal_times_ns) < 1 or len(self.joint_times_ns) < 2:
#             raise RuntimeError("Insufficient data to build exact thermal timeline.")

#         t_pc = self._ns_to_s(self.thermal_times_ns)
#         t_js = self._ns_to_s(self.joint_times_ns)

#         order_pc = np.argsort(t_pc)
#         t_pc = t_pc[order_pc]
#         frames_pc = [self.thermal_frames[i] for i in order_pc]

#         t0 = float(t_js.min())
#         t1 = float(t_js.max())
#         keep = (t_pc >= t0) & (t_pc <= t1)

#         if not np.any(keep):
#             raise RuntimeError("No temporal overlap between thermal and joint streams.")

#         t_exact = t_pc[keep].astype(np.float64)
#         frames_exact = [frames_pc[i] for i in np.nonzero(keep)[0]]

#         return t_exact, frames_exact

#     def _interp_joints(self, t_query):
#         """Linearly interpolate each joint independently to arbitrary query times."""
#         t_js = self._ns_to_s(self.joint_times_ns)
#         order = np.argsort(t_js)
#         t_js = t_js[order]
#         J = np.stack([self.joint_series[i] for i in order], axis=0)

#         n_joints = J.shape[1]
#         interp = np.zeros((len(t_query), n_joints), dtype=np.float32)

#         for j in range(n_joints):
#             interp[:, j] = np.interp(t_query, t_js, J[:, j])

#         return [interp[i] for i in range(interp.shape[0])]

#     def _fk_eef(self, joint_vec):
#         """Compute EEF pose [x,y,z,qx,qy,qz,qw] from URDF FK."""
#         cfg = {n: float(v) for n, v in zip(self.joint_names, joint_vec)}

#         try:
#             tf_dict = self.robot.link_fk(cfg=cfg)
#         except Exception as e:
#             self.get_logger().warn(f"URDF FK failed: {e}")
#             return np.zeros(7, dtype=np.float32)

#         T_eef = None
#         for link, T in tf_dict.items():
#             if link.name == self.eef_link_name:
#                 T_eef = np.asarray(T, dtype=np.float64)
#                 break

#         if T_eef is None:
#             self.get_logger().warn(
#                 f"EEF link '{self.eef_link_name}' not found in FK result; zeros returned."
#             )
#             return np.zeros(7, dtype=np.float32)

#         pos = T_eef[:3, 3].astype(np.float64)
#         quat = R.from_matrix(T_eef[:3, :3]).as_quat()
#         return np.hstack([pos, quat]).astype(np.float32)

#     def save_npz(self):
#         if not self.thermal_frames:
#             self.get_logger().error("No thermal point clouds recorded; nothing to save.")
#             return

#         if not self.joint_series:
#             self.get_logger().error("No joint states recorded; nothing to save.")
#             return

#         if self.joint_names is None:
#             self.get_logger().error("Joint names were never initialized; nothing to save.")
#             return

#         timestamps_exact, thermal_frames_exact = self._build_exact_thermal_timeline()
#         joints_interp = self._interp_joints(timestamps_exact)
#         eef_pos_quat = np.stack([self._fk_eef(jv) for jv in joints_interp], axis=0)

#         npz_save_path = os.path.join(
#             "/".join(
#                 get_package_share_directory("thermal_camera").split("/")[:-4]
#                 + ["src", "thermal_camera"]
#             ),
#             "coldest_node_data",
#             # "04_24_26_greedy_test_4.npz",
#             # "04_27_26_mpc_test_2.npz",s
#             # "04_28_26_mpc_maintenance_10.npz",
#             NPZ_SAVE_NAME,
#         )

#         save_dict = {
#             "thermal_frames": np.array(thermal_frames_exact, dtype=object),
#             "eef_pos_quat": eef_pos_quat.astype(np.float32),
#             "joint_positions": np.array(joints_interp, dtype=object),
#             "joint_names": np.array(self.joint_names, dtype=object),
#             "timestamps": timestamps_exact.astype(np.float64),
#             "dt_target": float(self.dt_target),
#             "target_frame": self.target_frame,
#         }

#         if self.selected_surface_points is not None:
#             save_dict["selected_surface_points"] = self.selected_surface_points.astype(
#                 np.float32, copy=False
#             )
#             save_dict["selected_surface_xyz"] = self.selected_surface_points[:, :3].astype(
#                 np.float32, copy=False
#             )
#             save_dict["selected_surface_time_ns"] = np.int64(self.selected_surface_time_ns)
#         else:
#             self.get_logger().warn(
#                 "[Reader] No selected surface was received. Saving NPZ without selected-surface fields."
#             )

#         if len(self.planner_debug_json) > 0:
#             save_dict["planner_debug_json"] = np.array(self.planner_debug_json, dtype=object)
#             save_dict["planner_debug_timestamps"] = np.asarray(
#                 self.planner_debug_timestamps, dtype=np.float64
#             )
#             save_dict["planner_debug_plan_index"] = np.asarray(
#                 self.planner_debug_plan_index, dtype=np.int32
#             )
#             self.get_logger().info(
#                 f"[Reader] Saving {len(self.planner_debug_json)} planner debug records"
#             )
#         else:
#             self.get_logger().warn(
#                 "[Reader] No planner debug records were received. Saving NPZ without planner debug fields."
#             )

#         np.savez(npz_save_path, **save_dict)
#         self.get_logger().info(f"[Reader] Saved NPZ: {npz_save_path}")

#     def on_shutdown(self):
#         try:
#             self.save_npz()
#         except Exception as e:
#             self.get_logger().error(f"Failed to save NPZ: {e}")


# def main(args=None):
#     rclpy.init(args=args)
#     node = ThermalBagReader()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.save_npz()
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()
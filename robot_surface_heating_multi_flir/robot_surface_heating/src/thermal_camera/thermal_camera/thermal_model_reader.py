#!/usr/bin/env python3
"""
rosbag_reader.py
----------------
Reads a rosbag with thermal point clouds & robot joint states, transforms point
clouds into a common frame, computes EEF poses from URDF (FK), resamples to
regular timesteps, and saves a training-ready NPZ.

Outputs (NPZ):
  - thermal_frames: (T,) object array; each entry is (Ni,4) [x,y,z,thermal] in target_frame
  - eef_pos_quat:   (T,7) float32 array; [x,y,z,qx,qy,qz,qw] in target_frame
  - joint_positions:(T,) object array; (n_joints,) interpolated joints at each time
  - joint_names:    (n_joints,) object array
  - timestamps:     (T,) float64 seconds, uniform timeline
  - dt_target:      float64 seconds
  - target_frame:   str (e.g., "base_link")
"""

import os
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2, JointState
from sensor_msgs_py import point_cloud2
import tf2_ros
from scipy.spatial.transform import Rotation as R

from urdfpy import URDF  # FK on joint states

# ------------------------
# Config helpers (edit here)
# ------------------------

URDF_PATH = "/tmp/my_robot_cell.urdf"
MESH_ROOT = (
    "/home/cam/robot_surface_heating_dev/"
    "robot_surface_heating_multi_flir/"
    "robot_surface_heating/"
    "src/Universal_Robots_ROS2_Description"
)
TARGET_FRAME = "base_link"  # coordinate frame for pointclouds & EEF pose
EEF_LINK = "heat_gun"       # your end-effector link name (URDF)


class ThermalBagReader(Node):
    def __init__(self):
        super().__init__("thermal_bag_reader")

        # ---------- Parameters you may tune ----------
        self.target_frame = TARGET_FRAME
        self.eef_link_name = EEF_LINK

        # Match to camera FPS: e.g., 9 Hz -> dt ≈ 0.111 s
        self.dt_target = 0.125

        # Save path
        self.output_dir = os.path.join(
            os.path.expanduser("~"),
            "thermal_npz_datasets",
        )
        os.makedirs(self.output_dir, exist_ok=True)

        self.get_logger().info(f"[Reader] target_frame={self.target_frame}, dt_target={self.dt_target:.3f}s")
        self.get_logger().info(f"[Reader] URDF={URDF_PATH}, eef_link={self.eef_link_name}")

        # ---------- Load URDF for FK ----------
        try:
            self.robot = URDF.load(URDF_PATH)
            # quick validation that link exists:
            _ = {lnk.name for lnk in self.robot.links}
            if self.eef_link_name not in _:
                self.get_logger().warn(f"EEF link '{self.eef_link_name}' not found in URDF; "
                                       f"FK will likely fail unless corrected.")
        except Exception as e:
            self.get_logger().error(f"Failed to load URDF '{URDF_PATH}': {e}")
            raise

        # ---------- TF ----------
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # ---------- Buffers ----------
        # Thermal (variable-sized frames) + timestamps (ns)
        self.thermal_frames = []     # list of (Ni,4) float32 [x,y,z,thermal] in target_frame
        self.thermal_times_ns = []   # list of int (nanoseconds)

        # Joint states raw stream
        self.joint_times_ns = []     # list of int (nanoseconds)
        self.joint_names = None      # list[str], set on first joint msg
        self.joint_series = []       # list of float32 arrays (n_joints,), same order as joint_names

        # ---------- Subscriptions ----------
        self.pc_topic = "/raw_thermal_selected"
        self.js_topic = "/joint_states"

        self.create_subscription(PointCloud2, self.pc_topic, self.pc_cb, 10)
        self.create_subscription(JointState, self.js_topic, self.js_cb, 50)

        self.get_logger().info(f"Listening:\n  - {self.pc_topic}\n  - {self.js_topic}")

    # ============================================================
    # Callbacks
    # ============================================================
    def pc_cb(self, msg: PointCloud2):
        """Ingest point cloud, transform into target_frame, and buffer."""
        pts_list = list(point_cloud2.read_points(msg, field_names=("x", "y", "z", "thermal"), skip_nans=True))
        if not pts_list:
            return

        # FIX: handle structured dtype properly
        arr = np.array(pts_list)
        if getattr(arr.dtype, "names", None):  # structured array
            arr = np.stack([arr["x"], arr["y"], arr["z"], arr["thermal"]], axis=-1).astype(np.float32)
        else:
            arr = np.asarray(arr, dtype=np.float32).reshape(-1, 4)

        # Transform to target_frame (e.g., base_link)
        try:
            transform = self.tf_buffer.lookup_transform(self.target_frame, msg.header.frame_id, rclpy.time.Time())
            t = transform.transform.translation
            q = transform.transform.rotation
            rot = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = rot
            T[:3, 3] = [t.x, t.y, t.z]
            pts_h = np.hstack([arr[:, :3], np.ones((arr.shape[0], 1), dtype=np.float32)])
            arr[:, :3] = (T @ pts_h.T).T[:, :3]
        except Exception as e:
            self.get_logger().warn(
                f"TF lookup failed ({msg.header.frame_id}→{self.target_frame}): {e}. Using raw coordinates."
            )

        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.thermal_frames.append(arr)
        self.thermal_times_ns.append(stamp_ns)


    def js_cb(self, msg: JointState):
        """Buffer joint vectors; initialize joint_names on first message."""
        if self.joint_names is None:
            self.joint_names = list(msg.name)
            self.get_logger().info(f"[Reader] joint_names = {self.joint_names}")

        # Ensure consistent joint order; if different order arrives, reorder to first-seen order
        name_to_idx = {n: i for i, n in enumerate(msg.name)}
        vec = np.zeros(len(self.joint_names), dtype=np.float32)
        for i, n in enumerate(self.joint_names):
            if n in name_to_idx:
                vec[i] = float(msg.position[name_to_idx[n]])
            else:
                # missing joint; keep 0 (or last known)? zero is fine; interpolation will smooth
                pass

        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.joint_times_ns.append(stamp_ns)
        self.joint_series.append(vec)

    # ============================================================
    # Post-processing: build uniform timeline, FK, packing
    # ============================================================
    @staticmethod
    def _ns_to_s(ns_array):
        return np.asarray(ns_array, dtype=np.float64) * 1e-9

    def _build_uniform_timeline(self):
        """Build uniform timestamps (seconds) spanning overlap of PC & JS streams."""
        if len(self.thermal_times_ns) < 2 or len(self.joint_times_ns) < 2:
            raise RuntimeError("Insufficient data to build a timeline.")

        t_pc = self._ns_to_s(self.thermal_times_ns)
        t_js = self._ns_to_s(self.joint_times_ns)

        t0 = max(t_pc.min(), t_js.min())
        t1 = min(t_pc.max(), t_js.max())
        if t1 <= t0:
            raise RuntimeError("No temporal overlap between thermal and joint data streams.")

        # uniform timestamps
        T = int(np.floor((t1 - t0) / self.dt_target)) + 1
        t_uniform = (t0 + np.arange(T) * self.dt_target).astype(np.float64)
        return t_uniform

    def _resample_pc_nearest(self, t_uniform):
        """Pick nearest thermal frame for each uniform time."""
        t_pc = self._ns_to_s(self.thermal_times_ns)
        order = np.argsort(t_pc)
        t_pc = t_pc[order]
        pc_frames_sorted = [self.thermal_frames[i] for i in order]

        out_frames = []
        out_src_times = []
        for tu in t_uniform:
            idx = int(np.argmin(np.abs(t_pc - tu)))
            out_frames.append(pc_frames_sorted[idx])
            out_src_times.append(t_pc[idx])
        return out_frames, np.asarray(out_src_times)

    def _interp_joints(self, t_uniform):
        """Linearly interpolate each joint independently to t_uniform."""
        t_js = self._ns_to_s(self.joint_times_ns)
        order = np.argsort(t_js)
        t_js = t_js[order]
        J = np.stack([self.joint_series[i] for i in order], axis=0)  # (S, n_joints)

        n_j = J.shape[1]
        interp = np.zeros((len(t_uniform), n_j), dtype=np.float32)
        for j in range(n_j):
            interp[:, j] = np.interp(t_uniform, t_js, J[:, j])  # 1D linear
        # Return list-of-vectors for symmetry with PC frames
        return [interp[i] for i in range(interp.shape[0])]

    def _fk_eef(self, joint_vec):
        """Compute EEF pose [x,y,z,qx,qy,qz,qw] in the robot base frame (URDF base).
           We assume target_frame == URDF base frame name (commonly 'base_link')."""
        # urdfpy expects a dict of joint_name -> position
        cfg = {n: float(v) for n, v in zip(self.joint_names, joint_vec)}
        try:
            tf_dict = self.robot.link_fk(cfg=cfg)  # dict{Link: 4x4}
        except Exception as e:
            # FK failure; return zeros to keep array shapes intact
            self.get_logger().warn(f"URDF FK failed: {e}")
            return np.zeros(7, dtype=np.float32)

        # Find the matrix for eef_link_name
        T_eef = None
        for link, T in tf_dict.items():
            if link.name == self.eef_link_name:
                T_eef = np.asarray(T, dtype=np.float64)
                break

        if T_eef is None:
            self.get_logger().warn(f"EEF link '{self.eef_link_name}' not found in FK result; zeros returned.")
            return np.zeros(7, dtype=np.float32)

        pos = T_eef[:3, 3].astype(np.float64)
        quat = R.from_matrix(T_eef[:3, :3]).as_quat()  # [qx,qy,qz,qw]
        out = np.hstack([pos, quat]).astype(np.float32)
        return out

    # ============================================================
    # Save NPZ
    # ============================================================
    def save_npz(self):
        if not self.thermal_frames or not self.joint_series:
            self.get_logger().error("No data recorded; nothing to save.")
            return

        # 1) Uniform timeline inside overlap:
        t_uniform = self._build_uniform_timeline()

        # 2) Resample thermal frames (nearest)
        pc_frames_u, pc_src_times = self._resample_pc_nearest(t_uniform)

        # 3) Interpolate joints
        joints_u = self._interp_joints(t_uniform)  # list of (n_joints,)

        # 4) FK for each time
        eef_pos_quat = np.stack([self._fk_eef(jv) for jv in joints_u], axis=0)  # (T,7)

        # 5) Pack & save
        # fname = os.path.join(self.output_dir, "thermal_robot_dataset_fk.npz")
        fname = os.path.join(self.output_dir, "11-21-25_fine_moving_2.npz")
        np.savez(
            fname,
            thermal_frames=np.array(pc_frames_u, dtype=object),      # (T,) (Ni,4) [x,y,z,thermal] in target_frame
            eef_pos_quat=eef_pos_quat.astype(np.float32),            # (T,7) [x,y,z,qx,qy,qz,qw]
            joint_positions=np.array(joints_u, dtype=object),        # (T,) (n_joints,)
            joint_names=np.array(self.joint_names, dtype=object),    # (n_joints,)
            timestamps=t_uniform.astype(np.float64),                 # (T,)
            dt_target=float(self.dt_target),
            target_frame=self.target_frame,
            pc_source_times=pc_src_times.astype(np.float64),         # (T,) source PC timestamp for traceability
        )
        self.get_logger().info(f"[Reader] Saved NPZ: {fname}")

    # ============================================================
    # Spin/Shutdown
    # ============================================================
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

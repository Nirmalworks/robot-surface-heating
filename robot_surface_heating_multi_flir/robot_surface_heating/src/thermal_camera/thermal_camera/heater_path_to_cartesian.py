#!/usr/bin/env python3

import time

import math
import numpy as np
import rclpy
import tf2_ros

from rclpy.node import Node
from geometry_msgs.msg import Pose, PoseArray, Quaternion
from thermal_camera_interfaces.msg import Extrema, ProjectedHeaterPath
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R, Slerp
from tf2_ros import TransformException


# ============================================================
# User settings
# ============================================================

PROJECTED_PATH_TOPIC = "/heater_path_projected"
SURFACE_TOPIC = "/cad_pointcloud"

EXTREMA_PATH_TOPIC = "/heater_policy_poses"
DEBUG_POSE_ARRAY_TOPIC = "/heater_cartesian_path_debug"

EEF_FRAME = "heat_gun"


# --- COMPOSITE PART SETTINGS ---

# # Standoff distance from surface along normal
# # SURFACE_OFFSET_M = 0.2
# SURFACE_OFFSET_M = 0.17

# # Final published waypoint spacing to executor
# FINAL_WAYPOINT_SPACING_M = 0.02

# # Internal resampling spacing before final downsampling
# INTERNAL_RESAMPLE_SPACING_M = 0.005

# # Weighted local interpolation in projected space
# LIFT_K = 8
# LIFT_SIGMA_M = 0.015
# LIFT_EPS = 1e-9

# # Smoothing windows
# POSITION_SMOOTH_WINDOW = 11
# NORMAL_SMOOTH_WINDOW = 11

# # Preserve the start of the path so smoothing does not drag it away
# SMOOTH_FIXED_PREFIX_COUNT = 2

# # Live EEF anchoring / blending
# ANCHOR_TO_LIVE_EEF = True
# MAX_LIVE_ANCHOR_DIST_M = 0.10
# POSITION_BLEND_COUNT = 6
# ORIENTATION_BLEND_COUNT = 6

# # Orientation generation
# TANGENT_EPS = 1e-8
# REF_EPS = 1e-8
# WORLD_REF_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)

# # Tool axis convention:
# # If the heat gun points "down" along local -Z, use -1
# # If it points "up" along local +Z, use +1
# TOOL_AXIS_SIGN = 1.0

# # Designate up axis for pointcloud
# UP_AXIS = 1   # 0=X, 1=Y, 2=Z

# # Use a fixed world reference to keep constant twist about tool Z
# TWIST_REFERENCE_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)

# # Constant twist about local tool Z, in radians
# TOOL_TWIST_RAD = 0.0

# --- COMPOSITE PART SETTINGS ---

# Standoff distance from surface along normal
# SURFACE_OFFSET_M = 0.2
SURFACE_OFFSET_M = 0.17

# Final published waypoint spacing to executor
FINAL_WAYPOINT_SPACING_M = 0.02

# Internal resampling spacing before final downsampling
INTERNAL_RESAMPLE_SPACING_M = 0.005

# Weighted local interpolation in projected space
LIFT_K = 8
LIFT_SIGMA_M = 0.015
LIFT_EPS = 1e-9

# Smoothing windows
POSITION_SMOOTH_WINDOW = 11
NORMAL_SMOOTH_WINDOW = 11

# Preserve the start of the path so smoothing does not drag it away
SMOOTH_FIXED_PREFIX_COUNT = 2

# Live EEF anchoring / blending
ANCHOR_TO_LIVE_EEF = True
MAX_LIVE_ANCHOR_DIST_M = 0.10
POSITION_BLEND_COUNT = 6
ORIENTATION_BLEND_COUNT = 6

# Orientation generation
TANGENT_EPS = 1e-8
REF_EPS = 1e-8
WORLD_REF_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)

# Tool axis convention:
# If the heat gun points "down" along local -Z, use -1
# If it points "up" along local +Z, use +1
# TOOL_AXIS_SIGN = 1.0
TOOL_AXIS_SIGN = 1.0

# Designate up axis for pointcloud
# ====== CHANGE THESE ======
UP_AXIS = 2   # 0=X, 1=Y, 2=Z

# Constant twist about local tool Z, in radians
TOOL_TWIST_RAD = math.pi
# TOOL_TWIST_RAD = 0.0
# ====== CHANGE THESE ======

# Use a fixed world reference to keep constant twist about tool Z
TWIST_REFERENCE_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)

# ============================================================
# Utilities
# ============================================================

def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return np.zeros_like(v)
    return v / n


def normalize_rows(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    out = M.copy()
    n = np.linalg.norm(out, axis=1, keepdims=True)
    good = n[:, 0] > eps
    out[good] /= n[good]
    return out


def project_vector_to_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
    return v - np.dot(v, n) * n


def ensure_normal_continuity(normals: np.ndarray) -> np.ndarray:
    out = normals.copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0.0:
            out[i] *= -1.0
    return out


def enforce_positive_slope(normals: np.ndarray, up_axis: int) -> np.ndarray:
    out = normals.copy()
    flip_mask = out[:, up_axis] < 0.0
    out[flip_mask] *= -1.0
    return out


def ensure_quat_continuity(quats_xyzw: np.ndarray) -> np.ndarray:
    out = quats_xyzw.copy()
    for i in range(1, len(out)):
        if np.dot(out[i], out[i - 1]) < 0.0:
            out[i] *= -1.0
    return out


def rotation_matrix_to_quaternion(Rm: np.ndarray) -> np.ndarray:
    return R.from_matrix(Rm).as_quat()


def make_pose(position_xyz: np.ndarray, quat_xyzw: np.ndarray) -> Pose:
    p = Pose()
    p.position.x = float(position_xyz[0])
    p.position.y = float(position_xyz[1])
    p.position.z = float(position_xyz[2])
    p.orientation = Quaternion(
        x=float(quat_xyzw[0]),
        y=float(quat_xyzw[1]),
        z=float(quat_xyzw[2]),
        w=float(quat_xyzw[3]),
    )
    return p


def moving_average_vectors(arr: np.ndarray, window: int) -> np.ndarray:
    if len(arr) < 3 or window < 3:
        return arr.copy()

    out = arr.copy()
    half = window // 2

    for i in range(len(arr)):
        i0 = max(0, i - half)
        i1 = min(len(arr), i + half + 1)
        out[i] = np.mean(arr[i0:i1], axis=0)

    return out


def smooth_positions(points: np.ndarray, window: int) -> np.ndarray:
    return moving_average_vectors(points, window)


def smooth_normals(normals: np.ndarray, window: int) -> np.ndarray:
    out = moving_average_vectors(normals, window)
    out = normalize_rows(out)
    out = ensure_normal_continuity(out)
    return out


def smooth_positions_preserve_prefix(points: np.ndarray, window: int, fixed_prefix_count: int) -> np.ndarray:
    out = smooth_positions(points, window)
    k = max(0, min(int(fixed_prefix_count), len(points)))
    if k > 0:
        out[:k] = points[:k]
    return out


def smooth_normals_preserve_prefix(normals: np.ndarray, window: int, fixed_prefix_count: int) -> np.ndarray:
    out = smooth_normals(normals, window)
    k = max(0, min(int(fixed_prefix_count), len(normals)))
    if k > 0:
        out[:k] = normals[:k]
    out = normalize_rows(out)
    out = ensure_normal_continuity(out)
    return out


def cumulative_arclength(points_xyz: np.ndarray) -> np.ndarray:
    if len(points_xyz) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(points_xyz) == 1:
        return np.zeros((1,), dtype=np.float64)

    seg = np.linalg.norm(points_xyz[1:] - points_xyz[:-1], axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    return s


def resample_path_uniform(
    positions_xyz: np.ndarray,
    normals_xyz: np.ndarray,
    spacing_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(positions_xyz) <= 2:
        return positions_xyz.copy(), normals_xyz.copy()

    s = cumulative_arclength(positions_xyz)
    total = float(s[-1])

    if total < 1e-9:
        return positions_xyz.copy(), normals_xyz.copy()

    num = max(2, int(np.ceil(total / spacing_m)) + 1)
    s_new = np.linspace(0.0, total, num)

    pos_new = np.zeros((num, 3), dtype=np.float64)
    nrm_new = np.zeros((num, 3), dtype=np.float64)

    for j in range(3):
        pos_new[:, j] = np.interp(s_new, s, positions_xyz[:, j])
        nrm_new[:, j] = np.interp(s_new, s, normals_xyz[:, j])

    nrm_new = normalize_rows(nrm_new)
    nrm_new = ensure_normal_continuity(nrm_new)
    return pos_new, nrm_new


def downsample_by_spacing(points_xyz: np.ndarray, min_spacing_m: float) -> np.ndarray:
    if len(points_xyz) <= 1:
        return np.arange(len(points_xyz), dtype=np.int32)

    keep = [0]
    last = points_xyz[0]

    for i in range(1, len(points_xyz) - 1):
        if np.linalg.norm(points_xyz[i] - last) >= min_spacing_m:
            keep.append(i)
            last = points_xyz[i]

    keep.append(len(points_xyz) - 1)
    return np.asarray(keep, dtype=np.int32)


def blend_prefix_positions(
    points_xyz: np.ndarray,
    live_position_xyz: np.ndarray,
    blend_count: int,
) -> np.ndarray:
    out = points_xyz.copy()
    if len(out) == 0:
        return out

    n = max(1, min(int(blend_count), len(out)))
    original = out.copy()
    out[0] = live_position_xyz

    if n == 1:
        return out

    for i in range(1, n):
        alpha = float(i) / float(n - 1)
        out[i] = (1.0 - alpha) * live_position_xyz + alpha * original[i]

    return out


def blend_prefix_quaternions(
    quats_xyzw: np.ndarray,
    live_quat_xyzw: np.ndarray,
    blend_count: int,
) -> np.ndarray:
    out = quats_xyzw.copy()
    if len(out) == 0:
        return out

    n = max(1, min(int(blend_count), len(out)))
    if n == 1:
        out[0] = live_quat_xyzw
        return ensure_quat_continuity(out)

    key_rots = R.from_quat(np.vstack([live_quat_xyzw, quats_xyzw[n - 1]]))
    slerp = Slerp([0.0, 1.0], key_rots)
    interp_rots = slerp(np.linspace(0.0, 1.0, n))
    out[:n] = interp_rots.as_quat()
    out = ensure_quat_continuity(out)
    return out


# ============================================================
# Main node
# ============================================================

class HeaterPathToCartesianNode(Node):
    def __init__(self):
        super().__init__("heater_path_to_cartesian")

        self.surface_points = None
        self.surface_normals = None
        self.surface_frame_id = None

        self.path_sub = self.create_subscription(
            ProjectedHeaterPath,
            PROJECTED_PATH_TOPIC,
            self.projected_path_callback,
            10,
        )

        self.surface_sub = self.create_subscription(
            PointCloud2,
            SURFACE_TOPIC,
            self.surface_callback,
            10,
        )

        self.pose_pub = self.create_publisher(Extrema, EXTREMA_PATH_TOPIC, 10)
        self.debug_pose_pub = self.create_publisher(PoseArray, DEBUG_POSE_ARRAY_TOPIC, 10)

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.get_logger().info(f"Listening for projected paths on {PROJECTED_PATH_TOPIC}")
        self.get_logger().info(f"Listening for surface geometry on {SURFACE_TOPIC}")
        self.get_logger().info(f"Publishing cartesian waypoints on {EXTREMA_PATH_TOPIC}")

    # --------------------------------------------------------
    # Surface geometry
    # --------------------------------------------------------

    def surface_callback(self, msg: PointCloud2):
        pts = []
        nrms = []

        for p in point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z", "normal_x", "normal_y", "normal_z"),
            skip_nans=True,
        ):
            x, y, z, nx, ny, nz = p
            pts.append([x, y, z])
            nrms.append([nx, ny, nz])

        if len(pts) == 0:
            self.get_logger().warn("Received empty CAD/surface pointcloud")
            return

        pts = np.asarray(pts, dtype=np.float64)
        nrms = np.asarray(nrms, dtype=np.float64)
        nrms = normalize_rows(nrms)

        bad = np.linalg.norm(nrms, axis=1) < 1e-12
        nrms[bad] = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        nrms = enforce_positive_slope(nrms, UP_AXIS)

        self.surface_points = pts
        self.surface_normals = nrms
        self.surface_frame_id = msg.header.frame_id

    # --------------------------------------------------------
    # Live EEF pose
    # --------------------------------------------------------

    def get_live_eef_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        if self.surface_frame_id is None:
            return None

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.surface_frame_id,
                EEF_FRAME,
                rclpy.time.Time(),
            )
        except TransformException:
            return None
        except Exception:
            return None

        pos = np.array(
            [
                tf_msg.transform.translation.x,
                tf_msg.transform.translation.y,
                tf_msg.transform.translation.z,
            ],
            dtype=np.float64,
        )

        quat = np.array(
            [
                tf_msg.transform.rotation.x,
                tf_msg.transform.rotation.y,
                tf_msg.transform.rotation.z,
                tf_msg.transform.rotation.w,
            ],
            dtype=np.float64,
        )

        qn = np.linalg.norm(quat)
        if qn < 1e-12:
            return None
        quat = quat / qn

        return pos, quat

    # --------------------------------------------------------
    # Projected path callback
    # --------------------------------------------------------

    def projected_path_callback(self, msg: ProjectedHeaterPath):
        t0 = time.perf_counter()

        if self.surface_points is None or self.surface_normals is None:
            self.get_logger().warn("No surface geometry available yet.")
            return

        if len(msg.path_proj) < 2:
            self.get_logger().warn("Projected path message contains too few path points.")
            return

        center = np.array(
            [msg.center.x, msg.center.y, msg.center.z],
            dtype=np.float64,
        )
        axis_u = normalize(np.array(
            [msg.axis_u.x, msg.axis_u.y, msg.axis_u.z],
            dtype=np.float64,
        ))
        axis_v = normalize(np.array(
            [msg.axis_v.x, msg.axis_v.y, msg.axis_v.z],
            dtype=np.float64,
        ))

        path_proj = np.array(
            [[p.x, p.y] for p in msg.path_proj],
            dtype=np.float64,
        )

        t1 = time.perf_counter()

        surf_q = self.surface_points - center[None, :]
        surf_u = surf_q @ axis_u
        surf_v = surf_q @ axis_v
        surf_uv = np.column_stack((surf_u, surf_v)).astype(np.float64)
        surf_tree = cKDTree(surf_uv)

        query_k = min(LIFT_K, len(self.surface_points))
        dists, idx = surf_tree.query(path_proj, k=query_k)

        if query_k == 1:
            dists = dists[:, None]
            idx = idx[:, None]

        weights = np.exp(-0.5 * (dists / max(LIFT_SIGMA_M, 1e-6)) ** 2)
        weights = weights / np.clip(np.sum(weights, axis=1, keepdims=True), LIFT_EPS, None)

        surface_path_xyz = np.zeros((len(path_proj), 3), dtype=np.float64)
        surface_path_nrm = np.zeros((len(path_proj), 3), dtype=np.float64)

        for i in range(len(path_proj)):
            nbr_xyz = self.surface_points[idx[i]]
            nbr_nrm = self.surface_normals[idx[i]]
            w = weights[i][:, None]

            surface_path_xyz[i] = np.sum(w * nbr_xyz, axis=0)
            surface_path_nrm[i] = np.sum(w * nbr_nrm, axis=0)

        surface_path_nrm = normalize_rows(surface_path_nrm)
        surface_path_nrm = enforce_positive_slope(surface_path_nrm, UP_AXIS)
        surface_path_nrm = ensure_normal_continuity(surface_path_nrm)

        t2 = time.perf_counter()

        raw_surface_path_xyz = surface_path_xyz.copy()
        raw_surface_path_nrm = surface_path_nrm.copy()

        surface_path_xyz = smooth_positions_preserve_prefix(
            surface_path_xyz,
            POSITION_SMOOTH_WINDOW,
            SMOOTH_FIXED_PREFIX_COUNT,
        )
        surface_path_nrm = smooth_normals_preserve_prefix(
            surface_path_nrm,
            NORMAL_SMOOTH_WINDOW,
            SMOOTH_FIXED_PREFIX_COUNT,
        )

        surface_path_xyz, surface_path_nrm = resample_path_uniform(
            surface_path_xyz,
            surface_path_nrm,
            INTERNAL_RESAMPLE_SPACING_M,
        )

        surface_path_xyz = smooth_positions_preserve_prefix(
            surface_path_xyz,
            POSITION_SMOOTH_WINDOW,
            1,
        )
        surface_path_nrm = smooth_normals_preserve_prefix(
            surface_path_nrm,
            NORMAL_SMOOTH_WINDOW,
            1,
        )

        surface_path_xyz[0] = raw_surface_path_xyz[0]
        surface_path_nrm[0] = raw_surface_path_nrm[0]
        surface_path_nrm = normalize_rows(surface_path_nrm)
        surface_path_nrm = enforce_positive_slope(surface_path_nrm, UP_AXIS)
        surface_path_nrm = ensure_normal_continuity(surface_path_nrm)

        tool_positions = surface_path_xyz + SURFACE_OFFSET_M * surface_path_nrm

        side_check = np.sum((tool_positions - surface_path_xyz) * surface_path_nrm, axis=1)
        if np.any(side_check < 0.0):
            self.get_logger().warn(
                "Some tool positions are on the wrong side of the surface. Re-enforcing positive slope."
            )
            surface_path_nrm = enforce_positive_slope(surface_path_nrm, UP_AXIS)
            surface_path_nrm = ensure_normal_continuity(surface_path_nrm)
            surface_path_nrm = normalize_rows(surface_path_nrm)
            tool_positions = surface_path_xyz + SURFACE_OFFSET_M * surface_path_nrm

        live_pose = self.get_live_eef_pose()
        live_anchor_used = False
        live_position = None
        live_quat = None

        if ANCHOR_TO_LIVE_EEF and live_pose is not None:
            live_position, live_quat = live_pose
            anchor_dist_m = float(np.linalg.norm(tool_positions[0] - live_position))

            if anchor_dist_m <= MAX_LIVE_ANCHOR_DIST_M:
                tool_positions = blend_prefix_positions(
                    tool_positions,
                    live_position,
                    POSITION_BLEND_COUNT,
                )
                live_anchor_used = True

        keep = downsample_by_spacing(tool_positions, FINAL_WAYPOINT_SPACING_M)

        if len(keep) == 0 or keep[0] != 0:
            keep = np.concatenate(([0], keep.astype(np.int32)))

        surface_path_xyz = surface_path_xyz[keep]
        surface_path_nrm = surface_path_nrm[keep]
        tool_positions = tool_positions[keep]

        if len(tool_positions) < 2:
            self.get_logger().warn("Lifted path has fewer than 2 poses after processing.")
            return

        t3 = time.perf_counter()

        quats = self.build_minimum_twist_quaternions(
            positions_xyz=tool_positions,
            normals_xyz=surface_path_nrm,
        )

        # if live_anchor_used and live_quat is not None:
        #     quats = blend_prefix_quaternions(
        #         quats,
        #         live_quat,
        #         ORIENTATION_BLEND_COUNT,
        #     )
        if live_anchor_used and live_quat is not None:
            live_quat_aligned = self.align_live_quaternion_to_path(
                live_quat_xyzw=live_quat,
                path_quat_xyzw=quats[0],
            )

            quats = blend_prefix_quaternions(
                quats,
                live_quat_aligned,
                ORIENTATION_BLEND_COUNT,
            )

        quats = ensure_quat_continuity(quats)

        msg_out = Extrema()
        msg_out.header = msg.header
        msg_out.header.frame_id = self.surface_frame_id
        msg_out.poses = [make_pose(p, q) for p, q in zip(tool_positions, quats)]
        msg_out.value = 0.0

        t4 = time.perf_counter()

        self.pose_pub.publish(msg_out)

        pose_array = PoseArray()
        pose_array.header = msg_out.header
        pose_array.poses = [make_pose(p, q) for p, q in zip(tool_positions, quats)]
        self.debug_pose_pub.publish(pose_array)

        t5 = time.perf_counter()

        self.get_logger().info(
            f"[TIMING path_to_cart] parse={t1-t0:.3f}s "
            f"lift={t2-t1:.3f}s "
            f"smooth_resample={t3-t2:.3f}s "
            f"orient_build={t4-t3:.3f}s "
            f"publish={t5-t4:.3f}s "
            f"total={t5-t0:.3f}s "
            f"in_pts={len(path_proj)} out_pts={len(msg_out.poses)} "
            f"live_anchor={live_anchor_used}"
        )

        self.get_logger().info(
            f"Published {len(msg_out.poses)} smooth cartesian heater poses to {EXTREMA_PATH_TOPIC}"
        )

    # --------------------------------------------------------
    # Orientation generation
    # --------------------------------------------------------

    def build_minimum_twist_quaternions(
        self,
        positions_xyz: np.ndarray,
        normals_xyz: np.ndarray,
    ) -> np.ndarray:
        N = len(positions_xyz)
        quats = np.zeros((N, 4), dtype=np.float64)

        prev_x = None

        for i in range(N):
            n = normalize(normals_xyz[i])

            # Tool Z follows the surface normal
            z_tool = normalize(TOOL_AXIS_SIGN * n)

            # Use ONLY a fixed world reference projected into the tangent plane
            x_ref = project_vector_to_plane(TWIST_REFERENCE_AXIS, z_tool)
            x_ref = normalize(x_ref)

            if np.linalg.norm(x_ref) < REF_EPS:
                alt = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                x_ref = project_vector_to_plane(alt, z_tool)
                x_ref = normalize(x_ref)

            if np.linalg.norm(x_ref) < REF_EPS:
                alt = np.array([0.0, 1.0, 0.0], dtype=np.float64)
                x_ref = project_vector_to_plane(alt, z_tool)
                x_ref = normalize(x_ref)

            if np.linalg.norm(x_ref) < REF_EPS and prev_x is not None:
                x_ref = project_vector_to_plane(prev_x, z_tool)
                x_ref = normalize(x_ref)

            if np.linalg.norm(x_ref) < REF_EPS:
                x_ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
                x_ref = project_vector_to_plane(x_ref, z_tool)
                x_ref = normalize(x_ref)

            # Build orthonormal basis
            y_ref = normalize(np.cross(z_tool, x_ref))
            x_ref = normalize(np.cross(y_ref, z_tool))

            # Apply constant twist about local Z
            c = np.cos(TOOL_TWIST_RAD)
            s = np.sin(TOOL_TWIST_RAD)

            x_tool = normalize(c * x_ref + s * y_ref)
            y_tool = normalize(np.cross(z_tool, x_tool))
            x_tool = normalize(np.cross(y_tool, z_tool))

            # Keep sign continuity only
            if prev_x is not None and np.dot(x_tool, prev_x) < 0.0:
                x_tool *= -1.0
                y_tool *= -1.0

            prev_x = x_tool.copy()

            Rm = np.column_stack((x_tool, y_tool, z_tool))
            quats[i] = rotation_matrix_to_quaternion(Rm)

        quats = ensure_quat_continuity(quats)
        return quats

    def align_live_quaternion_to_path(
        self,
        live_quat_xyzw: np.ndarray,
        path_quat_xyzw: np.ndarray,
    ) -> np.ndarray:
        live_rot = R.from_quat(live_quat_xyzw)
        path_rot = R.from_quat(path_quat_xyzw)

        # Candidate 0: raw live orientation
        cand0 = live_rot

        # Candidate 1: flip 180 deg about local X
        cand1 = live_rot * R.from_rotvec(np.pi * np.array([1.0, 0.0, 0.0], dtype=np.float64))

        # Candidate 2: flip 180 deg about local Y
        cand2 = live_rot * R.from_rotvec(np.pi * np.array([0.0, 1.0, 0.0], dtype=np.float64))

        # Candidate 3: flip 180 deg about local Z
        cand3 = live_rot * R.from_rotvec(np.pi * np.array([0.0, 0.0, 1.0], dtype=np.float64))

        candidates = [cand0, cand1, cand2, cand3]

        best_quat = live_quat_xyzw.copy()
        best_score = -np.inf

        path_axes = path_rot.as_matrix()
        path_x = path_axes[:, 0]
        path_y = path_axes[:, 1]
        path_z = path_axes[:, 2]

        for cand in candidates:
            axes = cand.as_matrix()
            cx = axes[:, 0]
            cy = axes[:, 1]
            cz = axes[:, 2]

            # Prefer candidates whose axes match the path frame best
            score = (
                np.dot(cx, path_x) +
                np.dot(cy, path_y) +
                np.dot(cz, path_z)
            )

            if score > best_score:
                best_score = score
                best_quat = cand.as_quat()

        return best_quat

def main(args=None):
    rclpy.init(args=args)
    node = HeaterPathToCartesianNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


# #!/usr/bin/env python3

# import time

# import numpy as np
# import rclpy

# from rclpy.node import Node
# from geometry_msgs.msg import Pose, PoseArray, Quaternion
# from thermal_camera_interfaces.msg import Extrema, ProjectedHeaterPath
# from sensor_msgs.msg import PointCloud2
# from sensor_msgs_py import point_cloud2
# from scipy.spatial import cKDTree
# from scipy.spatial.transform import Rotation as R


# # ============================================================
# # User settings
# # ============================================================

# PROJECTED_PATH_TOPIC = "/heater_path_projected"
# SURFACE_TOPIC = "/cad_pointcloud"

# EXTREMA_PATH_TOPIC = "/heater_policy_poses"
# DEBUG_POSE_ARRAY_TOPIC = "/heater_cartesian_path_debug"

# # Standoff distance from surface along normal
# SURFACE_OFFSET_M = 0.15

# # Final published waypoint spacing to executor
# FINAL_WAYPOINT_SPACING_M = 0.02

# # Internal resampling spacing before final downsampling
# INTERNAL_RESAMPLE_SPACING_M = 0.005

# # Weighted local interpolation in projected space
# LIFT_K = 8
# LIFT_SIGMA_M = 0.015
# LIFT_EPS = 1e-9

# # Smoothing windows
# POSITION_SMOOTH_WINDOW = 11
# NORMAL_SMOOTH_WINDOW = 11

# # Orientation generation
# TANGENT_EPS = 1e-8
# REF_EPS = 1e-8
# WORLD_REF_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)

# # Tool axis convention:
# # If the heat gun points "down" along local -Z, use -1
# # If it points "up" along local +Z, use +1
# TOOL_AXIS_SIGN = 1.0

# # Designate up axis for pointcloud
# UP_AXIS = 1   # 0=X, 1=Y, 2=Z

# # Use a fixed world reference to keep constant twist about tool Z
# TWIST_REFERENCE_AXIS = np.array([1.0, 0.0, 0.0], dtype=np.float64)

# # Constant twist about local tool Z, in radians
# # Tune this value until the gun cable / wrist orientation looks right
# TOOL_TWIST_RAD = 0.0


# # ============================================================
# # Utilities
# # ============================================================

# def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
#     n = np.linalg.norm(v)
#     if n < eps:
#         return np.zeros_like(v)
#     return v / n


# def normalize_rows(M: np.ndarray, eps: float = 1e-12) -> np.ndarray:
#     out = M.copy()
#     n = np.linalg.norm(out, axis=1, keepdims=True)
#     good = n[:, 0] > eps
#     out[good] /= n[good]
#     return out


# def project_vector_to_plane(v: np.ndarray, n: np.ndarray) -> np.ndarray:
#     return v - np.dot(v, n) * n


# def ensure_normal_continuity(normals: np.ndarray) -> np.ndarray:
#     out = normals.copy()
#     for i in range(1, len(out)):
#         if np.dot(out[i], out[i - 1]) < 0.0:
#             out[i] *= -1.0
#     return out


# def enforce_positive_slope(normals: np.ndarray, up_axis: int) -> np.ndarray:
#     out = normals.copy()
#     flip_mask = out[:, up_axis] < 0.0
#     out[flip_mask] *= -1.0
#     return out


# def ensure_quat_continuity(quats_xyzw: np.ndarray) -> np.ndarray:
#     out = quats_xyzw.copy()
#     for i in range(1, len(out)):
#         if np.dot(out[i], out[i - 1]) < 0.0:
#             out[i] *= -1.0
#     return out


# def rotation_matrix_to_quaternion(Rm: np.ndarray) -> np.ndarray:
#     return R.from_matrix(Rm).as_quat()


# def make_pose(position_xyz: np.ndarray, quat_xyzw: np.ndarray) -> Pose:
#     p = Pose()
#     p.position.x = float(position_xyz[0])
#     p.position.y = float(position_xyz[1])
#     p.position.z = float(position_xyz[2])
#     p.orientation = Quaternion(
#         x=float(quat_xyzw[0]),
#         y=float(quat_xyzw[1]),
#         z=float(quat_xyzw[2]),
#         w=float(quat_xyzw[3]),
#     )
#     return p


# def moving_average_vectors(arr: np.ndarray, window: int) -> np.ndarray:
#     if len(arr) < 3 or window < 3:
#         return arr.copy()

#     out = arr.copy()
#     half = window // 2

#     for i in range(len(arr)):
#         i0 = max(0, i - half)
#         i1 = min(len(arr), i + half + 1)
#         out[i] = np.mean(arr[i0:i1], axis=0)

#     return out


# def smooth_positions(points: np.ndarray, window: int) -> np.ndarray:
#     return moving_average_vectors(points, window)


# def smooth_normals(normals: np.ndarray, window: int) -> np.ndarray:
#     out = moving_average_vectors(normals, window)
#     out = normalize_rows(out)
#     out = ensure_normal_continuity(out)
#     return out


# def cumulative_arclength(points_xyz: np.ndarray) -> np.ndarray:
#     if len(points_xyz) == 0:
#         return np.zeros((0,), dtype=np.float64)
#     if len(points_xyz) == 1:
#         return np.zeros((1,), dtype=np.float64)

#     seg = np.linalg.norm(points_xyz[1:] - points_xyz[:-1], axis=1)
#     s = np.concatenate(([0.0], np.cumsum(seg)))
#     return s


# def resample_path_uniform(
#     positions_xyz: np.ndarray,
#     normals_xyz: np.ndarray,
#     spacing_m: float,
# ) -> tuple[np.ndarray, np.ndarray]:
#     if len(positions_xyz) <= 2:
#         return positions_xyz.copy(), normals_xyz.copy()

#     s = cumulative_arclength(positions_xyz)
#     total = float(s[-1])

#     if total < 1e-9:
#         return positions_xyz.copy(), normals_xyz.copy()

#     num = max(2, int(np.ceil(total / spacing_m)) + 1)
#     s_new = np.linspace(0.0, total, num)

#     pos_new = np.zeros((num, 3), dtype=np.float64)
#     nrm_new = np.zeros((num, 3), dtype=np.float64)

#     for j in range(3):
#         pos_new[:, j] = np.interp(s_new, s, positions_xyz[:, j])
#         nrm_new[:, j] = np.interp(s_new, s, normals_xyz[:, j])

#     nrm_new = normalize_rows(nrm_new)
#     nrm_new = ensure_normal_continuity(nrm_new)
#     return pos_new, nrm_new


# def downsample_by_spacing(points_xyz: np.ndarray, min_spacing_m: float) -> np.ndarray:
#     if len(points_xyz) <= 1:
#         return np.arange(len(points_xyz), dtype=np.int32)

#     keep = [0]
#     last = points_xyz[0]

#     for i in range(1, len(points_xyz) - 1):
#         if np.linalg.norm(points_xyz[i] - last) >= min_spacing_m:
#             keep.append(i)
#             last = points_xyz[i]

#     keep.append(len(points_xyz) - 1)
#     return np.asarray(keep, dtype=np.int32)


# # ============================================================
# # Main node
# # ============================================================

# class HeaterPathToCartesianNode(Node):
#     def __init__(self):
#         super().__init__("heater_path_to_cartesian")

#         self.surface_points = None
#         self.surface_normals = None
#         self.surface_frame_id = None

#         self.path_sub = self.create_subscription(
#             ProjectedHeaterPath,
#             PROJECTED_PATH_TOPIC,
#             self.projected_path_callback,
#             10,
#         )

#         self.surface_sub = self.create_subscription(
#             PointCloud2,
#             SURFACE_TOPIC,
#             self.surface_callback,
#             10,
#         )

#         self.pose_pub = self.create_publisher(Extrema, EXTREMA_PATH_TOPIC, 10)
#         self.debug_pose_pub = self.create_publisher(PoseArray, DEBUG_POSE_ARRAY_TOPIC, 10)

#         self.get_logger().info(f"Listening for projected paths on {PROJECTED_PATH_TOPIC}")
#         self.get_logger().info(f"Listening for surface geometry on {SURFACE_TOPIC}")
#         self.get_logger().info(f"Publishing cartesian waypoints on {EXTREMA_PATH_TOPIC}")

#     # --------------------------------------------------------
#     # Surface geometry
#     # --------------------------------------------------------

#     def surface_callback(self, msg: PointCloud2):
#         pts = []
#         nrms = []

#         for p in point_cloud2.read_points(
#             msg,
#             field_names=("x", "y", "z", "normal_x", "normal_y", "normal_z"),
#             skip_nans=True,
#         ):
#             x, y, z, nx, ny, nz = p
#             pts.append([x, y, z])
#             nrms.append([nx, ny, nz])

#         if len(pts) == 0:
#             self.get_logger().warn("Received empty CAD/surface pointcloud")
#             return

#         # pts = np.asarray(pts, dtype=np.float64)
#         # nrms = np.asarray(nrms, dtype=np.float64)
#         # nrms = normalize_rows(nrms)

#         # bad = np.linalg.norm(nrms, axis=1) < 1e-12
#         # nrms[bad] = np.array([0.0, 0.0, 1.0], dtype=np.float64)
#         pts = np.asarray(pts, dtype=np.float64)
#         nrms = np.asarray(nrms, dtype=np.float64)
#         nrms = normalize_rows(nrms)

#         bad = np.linalg.norm(nrms, axis=1) < 1e-12
#         nrms[bad] = np.array([0.0, 1.0, 0.0], dtype=np.float64)

#         nrms = enforce_positive_slope(nrms, UP_AXIS)

#         self.surface_points = pts
#         self.surface_normals = nrms
#         self.surface_frame_id = msg.header.frame_id

#     # --------------------------------------------------------
#     # Projected path callback
#     # --------------------------------------------------------

#     def projected_path_callback(self, msg: ProjectedHeaterPath):

#         t0 = time.perf_counter()

#         if self.surface_points is None or self.surface_normals is None:
#             self.get_logger().warn("No surface geometry available yet.")
#             return

#         if len(msg.path_proj) < 2:
#             self.get_logger().warn("Projected path message contains too few path points.")
#             return

#         center = np.array(
#             [msg.center.x, msg.center.y, msg.center.z],
#             dtype=np.float64,
#         )
#         axis_u = normalize(np.array(
#             [msg.axis_u.x, msg.axis_u.y, msg.axis_u.z],
#             dtype=np.float64,
#         ))
#         axis_v = normalize(np.array(
#             [msg.axis_v.x, msg.axis_v.y, msg.axis_v.z],
#             dtype=np.float64,
#         ))

#         path_proj = np.array(
#             [[p.x, p.y] for p in msg.path_proj],
#             dtype=np.float64,
#         )

#         # after checking required surface data exists and reading msg path arrays
#         t1 = time.perf_counter()

#         # Build projected surface cloud under the same projection frame
#         surf_q = self.surface_points - center[None, :]
#         surf_u = surf_q @ axis_u
#         surf_v = surf_q @ axis_v
#         surf_uv = np.column_stack((surf_u, surf_v)).astype(np.float64)
#         surf_tree = cKDTree(surf_uv)

#         # ----------------------------------------------------
#         # Smooth lift from 2D projected path to 3D surface
#         # ----------------------------------------------------
#         query_k = min(LIFT_K, len(self.surface_points))
#         dists, idx = surf_tree.query(path_proj, k=query_k)

#         if query_k == 1:
#             dists = dists[:, None]
#             idx = idx[:, None]

#         weights = np.exp(-0.5 * (dists / max(LIFT_SIGMA_M, 1e-6)) ** 2)
#         weights = weights / np.clip(np.sum(weights, axis=1, keepdims=True), LIFT_EPS, None)

#         surface_path_xyz = np.zeros((len(path_proj), 3), dtype=np.float64)
#         surface_path_nrm = np.zeros((len(path_proj), 3), dtype=np.float64)

#         for i in range(len(path_proj)):
#             nbr_xyz = self.surface_points[idx[i]]
#             nbr_nrm = self.surface_normals[idx[i]]
#             w = weights[i][:, None]

#             surface_path_xyz[i] = np.sum(w * nbr_xyz, axis=0)
#             surface_path_nrm[i] = np.sum(w * nbr_nrm, axis=0)

#         surface_path_nrm = normalize_rows(surface_path_nrm)
#         surface_path_nrm = enforce_positive_slope(surface_path_nrm, UP_AXIS)
#         surface_path_nrm = ensure_normal_continuity(surface_path_nrm)

#         # after nearest-surface lift / 3D point generation
#         t2 = time.perf_counter()

#         # ----------------------------------------------------
#         # Smooth path geometry first
#         # ----------------------------------------------------
#         surface_path_xyz = smooth_positions(surface_path_xyz, POSITION_SMOOTH_WINDOW)
#         surface_path_nrm = smooth_normals(surface_path_nrm, NORMAL_SMOOTH_WINDOW)

#         # Uniform arc-length resampling before final pose generation
#         surface_path_xyz, surface_path_nrm = resample_path_uniform(
#             surface_path_xyz,
#             surface_path_nrm,
#             INTERNAL_RESAMPLE_SPACING_M,
#         )

#         # Smooth once more after resampling
#         surface_path_xyz = smooth_positions(surface_path_xyz, POSITION_SMOOTH_WINDOW)
#         surface_path_nrm = smooth_normals(surface_path_nrm, NORMAL_SMOOTH_WINDOW)
#         surface_path_nrm = normalize_rows(surface_path_nrm)
#         surface_path_nrm = enforce_positive_slope(surface_path_nrm, UP_AXIS)
#         surface_path_nrm = ensure_normal_continuity(surface_path_nrm)

#         # ----------------------------------------------------
#         # Offset tool path above the surface
#         # ----------------------------------------------------
#         tool_positions = surface_path_xyz + SURFACE_OFFSET_M * surface_path_nrm

#         side_check = np.sum((tool_positions - surface_path_xyz) * surface_path_nrm, axis=1)
#         if np.any(side_check < 0.0):
#             self.get_logger().warn("Some tool positions are on the wrong side of the surface. Re-enforcing positive slope.")
#             surface_path_nrm = enforce_positive_slope(surface_path_nrm, UP_AXIS)
#             surface_path_nrm = ensure_normal_continuity(surface_path_nrm)
#             surface_path_nrm = normalize_rows(surface_path_nrm)
#             tool_positions = surface_path_xyz + SURFACE_OFFSET_M * surface_path_nrm

#         # Final spacing reduction for executor
#         keep = downsample_by_spacing(tool_positions, FINAL_WAYPOINT_SPACING_M)
#         surface_path_xyz = surface_path_xyz[keep]
#         surface_path_nrm = surface_path_nrm[keep]
#         tool_positions = tool_positions[keep]

#         if len(tool_positions) < 2:
#             self.get_logger().warn("Lifted path has fewer than 2 poses after processing.")
#             return

#         # after smoothing + resampling + downsampling
#         t3 = time.perf_counter()

#         # ----------------------------------------------------
#         # Smooth minimum-twist orientations
#         # ----------------------------------------------------
#         quats = self.build_minimum_twist_quaternions(
#             positions_xyz=tool_positions,
#             normals_xyz=surface_path_nrm,
#         )

#         # ----------------------------------------------------
#         # Publish Extrema for modular_executor
#         # ----------------------------------------------------
#         msg_out = Extrema()
#         msg_out.header = msg.header
#         msg_out.header.frame_id = self.surface_frame_id
#         msg_out.poses = [make_pose(p, q) for p, q in zip(tool_positions, quats)]
#         msg_out.value = 0.0

#         # after quaternion/orientation generation and message creation
#         t4 = time.perf_counter()

#         self.pose_pub.publish(msg_out)
        
#         pose_array = PoseArray()
#         pose_array.header = msg_out.header
#         pose_array.poses = [make_pose(p, q) for p, q in zip(tool_positions, quats)]
#         self.debug_pose_pub.publish(pose_array)

#         t5 = time.perf_counter()

#         self.get_logger().info(
#             f"[TIMING path_to_cart] parse={t1-t0:.3f}s "
#             f"lift={t2-t1:.3f}s "
#             f"smooth_resample={t3-t2:.3f}s "
#             f"orient_build={t4-t3:.3f}s "
#             f"publish={t5-t4:.3f}s "
#             f"total={t5-t0:.3f}s "
#             f"in_pts={len(path_proj)} out_pts={len(msg_out.poses)}"
#         )

#         self.get_logger().info(
#             f"Published {len(msg_out.poses)} smooth cartesian heater poses to {EXTREMA_PATH_TOPIC}"
#         )

#     def build_minimum_twist_quaternions(
#         self,
#         positions_xyz: np.ndarray,
#         normals_xyz: np.ndarray,
#     ) -> np.ndarray:
#         N = len(positions_xyz)
#         quats = np.zeros((N, 4), dtype=np.float64)

#         prev_x = None

#         for i in range(N):
#             n = normalize(normals_xyz[i])

#             # Tool Z follows the surface normal
#             z_tool = normalize(TOOL_AXIS_SIGN * n)

#             # Build a reference x-axis from a fixed world axis projected into the local tangent plane
#             x_ref = project_vector_to_plane(TWIST_REFERENCE_AXIS, z_tool)
#             x_ref = normalize(x_ref)

#             # Fallback if reference axis is too parallel to z_tool
#             if np.linalg.norm(x_ref) < REF_EPS:
#                 alt = np.array([0.0, 0.0, 1.0], dtype=np.float64)
#                 x_ref = project_vector_to_plane(alt, z_tool)
#                 x_ref = normalize(x_ref)

#             if np.linalg.norm(x_ref) < REF_EPS:
#                 alt = np.array([0.0, 1.0, 0.0], dtype=np.float64)
#                 x_ref = project_vector_to_plane(alt, z_tool)
#                 x_ref = normalize(x_ref)

#             if np.linalg.norm(x_ref) < REF_EPS and prev_x is not None:
#                 x_ref = project_vector_to_plane(prev_x, z_tool)
#                 x_ref = normalize(x_ref)

#             if np.linalg.norm(x_ref) < REF_EPS:
#                 # Final fallback
#                 x_ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
#                 x_ref = project_vector_to_plane(x_ref, z_tool)
#                 x_ref = normalize(x_ref)

#             # Build orthonormal tangent-plane basis
#             y_ref = normalize(np.cross(z_tool, x_ref))
#             x_ref = normalize(np.cross(y_ref, z_tool))

#             # Apply constant twist about local Z
#             c = np.cos(TOOL_TWIST_RAD)
#             s = np.sin(TOOL_TWIST_RAD)

#             x_tool = normalize(c * x_ref + s * y_ref)
#             y_tool = normalize(np.cross(z_tool, x_tool))
#             x_tool = normalize(np.cross(y_tool, z_tool))

#             # Keep sign continuity to avoid quaternion flips
#             if prev_x is not None and np.dot(x_tool, prev_x) < 0.0:
#                 x_tool *= -1.0
#                 y_tool *= -1.0

#             prev_x = x_tool.copy()

#             Rm = np.column_stack((x_tool, y_tool, z_tool))
#             quats[i] = rotation_matrix_to_quaternion(Rm)

#         quats = ensure_quat_continuity(quats)
#         return quats


# def main(args=None):
#     rclpy.init(args=args)
#     node = HeaterPathToCartesianNode()
#     try:
#         rclpy.spin(node)
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()
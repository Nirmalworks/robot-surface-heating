#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseArray, Point, Quaternion, PoseStamped
from std_msgs.msg import Header
import copy

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from moveit_msgs.srv import GetCartesianPath, GetPositionFK
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient
import tf_transformations
import tf2_ros
import tf2_geometry_msgs

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

import time

### wait for message utilities ###
from typing import Union
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec
from thermal_camera import common_motionplan_utilities
from thermal_camera_interfaces.msg import Extrema
### ### ### ### ### ### 

import signal
import threading
from typing import List

######### DEMO PATH POSE CREATION #########
# ---------- small quaternion & pose helpers ---------------------------------
def _vec(p):          # Pose  → numpy xyz
    return np.array([p.position.x, p.position.y, p.position.z], dtype=float)

def _quat(p):         # Pose  → numpy xyzw
    return np.array([p.orientation.x, p.orientation.y,
                     p.orientation.z, p.orientation.w], dtype=float)

def _to_pose(pos, quat):
    po = Pose()
    po.position.x, po.position.y, po.position.z = pos
    po.orientation.x, po.orientation.y, po.orientation.z, po.orientation.w = quat
    return po

def _slerp(q1, q2, n):
    # ensure shortest-arc interpolation
    if np.dot(q1, q2) < 0.0:
        q2 = -q2
    return Slerp([0, 1], R.from_quat([q1, q2]))(np.linspace(0, 1, n)).as_quat()


# ---------- main generator ---------------------------------------------------
def trapezoid_zigzag_endpoints_only(p0, p1, p2, p3, *,
                                   rows: int = 20,
                                   return_to_base: bool = True,
                                   row_passes: int = 1) -> list[Pose]:
    """
    Build a trapezoid raster path with only endpoints (left and right edges) per row.

    Corner order (counter-clockwise):
        p0 ───────── p1   ← large base (row 0)
         │           │
         │           │
        p3 ───────── p2   ← narrow top (row N)

    Args:
        p0..p3          : Four corner poses
        rows            : Number of horizontal sweep rows (vertical resolution)
        return_to_base  : If True, adds a zigzag back down
        row_passes      : Number of left-right or right-left repeats per row

    Returns:
        List of Pose at left and right edges only, in continuous trajectory order
    """
    poses = []

    left_edge_pos  = np.linspace(_vec(p0), _vec(p3), rows)
    right_edge_pos = np.linspace(_vec(p1), _vec(p2), rows)
    left_quats     = _slerp(_quat(p0), _quat(p3), rows)
    right_quats    = _slerp(_quat(p1), _quat(p2), rows)

    def _append_row(i, direction: bool):
        # Only two points: left and right ends
        row_pos = np.array([left_edge_pos[i], right_edge_pos[i]])
        row_quat = np.array([left_quats[i], right_quats[i]])
        if not direction:
            row_pos = row_pos[::-1]
            row_quat = row_quat[::-1]
        for p, q in zip(row_pos, row_quat):
            poses.append(_to_pose(p, q))

    # ---- Forward pass (bottom to top) ----
    for i in range(rows):
        direction = True  # start left-to-right
        for _ in range(row_passes):
            _append_row(i, direction)
            direction = not direction  # flip direction

    # ---- Return pass (top to bottom) ----
    if return_to_base:
        for i in range(rows - 2, -1, -1):
            direction = False  # start right-to-left on return
            for _ in range(row_passes):
                _append_row(i, direction)
                direction = not direction

    return poses
######### ######### ######### ######### #########

def select_trapezoid_corners_from_boundary(boundary_poses: list[Pose]) -> tuple[Pose, Pose, Pose, Pose]:
    """
    From an n-gon boundary, choose 4 endpoints for the trapezoid-style raster:
    - two lowest X (left side), two highest X (right side)
    - within each side, sort by Y ascending (bottom, then top)
    - return (p0, p1, p2, p3) matching:
        p0 ───── p1   (bottom row)
         │       │
        p3 ───── p2   (top row)
    All returned poses will have z = 0.3
    """
    assert len(boundary_poses) >= 4, "Need at least 4 boundary points"

    xs = np.array([p.position.x for p in boundary_poses], dtype=float)
    # indices of 2 smallest X (left) and 2 largest X (right)
    idx_sorted_by_x = np.argsort(xs)
    left_idxs = idx_sorted_by_x[:2]
    right_idxs = idx_sorted_by_x[-2:]

    # sort each side by Y ascending
    left_sorted = sorted(left_idxs, key=lambda i: boundary_poses[i].position.y)
    right_sorted = sorted(right_idxs, key=lambda i: boundary_poses[i].position.y)

    # extract poses
    p0 = copy.deepcopy(boundary_poses[left_sorted[0]])  # left-bottom
    p3 = copy.deepcopy(boundary_poses[left_sorted[1]])  # left-top
    p1 = copy.deepcopy(boundary_poses[right_sorted[0]]) # right-bottom
    p2 = copy.deepcopy(boundary_poses[right_sorted[1]]) # right-top

    # force z height to 0.3
    for p in (p0, p1, p2, p3):
        p.position.z = 0.3

    return p0, p1, p2, p3



def _sample_chain_xyz(chain_xyz: np.ndarray, rows: int) -> np.ndarray:
    """
    Evenly sample 'rows' points along a 3D polyline by arclength.
    chain_xyz: (M,3) points in order along the chain.
    Returns: (rows,3)
    """
    if chain_xyz.shape[0] == 1:
        return np.repeat(chain_xyz, rows, axis=0)

    seg = chain_xyz[1:] - chain_xyz[:-1]
    seg_len = np.linalg.norm(seg, axis=1)
    cum = np.concatenate(([0.0], np.cumsum(seg_len)))
    total = float(cum[-1])
    if total == 0.0:
        return np.repeat(chain_xyz[:1], rows, axis=0)

    targets = np.linspace(0.0, total, rows)
    out = np.zeros((rows, 3), dtype=float)

    j = 0
    for k, t in enumerate(targets):
        while j + 1 < len(cum) - 1 and cum[j + 1] < t:
            j += 1
        seg_d = seg_len[j]
        if seg_d > 0:
            alpha = (t - cum[j]) / seg_d
        else:
            alpha = 0.0
        out[k] = (1.0 - alpha) * chain_xyz[j] + alpha * chain_xyz[j + 1]
    return out


def _walk_indices(n: int, start: int, end: int, step: int) -> list[int]:
    """
    Return indices walking on a circular list of length n from start→end (inclusive),
    moving by 'step' (+1 forward / -1 backward).
    """
    idxs = [start]
    i = start
    while i != end:
        i = (i + step) % n
        idxs.append(i)
    return idxs


def polygon_zigzag_endpoints_only(
    boundary_poses: list[Pose],
    *,
    rows: int = 20,
    return_to_base: bool = True,
    row_passes: int = 1,
    use_constant_z: bool = True
) -> list[Pose]:
    """
    Generalization of trapezoid_zigzag_endpoints_only() for an n-sided polygon.

    - Start vertex = max X; End vertex = min X (by request).
    - Build two boundary chains between start→end (forward and backward around the list).
    - Resample each chain to 'rows' points along arclength (keeps small Z variations).
    - Orientation per chain is SLERP(start_quat→end_quat) across rows (like original).
    - If use_constant_z, overwrite all Z with start vertex's Z.
    """
    assert len(boundary_poses) >= 4, "Need at least 3 vertices"


    # Pick start (max X) and end (min X)
    xs = np.array([p.position.x for p in boundary_poses], dtype=float)
    start_idx = int(np.argmax(xs))
    end_idx   = int(np.argmin(xs))

    n = len(boundary_poses)
    # Two chains that meet at start/end, analogous to (p0→p3) and (p1→p2)
    left_idxs  = _walk_indices(n, start_idx, end_idx, +1)  # forward
    right_idxs = _walk_indices(n, start_idx, end_idx, -1)  # backward

    # Extract chains as arrays
    left_xyz  = np.array([_vec(boundary_poses[i])  for i in left_idxs],  dtype=float)
    right_xyz = np.array([_vec(boundary_poses[i])  for i in right_idxs], dtype=float)
    left_qs   = np.array([_quat(boundary_poses[i]) for i in left_idxs],  dtype=float)
    right_qs  = np.array([_quat(boundary_poses[i]) for i in right_idxs], dtype=float)

    # Resample positions along each chain (arclength)
    left_edge_pos  = _sample_chain_xyz(left_xyz,  rows)
    right_edge_pos = _sample_chain_xyz(right_xyz, rows)

    # Z handling (default: constant Z taken from starting point)
    if use_constant_z:
        z0 = boundary_poses[start_idx].position.z
        left_edge_pos[:, 2]  = z0
        right_edge_pos[:, 2] = z0

    # Chain orientation: SLERP from chain endpoints (like original)
    left_quats  = _slerp(left_qs[0],  left_qs[-1],  rows)
    right_quats = _slerp(right_qs[0], right_qs[-1], rows)

    poses: list[Pose] = []

    def _append_row(i: int, direction: bool):
        row_pos  = np.array([left_edge_pos[i],  right_edge_pos[i]])
        row_quat = np.array([left_quats[i],     right_quats[i]])
        if not direction:
            row_pos  = row_pos[::-1]
            row_quat = row_quat[::-1]
        for p, q in zip(row_pos, row_quat):
            poses.append(_to_pose(p, q))

    # Forward pass (bottom→top conceptually; we'll start left→right)
    for i in range(rows):
        direction = True
        for _ in range(row_passes):
            _append_row(i, direction)
            direction = not direction

    # Return pass
    if return_to_base:
        for i in range(rows - 2, -1, -1):
            direction = False
            for _ in range(row_passes):
                _append_row(i, direction)
                direction = not direction

    return poses



def two_edge_midpoints_by_x(
    boundary_poses: List[Pose],
    *,
    use_constant_z: bool = True,
    z_offset: float = 0.0
) -> List[Pose]:
    """
    Assumes exactly 4 boundary poses forming a rectangle (unordered).
    - First edge: the two poses with the highest X.
    - Second edge: the two poses with the lowest X.
    Returns two poses total: midpoint of max-X edge, then midpoint of min-X edge.
    Orientation at each midpoint is the SLERP midpoint between that edge’s endpoint quaternions.
    If use_constant_z, both returned Z's are set to the Z of the max-X edge's first point.
    """
    assert len(boundary_poses) == 4, "Expected exactly 4 boundary poses"

    # Collect indices sorted by X (descending)
    xs = [(i, float(p.position.x)) for i, p in enumerate(boundary_poses)]
    xs.sort(key=lambda t: t[1], reverse=True)

    # Max-X edge (first two after sorting), Min-X edge (last two)
    max_edge_idx = [xs[0][0], xs[1][0]]
    min_edge_idx = [xs[2][0], xs[3][0]]

    def _midpoint_pose(i0: int, i1: int) -> Pose:
        p0 = _vec(boundary_poses[i0])  # [x,y,z]
        p1 = _vec(boundary_poses[i1])
        q0 = _quat(boundary_poses[i0]) # [x,y,z,w]
        q1 = _quat(boundary_poses[i1])

        # Position midpoint
        mid_pos = 0.5 * (p0 + p1)

        # Orientation midpoint via SLERP; get the middle sample by asking for 3
        mid_quat = _slerp(q0, q1, 3)[1]

        return _to_pose(mid_pos, mid_quat)

    max_mid = _midpoint_pose(max_edge_idx[0], max_edge_idx[1])
    min_mid = _midpoint_pose(min_edge_idx[0], min_edge_idx[1])

    if use_constant_z:
        z0 = boundary_poses[max_edge_idx[0]].position.z
        max_mid.position.z = z0
        min_mid.position.z = z0

    # Apply z-offset
    max_mid.position.z += z_offset
    min_mid.position.z += z_offset

    return [max_mid, min_mid]


######### CARTESIAN PLANNER UTILITIES #########

def wait_for_message(
    msg_type,
    node: 'Node',
    topic: str,
    *,
    qos_profile: Union[QoSProfile, int] = 1,
    time_to_wait=-1
):
    """
    Wait for the next incoming message.

    :param msg_type: message type
    :param node: node to initialize the subscription on
    :param topic: topic name to wait for message
    :param qos_profile: QoS profile to use for the subscription
    :param time_to_wait: seconds to wait before returning
    :returns: (True, msg) if a message was successfully received, (False, None) if message
        could not be obtained or shutdown was triggered asynchronously on the context.
    """
    context = node.context
    wait_set = _rclpy.WaitSet(1, 1, 0, 0, 0, 0, context.handle)
    wait_set.clear_entities()

    sub = node.create_subscription(msg_type, topic, lambda _: None, qos_profile=qos_profile)
    try:
        wait_set.add_subscription(sub.handle)
        sigint_gc = SignalHandlerGuardCondition(context=context)
        wait_set.add_guard_condition(sigint_gc.handle)

        timeout_nsec = timeout_sec_to_nsec(time_to_wait)
        wait_set.wait(timeout_nsec)

        subs_ready = wait_set.get_ready_entities('subscription')
        guards_ready = wait_set.get_ready_entities('guard_condition')

        if guards_ready:
            if sigint_gc.handle.pointer in guards_ready:
                return False, None

        if subs_ready:
            if sub.handle.pointer in subs_ready:
                msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
                if msg_info is not None:
                    return True, msg_info[0]
    finally:
        node.destroy_subscription(sub)

    return False, None

def matrix_to_pose(transform: np.ndarray) -> Pose:
    """
    Convert a 4x4 transformation matrix to a ROS2 Pose message.
    
    Args:
        transform: 4x4 numpy array representing homogeneous transform.
        
    Returns:
        geometry_msgs.msg.Pose with position and orientation.
    """
    assert transform.shape == (4, 4), "Input must be a 4x4 matrix"

    pose = Pose()
    # Translation is last column (first 3 rows)
    pose.position.x = float(transform[0, 3])
    pose.position.y = float(transform[1, 3])
    pose.position.z = float(transform[2, 3])

    # Rotation is upper-left 3x3 matrix
    rot_matrix = transform[:3, :3]
    r = R.from_matrix(rot_matrix)
    quat = r.as_quat()  # x, y, z, w

    pose.orientation.x = float(quat[0])
    pose.orientation.y = float(quat[1])
    pose.orientation.z = float(quat[2])
    pose.orientation.w = float(quat[3])

    return pose

def retime_trajectory(traj, base_speed: float = 0.05):
    """
    Assigns time_from_start to each trajectory point based on scaled velocity
    per segment length, ensuring smooth velocity and acceleration transitions.
    """
    if not traj.points or len(traj.points) < 2:
        return

    t = 0.0
    traj.points[0].time_from_start.sec = 0
    traj.points[0].time_from_start.nanosec = 0
    n_joints = len(traj.points[0].positions)

    # Set initial velocity and acceleration to 0
    traj.points[0].velocities = [0.0] * n_joints
    traj.points[0].accelerations = [0.0] * n_joints

    for i in range(1, len(traj.points)):
        q1 = np.array(traj.points[i - 1].positions)
        q2 = np.array(traj.points[i].positions)
        distance = np.linalg.norm(q2 - q1)

        # Scale velocity with arc length, clamp to reasonable range
        speed = np.clip(distance * 10.0, base_speed, 0.8)  # dynamic scaling
        dt = distance / speed
        t += dt

        sec, nsec = divmod(t, 1.0)
        traj.points[i].time_from_start.sec = int(sec)
        traj.points[i].time_from_start.nanosec = int(nsec * 1e9)

        # Use scaled velocity
        vel = (q2 - q1) / dt
        traj.points[i].velocities = vel.tolist()

        # Estimate acceleration based on difference in velocity
        prev_vel = np.array(traj.points[i - 1].velocities)
        acc = (vel - prev_vel) / dt
        traj.points[i].accelerations = acc.tolist()

    # Set final velocity and acceleration to zero
    traj.points[-1].velocities = [0.0] * n_joints
    traj.points[-1].accelerations = [0.0] * n_joints

def stretch_trajectory_time(traj, factor: float):
    """
    Slow down (factor>1) or speed up (factor<1) the whole trajectory timing.
    Scales time_from_start, and adjusts velocities/accelerations consistently.
    """
    if factor <= 0.0 or not traj.points:
        return
    for pt in traj.points:
        t = pt.time_from_start.sec + pt.time_from_start.nanosec * 1e-9
        t *= factor
        sec, nsec = divmod(t, 1.0)
        pt.time_from_start.sec = int(sec)
        pt.time_from_start.nanosec = int(nsec * 1e9)
        if pt.velocities:
            pt.velocities = [v / factor for v in pt.velocities]
        if pt.accelerations:
            pt.accelerations = [a / (factor * factor) for a in pt.accelerations]

def compute_path_length(poses: list[Pose]) -> float:
    """Compute total linear path length of a list of poses."""
    total_length = 0.0
    for i in range(len(poses) - 1):
        p1 = np.array([poses[i].position.x, poses[i].position.y, poses[i].position.z])
        p2 = np.array([poses[i + 1].position.x, poses[i + 1].position.y, poses[i + 1].position.z])
        total_length += np.linalg.norm(p2 - p1)
    return total_length

def _pose_str(p: Pose) -> str:
    return (f"x={p.position.x:.3f}, y={p.position.y:.3f}, z={p.position.z:.3f} | "
            f"q=({p.orientation.x:.3f},{p.orientation.y:.3f},"
            f"{p.orientation.z:.3f},{p.orientation.w:.3f})")

def log_waypoint_debug(logger, poses: list[Pose], title: str = "Waypoints"):
    n = len(poses)
    if n == 0:
        logger.warn(f"{title}: (empty)")
        return
    xs = [p.position.x for p in poses]
    ys = [p.position.y for p in poses]
    zs = [p.position.z for p in poses]
    logger.info(f"{title}: count={n} | "
                f"X[{min(xs):.3f},{max(xs):.3f}] "
                f"Y[{min(ys):.3f},{max(ys):.3f}] "
                f"Z[{min(zs):.3f},{max(zs):.3f}]")

    # Print key waypoints (start, first move, mid, last-1, last)
    sample_idx = sorted(set(
        [0, 1, max(0, n//2), max(0, n-2), n-1]
    ))
    logger.warn("Key waypoints:")
    for i in sample_idx:
        logger.warn(f"  [{i:>4}] {_pose_str(poses[i])}")

    # Optionally print a few evenly spaced samples (uncomment if needed)
    # k = min(8, n)
    # idxs = np.linspace(0, n-1, k, dtype=int)
    # logger.info("Samples:")
    # for i in idxs:
    #     logger.info(f"  [{i:>4}] {_pose_str(poses[i])}")


######### ######### ######### ######### #########

class CartesianPathExecutor(Node):
    timeout_sec_ = 5.0
    delay_val = 0.5

    def __init__(self):
        super().__init__('cartesian_path_executor')

        # threading support
        self.interrupted = threading.Event()
        self.execution_done = threading.Event()

        # Parameters - adjust as needed
        self.move_group_name = 'ur10e'   # Your MoveIt move group
        self.base_frame = 'base_link'    # Robot base frame
        # self.eef_frame = 'tool0'
        # self.eef_frame = 'butane_torch'
        self.eef_frame = 'heat_gun'
        self.cartesian_path_service_name = '/compute_cartesian_path'
        self.trajectory_action_name = '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        self.joint_state_topic_ = '/joint_states'
        self.boundary_points_topic = '/selected_boundary_points'
        # self.boundary_frame_id = None
        # self.z_offset = -0.17
        self.z_offset = 0.0
        self.stretch_factor = 4.0 # 1.0 = normal speed, increase for slower robot speed
        self.boundary_poses = None
        self.active_goal = None
        self.shutting_down = threading.Event()
        self.in_emergency_homing = False

        # Set home pose
        self.home_pose = Pose(
            position=Point(x=0.73945, y=-0.29125, z=0.5),
            # orientation=Quaternion(x=0.96314, y=-0.26109, z=-0.00405, w=0.06467),
            orientation=Quaternion(x=1.0, y=0., z=-0., w=0.),
        )
        self.home_q = self.home_pose.orientation # lock orientation to this

        # FK solver client
        self.fk_client_ = self.create_client(GetPositionFK, "compute_fk")
        if not self.fk_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("FK service not available.")
            exit(1)

        # Service client to compute Cartesian path
        self.cartesian_path_client = self.create_client(GetCartesianPath, self.cartesian_path_service_name)
        if not self.cartesian_path_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error(f"Service {self.cartesian_path_service_name} not available.")
            raise RuntimeError("MoveIt Cartesian path service not available")

        # Action client to execute trajectory
        self.trajectory_client = ActionClient(self, FollowJointTrajectory, self.trajectory_action_name)
        if not self.trajectory_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error(f"Action server {self.trajectory_action_name} not available.")
            raise RuntimeError("Trajectory action server not available")
        
        self.debug_waypoints_pub = self.create_publisher(PoseArray, '/debug_midpoints', 10)
        
        # Subscribe to boundary points
        self.boundary_sub = self.create_subscription(
            PoseArray,
            self.boundary_points_topic,
            self.boundary_points_callback,
            10,
        )

        self.poses_pub = self.create_publisher(
            Extrema,
            "/policy_poses",
            # "zigzag_poses",
            10,
        )

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def publish_pose_array(self, poses, publisher):
        pose_array = PoseArray()
        pose_array.header.frame_id = "world"   # or self.base_frame if that’s what you use
        pose_array.poses = poses        

        publisher.publish(pose_array)

    def boundary_points_callback(self, msg: PoseArray):
        if not msg.poses:
            self.get_logger().warn("Received empty boundary; ignoring...")
            return
        
        # Store vertices only once
        if self.boundary_poses is None:

            self.boundary_poses = list(msg.poses)
            self.boundary_frame_id = msg.header.frame_id

            self.get_logger().info(f"Boundary frame_id: {self.boundary_frame_id}")

            # Show debug info
            self.get_logger().info(f"Received boundary with {len(self.boundary_poses)} vertices.")
            for i, pose in enumerate(self.boundary_poses):
                self.get_logger().info(
                    f"  Point {i+1}: "
                    f"x={pose.position.x:.3f}, "
                    f"y={pose.position.y:.3f}, "
                    f"z={pose.position.z:.3f}"
                )

            # Transform to world frame and add offset
            self.t = common_motionplan_utilities.get_transform_frame(
                self,
                self.tf_buffer,
                self.boundary_frame_id,
                'world'
            )
            
            self.boundary_poses_offset = []

            # Precompute the -90° about X rotation once (local-frame tilt)
            rx_neg_90 = R.from_euler('x', 90, degrees=True)
            for pose in self.boundary_poses:
                p_off = common_motionplan_utilities.get_tool_offset_pose(
                    self,
                    pose,
                    self.z_offset,
                    self.boundary_frame_id,
                    self.tf_buffer,
                    transform=self.t
                )

                # rotate orientation by -90° about *its own local X axis*
                r_cur = R.from_quat([p_off.orientation.x, p_off.orientation.y, p_off.orientation.z, p_off.orientation.w])
                r_new = r_cur * rx_neg_90          # local-axis rotation (post-multiply)
                qx, qy, qz, qw = r_new.as_quat()
                p_off.orientation = Quaternion(x=qx, y=qy, z=qz, w=qw)

                self.boundary_poses_offset.append(p_off)

            # Show debug info
            self.get_logger().info(f"Transformed {len(self.boundary_poses)} vertices.")
            for i, pose in enumerate(self.boundary_poses_offset):
                self.get_logger().info(
                    f"  Point {i+1}: "
                    f"x={pose.position.x:.3f}, "
                    f"y={pose.position.y:.3f}, "
                    f"z={pose.position.z:.3f}"
                )


    def get_joint_state(self):
        """Get current joint state (adapted from your robot interface)"""
        current_joint_state_set, current_joint_state = wait_for_message(
            JointState, self, self.joint_state_topic_, time_to_wait=2.0
        )
        if not current_joint_state_set:
            self.get_logger().error("Failed to get current joint state")
            return None
        
        return current_joint_state

    def get_fk(self) -> np.ndarray | None:
        current_joint_state = self.get_joint_state()
        if current_joint_state is None:
            self.get_logger().error("Failed to get current joint state")
            return None

        current_robot_state = RobotState()
        current_robot_state.joint_state = current_joint_state

        request = GetPositionFK.Request()

        request.header.frame_id = self.base_frame
        request.header.stamp = self.get_clock().now().to_msg()

        request.fk_link_names.append(self.eef_frame)
        request.robot_state = current_robot_state

        future = self.fk_client_.call_async(request)

        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error("Failed to get FK solution")
            return None

        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"Failed to get FK solution: {response.error_code.val}"
            )
            return None
        
        pose = response.pose_stamped[0].pose
        T = np.eye(4)
        T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        quat = [pose.orientation.x, pose.orientation.y, 
               pose.orientation.z, pose.orientation.w]
        T[:3, :3] = R.from_quat(quat).as_matrix()
        
        self.get_logger().info(f"EEF pose: [{pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}]")
        return T

    def spin_until_complete_or_interrupt(self, future, check_interval=0.1) -> bool:
        """
        Spins the node while waiting for a future to complete, allowing SIGINT-based interruption.

        Returns True if future completed successfully, False if interrupted.
        """
        while not future.done():
            if self.interrupted:
                self.get_logger().warn("Interrupted before future completed.")
                return False
            rclpy.spin_once(self, timeout_sec=check_interval)
        return True

    def interpolate_poses(self, poses: list[Pose], steps_per_segment: int = 10) -> list[Pose]:
        """
        Linearly interpolate positions and SLERP orientations between consecutive poses.
        
        Args:
            poses: List of geometry_msgs.msg.Pose (at least 2)
            steps_per_segment: Number of interpolation steps between each pair

        Returns:
            List of interpolated Pose waypoints
        """
        interpolated_poses = []

        for i in range(len(poses) - 1):
            start = poses[i]
            end = poses[i + 1]

            # Interpolate position
            start_pos = np.array([start.position.x, start.position.y, start.position.z])
            end_pos = np.array([end.position.x, end.position.y, end.position.z])

            # Interpolate orientation using SLERP
            start_quat = [start.orientation.x, start.orientation.y, start.orientation.z, start.orientation.w]
            end_quat = [end.orientation.x, end.orientation.y, end.orientation.z, end.orientation.w]

            key_rots = R.from_quat([start_quat, end_quat])
            key_times = [0, 1]
            if np.dot(start_quat, end_quat) < 0.0:
                end_quat = [-q for q in end_quat]
            slerp = Slerp(key_times, key_rots)
            interp_times = np.linspace(0, 1, steps_per_segment)
            interp_rots = slerp(interp_times).as_quat()  # shape: (N, 4)

            for t, quat in zip(interp_times, interp_rots):
                interp_pos = (1 - t) * start_pos + t * end_pos

                p = Pose()
                p.position.x, p.position.y, p.position.z = interp_pos
                p.orientation.x, p.orientation.y, p.orientation.z, p.orientation.w = quat

                interpolated_poses.append(p)

        return interpolated_poses
    

    def go_home_immediately(self):
        """Cancel any running goal and execute a direct homing path safely."""

        # 1) Cancel an active trajectory goal if present
        if getattr(self, "active_goal", None):
            try: self.active_goal.cancel_goal_async()
            except: pass

        # 2) Try to start homing from current FK; fall back to home_pose if FK is unavailable
        start_pose = None
        try:
            fk_T = self.get_fk()
            if fk_T is not None:
                start_pose = matrix_to_pose(fk_T)
        except Exception as e:
            self.get_logger().warn(f"FK during homing failed: {e}")

        if start_pose is None:
            self.get_logger().warn("FK unavailable during shutdown; homing with fallback")
            start_pose = self.home_pose

        # 3) Prevent normal shutdown cancelation logic from interrupting this homing   
        self.in_emergency_homing = True
        try:
            self.compute_and_execute_cartesian_path([start_pose, self.home_pose], stretch_override=1.0)
        finally:
            self.in_emergency_homing = False


    def compute_and_execute_cartesian_path(self, poses, *, stretch_override: float | None = None,):
        """
        Given a list of poses, compute Cartesian path and execute it.
        """

        if len(poses) < 2:
            self.get_logger().error("Need at least two poses for Cartesian path planning")
            return

        path_length = compute_path_length(poses)

        # Adaptive interpolation
        base_steps = 5
        scaling_factor = 20
        steps_per_segment = int(np.clip(base_steps + scaling_factor * path_length, 5, 50))

        # Adaptive velocity: slower for shorter paths
        # # You can tune min/max values depending on robot characteristics
        # # min_speed = 0.02   # [rad/s] or units appropriate to your joint scale
        # min_speed = 0.2   # [rad/s] or units appropriate to your joint scale
        # # min_speed = 0.5   # [rad/s] or units appropriate to your joint scale
        # # max_speed = 0.5
        # max_speed = 0.25

        min_speed = 0.001
        max_speed = 0.002
        velocity_scaling = min_speed + (max_speed - min_speed) * min(path_length / 1.0, 1.0)

        self.get_logger().info(f"Path length: {path_length:.3f} m → steps_per_segment={steps_per_segment}, speed={velocity_scaling:.3f}")

        waypoints = self.interpolate_poses(poses, steps_per_segment=steps_per_segment)

        # Publish debug PoseArray
        pose_array_pub = self.create_publisher(PoseArray, '/debug_cartesian_waypoints', 10)
        pose_array = PoseArray()
        pose_array.header.frame_id = self.base_frame
        pose_array.poses = waypoints
        pose_array_pub.publish(pose_array)

        # Create Cartesian path request
        request = GetCartesianPath.Request()
        request.header.frame_id = self.base_frame
        request.header.stamp = self.get_clock().now().to_msg()
        request.group_name = self.move_group_name
        request.waypoints = waypoints
        request.max_step = 0.02
        request.jump_threshold = 0.0

        future = self.cartesian_path_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
        # future = self.cartesian_path_client.call_async(request)
        # if not self.spin_until_complete_or_interrupt(future):
        #     return

        if future.result() is None or future.result().error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error("Cartesian path planning failed.")
            return

        traj = future.result().solution.joint_trajectory
        self.get_logger().info(f"Computed {len(traj.points)} trajectory points")

        # Adaptive velocity retiming
        # retime_trajectory(traj, speed=velocity_scaling)
        retime_trajectory(traj, base_speed=velocity_scaling)
        
        if stretch_override is not None:
            stretch_factor = stretch_override
        else:
            # default behavior
            stretch_factor = self.stretch_factor
        
        stretch_trajectory_time(traj, factor=stretch_factor) 

        traj_goal = FollowJointTrajectory.Goal()
        traj_goal.trajectory = traj

        send_goal_future = self.trajectory_client.send_goal_async(traj_goal)
        rclpy.spin_until_future_complete(self, send_goal_future)

        # send_goal_future = self.trajectory_client.send_goal_async(traj_goal)
        # if not self.spin_until_complete_or_interrupt(send_goal_future):
        #     return

        goal_handle = send_goal_future.result()
        self.active_goal = goal_handle
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            return None

        get_result_future = goal_handle.get_result_async()
        # rclpy.spin_until_future_complete(self, get_result_future)

        try:
            while not get_result_future.done():
                rclpy.spin_once(self, timeout_sec=0.05)
                # Only cancel automatically on shutdown if we're NOT in emergency homing
                if self.shutting_down.is_set() and not getattr(self, "in_emergency_homing", False):
                    try:
                        goal_handle.cancel_goal_async()
                        self.get_logger().warn("Shutdown detected → cancel sent")
                    except Exception as e:
                        self.get_logger().error(f"Cancel failed: {e}")
                    # exit immediately on shutdown; caller can send homing
                    return "canceled"
        finally:
            self.active_goal = None

        # get_result_future = goal_handle.get_result_async()
        # if not self.spin_until_complete_or_interrupt(get_result_future):
        #     return

        self.get_logger().info("Trajectory execution complete.")
        # time.sleep(self.delay_val)
        if not self.shutting_down.is_set():
                time.sleep(velocity_scaling)

        return get_result_future

    def run_demo(self):

        # Wait for polygon
        while rclpy.ok() and self.boundary_poses is None:
            self.get_logger().info("Waiting for 'selected_boundary_points' ...")
            rclpy.spin_once(self, timeout_sec=0.2)

        NUM_HEATING_PASSES = 1
        NUM_HEATING_X_DELIMS = 10

        # # derive corners from the polygon
        # c0, c1, c2, c3 = select_trapezoid_corners_from_boundary(self.boundary_poses_offset)

        # self.get_logger().info(
        #     f"Corners -> "
        #     f"p0(LB): [{c0.position.x:.3f}, {c0.position.y:.3f}, {c0.position.z:.3f}]  | "
        #     f"p1(RB): [{c1.position.x:.3f}, {c1.position.y:.3f}, {c1.position.z:.3f}]  | "
        #     f"p2(RT): [{c2.position.x:.3f}, {c2.position.y:.3f}, {c2.position.z:.3f}]  | "
        #     f"p3(LT): [{c3.position.x:.3f}, {c3.position.y:.3f}, {c3.position.z:.3f}]"
        # )

        # climbing_poses = trapezoid_zigzag_endpoints_only(
        #     c1, c2, c3, c0,
        #     rows=NUM_HEATING_X_DELIMS,
        #     return_to_base=True,
        #     row_passes=NUM_HEATING_PASSES,
        # )

        climbing_poses = polygon_zigzag_endpoints_only(
            self.boundary_poses_offset,
            rows=NUM_HEATING_X_DELIMS,
            return_to_base=True,
            row_passes=NUM_HEATING_PASSES,
        )

        def to_Extrema(poses) -> Extrema:
            msg = Extrema()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = 'world'
            msg.poses = list(poses)
            msg.value = float(len(poses))

            return msg

        midpoint_poses_AB = two_edge_midpoints_by_x(boundary_poses=self.boundary_poses_offset)
        midpoint_poses_CD = two_edge_midpoints_by_x(boundary_poses=self.boundary_poses_offset, z_offset=0.1)
        # self.publish_pose_array(midpoint_poses, self.debug_waypoints_pub)

        # Store poses in Extrema msg for publishing
        zigzag_msg = Extrema()
        zigzag_msg.header.stamp = self.get_clock().now().to_msg()
        zigzag_msg.header.frame_id = 'world'
        zigzag_msg.poses = list(climbing_poses)
        zigzag_msg.value = float(len(climbing_poses)) # store number of poses in list

        home_msg = Extrema()
        home_msg.header.stamp = self.get_clock().now().to_msg()
        home_msg.header.frame_id = 'world'
        home_msg.poses = [self.home_pose]
        home_msg.value = 1.0

        point_A_pose = midpoint_poses_AB[0]
        point_B_pose = midpoint_poses_AB[1]
        point_C_pose = midpoint_poses_CD[0]
        point_D_pose = midpoint_poses_CD[1]

        point_A_msg = to_Extrema([point_A_pose])
        point_B_msg = to_Extrema([point_B_pose])
        point_C_msg = to_Extrema([point_C_pose])
        point_D_msg = to_Extrema([point_D_pose])
        midpoint_msg_AB = to_Extrema(midpoint_poses_AB)

        self.shutting_down = threading.Event()

        # def _sig_handler(signum, frame):
        #     self.get_logger().warn(f"Signal {signum} → emergency homing")
        #     self.shutting_down.set()

        # signal.signal(signal.SIGINT, _sig_handler)
        # signal.signal(signal.SIGTERM, _sig_handler)  # Also handle SIGTERM

        def heating_thread():
            try:
                time.sleep(1)
                # # move to initial heating position (FK -> first raster point)
                # start_fk = matrix_to_pose(self.get_fk())
                # first_wp = climbing_poses[0]
                # self.compute_and_execute_cartesian_path([start_fk, first_wp], stretch_override=2.0)
                # # self.compute_and_execute_cartesian_path([start_fk, self.home_pose])

                # heat_gun_on_off_delay = 0.1
                # self.get_logger().info(f"Pausing for {heat_gun_on_off_delay} seconds. Enable heat gun.")
                # time.sleep(heat_gun_on_off_delay)
                # self.get_logger().info(f"Starting heating path.")

                # heating path loop
                # while not (self.interrupted.is_set() or self.shutting_down.is_set()):
                    # self.get_logger().info("Publishing poses...")

                
                # self.poses_pub.publish(zigzag_msg)
                # self.get_logger().info("Zigzag poses sent...")
                # time.sleep(10)
                # while(1):
                #     self.poses_pub.publish(home_msg)
                #     self.get_logger().info("Home pose sent...")    
                #     time.sleep(0.2)         
                
                # Alternate A/B 
                # alternator = True

                # while(1):

                #     if alternator:
                #         self.poses_pub.publish(point_A_msg)
                #         self.publish_pose_array(point_A_msg.poses, self.debug_waypoints_pub)    
                #         alternator = False
                #         self.get_logger().info("Publishing point A pose...")
                #     else:
                #         self.poses_pub.publish(point_B_msg)
                #         self.publish_pose_array(point_B_msg.poses, self.debug_waypoints_pub)    
                #         alternator = True
                #         self.get_logger().info("Publishing point B pose...")

                #     time.sleep(4.0)

                self.poses_pub.publish(point_A_msg)
                self.publish_pose_array(point_A_msg.poses, self.debug_waypoints_pub)    
                self.get_logger().info("Publishing point A pose...")

                time.sleep(6.5)

                self.poses_pub.publish(point_B_msg)
                self.publish_pose_array(point_B_msg.poses, self.debug_waypoints_pub)    
                self.get_logger().info("Publishing point B pose...")
                
                time.sleep(1.75)

                self.poses_pub.publish(point_D_msg)
                self.publish_pose_array(point_D_msg.poses, self.debug_waypoints_pub)    
                self.get_logger().info("Publishing point D pose...")

                # self.poses_pub.publish(point_C_msg)
                # self.publish_pose_array(point_C_msg.poses, self.debug_waypoints_pub)    
                # self.get_logger().info("Publishing point C pose...")

                time.sleep(5.0)

                # self.poses_pub.publish(point_A_msg)
                # self.publish_pose_array(point_A_msg.poses, self.debug_waypoints_pub)    
                # self.get_logger().info("Publishing point A pose...")

                # time.sleep(6.5)

                # self.poses_pub.publish(point_B_msg)
                # self.publish_pose_array(point_B_msg.poses, self.debug_waypoints_pub)    
                # self.get_logger().info("Publishing point B pose...")
                
                # time.sleep(1.75)

                # self.poses_pub.publish(point_C_msg)
                # self.publish_pose_array(point_C_msg.poses, self.debug_waypoints_pub)    
                # self.get_logger().info("Publishing point C pose...")

                #     result_future = self.compute_and_execute_cartesian_path(climbing_poses)
                #     if self.interrupted.is_set():
                #         self.get_logger().info("Exiting heating path...")
                #         break

                # # end of loop cleanup (only if NOT shutting down)
                # if not self.shutting_down.is_set():
                #     self.get_logger().info(f"Pausing for {heat_gun_on_off_delay} seconds. Disable heat gun.")
                #     time.sleep(heat_gun_on_off_delay)
                #     self.get_logger().info("Returning to home pose...")
                #     self.compute_and_execute_cartesian_path([climbing_poses[-1], self.home_pose])
                
            except Exception as e:
                self.get_logger().error(f"Heating task error: {e}")
            # finally:
            #     self.execution_done.set()
            finally:
                if self.shutting_down.is_set():
                    try:
                        # self.go_home_immediately()
                        pass
                    except Exception as e:
                        self.get_logger().error(f"Homing on shutdown failed: {e}")
                self.execution_done.set()  # let main() handle destroy/shutdown

        self.heating_thread = threading.Thread(target=heating_thread)
        self.heating_thread.start()


def main(args=None):
    # rclpy.init(args=args)
    # node = CartesianPathExecutor()
    # try:
    #     node.run_demo()
    # finally:
    #     node.destroy_node()
    #     rclpy.shutdown()

    rclpy.init(args=args)
    node = CartesianPathExecutor()
    try:
        node.run_demo()  # Starts the heating thread
        while rclpy.ok() and not node.execution_done.is_set():
            rclpy.spin_once(node, timeout_sec=0.1)
        node.heating_thread.join()
    finally:
        node.get_logger().info("Shutting down node.")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
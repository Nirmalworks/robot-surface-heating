#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from geometry_msgs.msg import Pose, PoseArray
from std_msgs.msg import Header

from moveit_msgs.srv import GetCartesianPath, GetPositionFK
from moveit_msgs.msg import MoveItErrorCodes, RobotState
from sensor_msgs.msg import JointState
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient

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
### ### ### ### ### ### 

import signal
import threading

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
def trapezoid_zigzag(p0, p1, p2, p3, *,
                     rows: int = 20,
                     base_cols: int = 10,
                     return_to_base: bool = True,
                     row_passes: int = 1) -> list[Pose]:
    """
    Build a trapezoid raster path with optional zigzag passes per row.

    Corner order (counter-clockwise):
        p0 ───────── p1   ← large base (row 0)
         │           │
         │           │
        p3 ───────── p2   ← narrow top (row N)

    Args:
        p0..p3          : Four corner poses
        rows            : Number of horizontal sweep rows (vertical resolution)
        base_cols       : Number of poses on the widest row
        return_to_base  : If True, adds a zigzag back down
        row_passes      : Number of left-right or right-left repeats per row

    Returns:
        List of Pose in a continuous trajectory
    """
    poses = []

    left_edge_pos  = np.linspace(_vec(p0), _vec(p3), rows)
    right_edge_pos = np.linspace(_vec(p1), _vec(p2), rows)
    left_quats     = _slerp(_quat(p0), _quat(p3), rows)
    right_quats    = _slerp(_quat(p1), _quat(p2), rows)

    def _append_row(i, direction: bool):
        cols = max(2, int(round(base_cols * (1 - i / (rows - 1)))))
        row_pos  = np.linspace(left_edge_pos[i], right_edge_pos[i], cols)
        row_quat = _slerp(left_quats[i], right_quats[i], cols)
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

# def retime_trajectory(traj, speed: float = 0.05):
#     """
#     Assigns time_from_start to each trajectory point to enforce constant joint-space speed.
#     """
#     if not traj.points:
#         return

#     t = 0.0
#     traj.points[0].time_from_start.sec = 0
#     traj.points[0].time_from_start.nanosec = 0
#     n_joints = len(traj.points[0].positions)

#     for i in range(1, len(traj.points)):
#         q1 = np.array(traj.points[i - 1].positions)
#         q2 = np.array(traj.points[i].positions)
#         distance = np.linalg.norm(q2 - q1)
#         dt = distance / max(speed, 1e-3)
#         t += dt

#         sec, nsec = divmod(t, 1.0)
#         traj.points[i].time_from_start.sec = int(sec)
#         traj.points[i].time_from_start.nanosec = int(nsec * 1e9)

#         # Use constant velocity approximation
#         vel = (q2 - q1) / dt
#         traj.points[i].velocities = vel.tolist()

#         # Optional: set acceleration to 0 (or compute real acceleration)
#         traj.points[i].accelerations = [0.0] * n_joints

#     # Final point: zero velocity and acceleration
#     traj.points[-1].velocities = [0.0] * n_joints
#     traj.points[-1].accelerations = [0.0] * n_joints

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

def compute_path_length(poses: list[Pose]) -> float:
    """Compute total linear path length of a list of poses."""
    total_length = 0.0
    for i in range(len(poses) - 1):
        p1 = np.array([poses[i].position.x, poses[i].position.y, poses[i].position.z])
        p2 = np.array([poses[i + 1].position.x, poses[i + 1].position.y, poses[i + 1].position.z])
        total_length += np.linalg.norm(p2 - p1)
    return total_length

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
        self.eef_frame = 'tool0'
        self.cartesian_path_service_name = '/compute_cartesian_path'
        self.trajectory_action_name = '/scaled_joint_trajectory_controller/follow_joint_trajectory'
        self.joint_state_topic_ = '/joint_states'

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

    def get_joint_state(self):
        """Get current joint state (adapted from your robot interface)"""
        current_joint_state_set, current_joint_state = wait_for_message(
            JointState, self, self.joint_state_topic_, time_to_wait=1.0
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

    def compute_and_execute_cartesian_path(self, poses):
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

        min_speed = 0.1
        max_speed = 0.2
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

        traj_goal = FollowJointTrajectory.Goal()
        traj_goal.trajectory = traj

        send_goal_future = self.trajectory_client.send_goal_async(traj_goal)
        rclpy.spin_until_future_complete(self, send_goal_future)

        # send_goal_future = self.trajectory_client.send_goal_async(traj_goal)
        # if not self.spin_until_complete_or_interrupt(send_goal_future):
        #     return

        goal_handle = send_goal_future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Trajectory goal rejected")
            return None

        get_result_future = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_result_future)

        # get_result_future = goal_handle.get_result_async()
        # if not self.spin_until_complete_or_interrupt(get_result_future):
        #     return

        self.get_logger().info("Trajectory execution complete.")
        # time.sleep(self.delay_val)
        time.sleep(velocity_scaling)
        return get_result_future

    def run_demo(self):

        # self.interrupted = False
        # self._interrupted_event = threading.Event()
        # self.execution_done = threading.Event()

        # def sigint_handler(signum, frame):
        #     self.get_logger().info("SIGINT received. Finishing current pass before shutdown...")
        #     self.interrupted = True
        #     # self._interrupted_event.set()

        # signal.signal(signal.SIGINT, sigint_handler)


        ######### DEMO BOUNDS #########
        # home
        pose_home = matrix_to_pose(np.array(
            [[ 0.40931665, -0.46981442,  0.78213445,  0.44616228],
            [-0.77694431, -0.62890906,  0.02882586, -0.13963888],
            [ 0.47834864, -0.61947381, -0.62244259,  1.03225611],
            [ 0.,          0.,          0.,          1.,        ]]
        ))

        ###### General Part Space ######

        # # min x min y (full part)
        # pose_minx_miny = matrix_to_pose(np.array(
        #     [[ 0.6550381,  -0.7555125,   0.01122299,  0.47987661],
        #     [-0.75464812, -0.65488802, -0.04034726, -0.46829699],
        #     [ 0.03783266,  0.01795958, -0.99912269,  0.416576  ],
        #     [ 0.,          0.,          0.,          1.        ]]
        # ))

        # # min x max y (full part)
        # pose_minx_maxy = matrix_to_pose(np.array(
        #     [[ 0.65541505, -0.75519219,  0.01076417,  0.45955768],
        #     [-0.75434105, -0.65524961, -0.0402185,  -0.13834646],
        #     [ 0.03742591,  0.01823996, -0.99913293,  0.41655557],
        #     [ 0.,          0.,          0.,          1.        ]]
        # ))

        # # min x min y (local region)
        # pose_minx_miny = matrix_to_pose(np.array(
        #     [[ 0.65472865, -0.7557862,   0.01084484, 0.81871899],
        #     [-0.75491376, -0.65455744, -0.04074029, -0.41807049],
        #     [ 0.03788952,  0.01848691, -0.99911091,  0.41658756],
        #     [ 0.,          0.,          0.,          1.        ]]
        # ))

        # # min x max y (local region)
        # pose_minx_maxy = matrix_to_pose(np.array(
        #     [[ 0.65483036, -0.75570025,  0.01069293,  0.80335758],
        #     [-0.75483097, -0.65465345, -0.04073168, -0.17293875],
        #     [ 0.0377811,   0.01860099, -0.9991129,   0.41655392],
        #     [ 0.,          0.,          0.,          1.        ]]
        # ))

        # # max x min y (black part)
        # pose_maxx_miny = matrix_to_pose(np.array(
        #     [[ 0.65449136, -0.75599302,  0.01075253,  0.98811647],
        #     [-0.7551145,  -0.65431314, -0.04094403, -0.39304256],
        #     [ 0.03798892,  0.01867813, -0.99910358,  0.41662144],
        #     [ 0.,          0.,          0.,          1.        ]]
        # ))

        # # max x max y (black part)
        # pose_maxx_maxy = matrix_to_pose(np.array(
        #     [[ 0.65450668, -0.75598059,  0.01069345,  0.97525838],
        #     [-0.75510234, -0.65432579, -0.0409661,  -0.19023055],
        #     [ 0.03796657,  0.01873793, -0.99910331,  0.41658741],
        #     [ 0.,          0.,          0.,          1.        ]]
        # ))

        # # max x min y (white part)
        # pose_maxx_miny = matrix_to_pose(np.array(
        #     [[ 0.65427522, -0.75617887,  0.01083793,  1.13168337],
        #     [-0.75529121, -0.65409683, -0.04114033, -0.39598511],
        #     [ 0.0381985,   0.01873131, -0.9990946,   0.4166503 ],
        #     [ 0.,          0. ,         0.,          1.        ]]
        # ))

        # # max x max y (white part)
        # pose_maxx_maxy = matrix_to_pose(np.array(
        #     [[ 0.65435863, -0.75610883,  0.01068772,  1.11802494],
        #     [-0.75522212, -0.65417347, -0.04119004, -0.18107452],
        #     [ 0.03813578,  0.01888145, -0.99909417,  0.41662394],
        #     [ 0.,          0.,          0.,          1.        ]]
        # ))

        ###### ###### ###### ###### ######

        ###### Single Face Part Space ######

        # min x min y (local region)
        pose_minx_miny = matrix_to_pose(np.array(
            [[ 0.6547132,  -0.75545014,  0.02541083,  0.39252196],
            [-0.75446062, -0.65517399, -0.0391946 , -0.3922955 ],
            [ 0.04625808,  0.00648975, -0.99890844,  0.41496275],
            [ 0.        ,  0.        ,  0.        ,  1.        ]]
        ))

        # min x max y (local region)
        pose_minx_maxy = matrix_to_pose(np.array(
            [[ 0.65496422, -0.75524338 , 0.02508583,  0.38244983],
            [-0.75425928, -0.65541019, -0.03912061, -0.21994859],
            [ 0.04598709,  0.00670138, -0.99891956,  0.41493384],
            [ 0.        ,  0.        ,  0.        ,  1.        ]]
        ))

        # max x min y (black part)
        pose_maxx_miny = matrix_to_pose(np.array(
            [[ 0.65389061, -0.75617536,  0.02501794,  1.02433643],
            [-0.75514438, -0.65432759, -0.04015438, -0.37984009],
            [ 0.04673368,  0.00736442, -0.99888024,  0.41510502],
            [ 0.        ,  0.        ,  0.        ,  1.        ]]
        ))

        # max x max y (black part)
        pose_maxx_maxy = matrix_to_pose(np.array(
            [[ 0.65398345, -0.75610267,  0.02478702,  1.0160745 ],
            [-0.7550766 , -0.65441005, -0.04008517, -0.24953834],
            [ 0.04652937,  0.00749894, -0.99888877,  0.41499778],
            [ 0.        ,  0.        ,  0.        ,  1.        ]]
        ))

        ###### ###### ###### ###### ######


        # NUM_HEATING_PASSES = 8
        NUM_HEATING_PASSES = 1
        NUM_HEATING_X_DELIMS = 10
        climbing_poses = trapezoid_zigzag_endpoints_only(
            pose_minx_miny, 
            pose_minx_maxy,
            pose_maxx_maxy,
            pose_maxx_miny,
            rows=NUM_HEATING_X_DELIMS,
            return_to_base=True,
            row_passes=NUM_HEATING_PASSES,
        )

        # self.compute_and_execute_cartesian_path([matrix_to_pose(self.get_fk()), pose_minx_miny])
        # heat_gun_on_off_delay = 4
        # self.get_logger().info(f"Pausing for {heat_gun_on_off_delay} seconds. Enable heat gun.")
        # time.sleep(heat_gun_on_off_delay)
        # self.get_logger().info(f"Starting heating path.")

        # # heating path loop
        # try:
        #     while not self.interrupted:
        #         # for idx in range(len(climbing_poses)-1):
        #         #     self.compute_and_execute_cartesian_path([climbing_poses[idx], climbing_poses[idx+1]])
        #         #     if self.interrupted:
        #         #         # Ctrl-C sets interrupt
        #         #         self.get_logger().info("Exiting heating path...")
        #         #         break  # exit after finishing current segment
        
        #         result_future = self.compute_and_execute_cartesian_path(climbing_poses)
        #         if self.interrupted:
        #             # Ctrl-C sets interrupt
        #             self.get_logger().info("Exiting heating path...")
        #             if result_future:
        #                 rclpy.spin_until_future_complete(self, result_future)
        #             break  # exit after finishing current segment
        # except Exception as e:
        #     self.get_logger().error(f"Exception during execution: {e}")
        # finally:
        #     # return to home pose after loop exit
        #     self.get_logger().info(f"Pausing for {heat_gun_on_off_delay} seconds. Disable heat gun.")
        #     time.sleep(heat_gun_on_off_delay)
        #     self.get_logger().info("Returning to home pose...")
        #     result_future = self.compute_and_execute_cartesian_path([climbing_poses[-1], pose_home])
        #     if result_future:
        #         rclpy.spin_until_future_complete(self, result_future)
        ######### ######### ######### #########

        def _sig_handler(signum, frame):
            self.get_logger().warn(f"Signal {signum} caught — requesting shutdown")
            self.interrupted.set()

        signal.signal(signal.SIGINT, _sig_handler)
        signal.signal(signal.SIGTERM, _sig_handler)  # Also handle SIGTERM

        def heating_thread():
            try:
                # move to initial heating position
                self.compute_and_execute_cartesian_path([matrix_to_pose(self.get_fk()), pose_minx_miny])
                heat_gun_on_off_delay = 4
                self.get_logger().info(f"Pausing for {heat_gun_on_off_delay} seconds. Enable heat gun.")
                time.sleep(heat_gun_on_off_delay)
                self.get_logger().info(f"Starting heating path.")

                # heating path loop
                while not self.interrupted.is_set():
                    result_future = self.compute_and_execute_cartesian_path(climbing_poses)
                    if self.interrupted.is_set():
                        self.get_logger().info("Exiting heating path...")
                        break
                    # if self.interrupted and result_future:
                    #     # Ctrl-C sets interrupt
                    #     rclpy.spin_until_future_complete(self, result_future)
                    #     break  # exit after finishing current segment

                # end of loop cleanup
                self.get_logger().info(f"Pausing for {heat_gun_on_off_delay} seconds. Disable heat gun.")
                time.sleep(heat_gun_on_off_delay)
                self.get_logger().info("Returning to home pose...")
                self.compute_and_execute_cartesian_path([climbing_poses[-1], pose_home])
            except Exception as e:
                self.get_logger().error(f"Heating task error: {e}")
            finally:
                self.execution_done.set()
        
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
import rclpy
import numpy as np

from rclpy.node import Node
from rclpy.client import Client
from moveit_msgs.srv import GetPositionIK, GetPositionFK
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose, PoseStamped, Point, Quaternion
from moveit_msgs.msg import MoveItErrorCodes, RobotState
import tf2_ros
import tf2_geometry_msgs
import asyncio
from scipy.spatial.transform import Rotation as R

from typing import Union
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.qos import QoSProfile
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec

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

def get_joint_state(node: Node, joint_state_topic: str) -> JointState:
    current_joint_state_set, current_joint_state = wait_for_message(
        JointState, node, joint_state_topic, time_to_wait=1.0
    )
    if not current_joint_state_set:
        node.get_logger().error("Failed to get current joint state")
        return None
    
    return current_joint_state

def sum_of_square_diff(joint_state_1: JointState, joint_state_2: JointState) -> float:
    return np.sum(
        np.square(np.subtract(joint_state_1.position, joint_state_2.position))
    )

def get_ik(
    node: Node, 
    target_pose: Pose,
    move_group_name: str,
    base_frame: str,
    ik_client: Client,
    verbose: bool = False,
) -> JointState | None:
    request = GetPositionIK.Request()

    request.ik_request.group_name = move_group_name
    request.ik_request.pose_stamped.header.frame_id = f"{base_frame}"
    request.ik_request.pose_stamped.header.stamp = node.get_clock().now().to_msg()
    request.ik_request.pose_stamped.pose = target_pose
    request.ik_request.avoid_collisions = True

    future = ik_client.call_async(request)

    rclpy.spin_until_future_complete(node, future)
    if future.result() is None:
        node.get_logger().error("Failed to get IK solution")
        return None

    response = future.result()
    if response.error_code.val != MoveItErrorCodes.SUCCESS:
        return None

    return response.solution.joint_state

def get_fk(
    node: Node,
    base_frame: str,
    eef_link_name: str,
    joint_state_topic: str,
    fk_client: Client,
    verbose: bool = False,
) -> Pose | None:
    if verbose:
        node.get_logger().info("start of fk")
    for _ in range(10):
        current_joint_state = get_joint_state(node, joint_state_topic)
        if current_joint_state is not None:
            break
    if current_joint_state is None:
        if verbose:
            node.get_logger().error("Failed to get current joint state")
        return None
    if verbose:
        node.get_logger().info("got joint state in fk")

    current_robot_state = RobotState()
    current_robot_state.joint_state = current_joint_state

    request = GetPositionFK.Request()

    request.header.frame_id = f"{base_frame}"
    request.header.stamp = node.get_clock().now().to_msg()

    request.fk_link_names.append(eef_link_name)
    request.robot_state = current_robot_state

    future = fk_client.call_async(request)
    if verbose:
        node.get_logger().info("calling for fk")

    rclpy.spin_until_future_complete(node, future)
    if future.result() is None:
        if verbose:
            node.get_logger().error("Failed to get FK solution")
        return None
    
    response = future.result()
    if response.error_code.val != MoveItErrorCodes.SUCCESS:
        if verbose:
            node.get_logger().error(
                f"Failed to get FK solution: {response.error_code.val}"
            )
        return None
    if verbose:
        node.get_logger().info("got fk")
    return response.pose_stamped[0].pose

def get_best_ik(
    node: Node,
    target_pose: Pose,
    joint_state_topic: str,
    move_group_name: str,
    base_frame: str,
    ik_client: Client,
    attempts: int = 10,
    verbose: bool = False,
) -> JointState | None:
    """Sample IK multiple times, taking the joint state
    with the closest pose to the target pose."""
    for _ in range(attempts):
        current_joint_state = get_joint_state(node, joint_state_topic)
        if current_joint_state is not None:
            break
    if current_joint_state is None:
        if verbose:
            node.get_logger().error("Failed to get current joint state")
        return None

    best_cost = np.inf
    best_joint_state = None

    for _ in range(attempts):
        joint_state = get_ik(
            node, target_pose, move_group_name, base_frame, ik_client, verbose
        )
        if joint_state is None:
            continue

        cost = sum_of_square_diff(current_joint_state, joint_state)
        if cost < best_cost:
            best_cost = cost
            best_joint_state = joint_state

    if not best_joint_state and verbose:
        node.get_logger().error("Failed to get IK solution")

    return best_joint_state

def get_transform_frame(
    node: Node, 
    tf_buffer: tf2_ros.Buffer,
    child_frame: str,
    target_frame: str
) -> tf2_ros.TransformStamped:
    """returns the transform from the target_frame to the child frame"""
    while True:
        try:
            tf_future = tf_buffer.wait_for_transform_async(
                target_frame,
                child_frame,
                rclpy.time.Time())
            rclpy.spin_until_future_complete(node, tf_future)
            node.get_logger().info("found transform future")

            t = asyncio.run(tf_buffer.lookup_transform_async(
                target_frame,
                child_frame,
                rclpy.time.Time()
            ))
            return t
        except tf2_ros.TransformException as ex:
            node.get_logger().error(f"Could not transform target frame \"{target_frame}\" to child frame \"{child_frame}\": {ex}", once=True)
            continue

def get_tool_offset_pose(
    node: Node, 
    original_pose: Pose, 
    z_offset: float,
    frame_id: str,
    tf_buffer: tf2_ros.Buffer,
    verbose: bool = False,
    transform: tf2_ros.TransformStamped | None = None
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
        node.get_logger().info(f"Original position: {original_pose.position}")
        node.get_logger().info(f"Offset vec: {offset_vec}")
        node.get_logger().info(f"New position: {offset_position}")
        node.get_logger().info(f"New orientation: {new_quat}")

    target_pose = PoseStamped()
    target_pose.header.frame_id = frame_id
    target_pose.header.stamp = node.get_clock().now().to_msg()
    target_pose.pose.position = Point(
        x = offset_position[0], y = offset_position[1], z = offset_position[2]
    )
    target_pose.pose.orientation = Quaternion(
        x = new_quat[0], y = new_quat[1], z = new_quat[2], w = new_quat[3],
    )

    # transform CAD local frame pose to world frame pose
    if transform is None:
        for _ in range(10):
            try:
                target_pose = tf_buffer.transform(target_pose, "world", timeout=rclpy.duration.Duration(seconds=0.1))
                return target_pose.pose
            except Exception as e:
                node.get_logger().warn(f"TF transform failed: {e}")
                continue
        return None
    else:
        return tf2_geometry_msgs.do_transform_pose(target_pose.pose, transform)
    
def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Multiply two XYZW quaternions q1 and q2, keeping the computational
    gradient graph intact."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    qw = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    qx = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    qy = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    qz = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    
    return np.stack([qx, qy, qz, qw])

def rotate_vec_by_quat(v: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Rotates a position vector `<x, y, z>` by a quaternion 
    `<qx, qy, qz, qw>`, keeping the computational gradient graph intact."""
    v_quat = np.concatenate([v, np.zeros(1)]) # create pure quaternion
    q_conj = np.zeros_like(q)
    q_conj[:3] = -q[:3]   # calculate q's conjugate
    q_conj[3] = q[3]

    # Perform quaternion rotation: v' = q * v * q^-1
    rotated_v = quat_multiply(quat_multiply(q, v_quat), q_conj)

    return rotated_v[:3]  # Return only the x, y, z part
import rclpy
import time
from rclpy import Node

######### Core Math Utilities #########
import numpy as np
from scipy.spatial.transform import Rotation as R
######### ######### ######### #########

######### ROS and MoveIt Message Types #########
from geometry_msgs.msg import Pose, Point, Quaternion
from sensor_msgs.msg import JointState
from moveit_msgs.msg import (
    RobotState,
    RobotTrajectory,
    MoveItErrorCodes,
    Constraints,
    JointConstraint
)
from moveit_msgs.srv import GetMotionPlan, GetPositionIK, GetPositionFK
from moveit_msgs.action import ExecuteTrajectory
######### ######### #########

######### ROS Transform Utilities #########
import tf2_ros
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
######### ######### ######### ######### #########

######## Custom ROS Utilities ########
# file is located at `src/thermal_camera/thermal_camera/common_motionplan_utilities.py`
# check out get_best_ik() and get_fk()
from thermal_camera import common_motionplan_utilities
######## ######## ######## ######## ########

######## Coldest/Hottest Pose Message Type ########
# check out the message file in `src/thermal_camera_interfaces/msg/Extrema.msg`
from thermal_camera_interfaces import Extrema
######## ######## ######## ######## ######## ########

"""
Your Task:

Subscribe to the hottest pose topic. This hottest pose
contains a temperature value in Celsius. If the value is too hot, the robot should
do a motion primitive to go away from the hottest point, moving upwards to mitigate further
heating of the surface.

Bonus:

Also keep track of the coldest pose. If the overheat safety is triggered,
the robot should move in the direction of the coldest pose instead of to
an arbitrary location.
"""

class RobotInfo:
    """Container for important information for the robot"""
    home_pose = Pose(
        position=Point(x=0.73945, y=-0.29125, z=0.7),
        orientation=Quaternion(x=0.96314, y=-0.26109, z=-0.00405, w=0.06467),
    )
    desired_temperature = 60.0 # Celsius, = 140 Fahrenheit
    threshold_temperature = 70.0 # Celsius, = 158F
    z_offset = 0.35 # 
    hottest_node_topic = "/hottest_pose"
    coldest_node_topic = "/coldest_pose"

    # key motion planning and robot information topics/move group
    move_group_name_ = "ur10e"              # movegroup used for motion planning
    joint_state_topic_ = "joint_states"     # topic to read robot joint states
    plan_srv_name_ = "plan_kinematic_path"  # MoveIt motion planning service, need to set up Service client for this
    ik_srv_name_ = "compute_ik"             # MoveIt IK service, need to set up Service client for this
    fk_srv_name_ = "compute_fk"             # MoveIt FK service, need to set up Service client for this
    execute_action_name_ = "execute_trajectory" # MoveIt trajectory controller topic, need to set up Action client for this


def check_same_pose(pose_1: Pose, pose_2: Pose, pos_threshold=0.001, angle_threshold=np.deg2rad(0.1)) -> bool:
    """Checks if two poses are the same, within thresholds (returns T/F).
    """
    # Position difference
    pos_diff = np.array([
        pose_1.position.x - pose_2.position.x,
        pose_1.position.y - pose_2.position.y,
        pose_1.position.z - pose_2.position.z
    ])
    pos_dist = np.linalg.norm(pos_diff)

    # Quaternion difference using rotation matrix
    q1 = [pose_1.orientation.x, pose_1.orientation.y, pose_1.orientation.z, pose_1.orientation.w]
    q2 = [pose_2.orientation.x, pose_2.orientation.y, pose_2.orientation.z, pose_2.orientation.w]

    # Convert to rotation matrix and compute angular difference
    r1 = R.from_quat(q1)
    r2 = R.from_quat(q2)
    relative_rotation = r1.inv() * r2
    angle_diff = relative_rotation.magnitude()  # Angular distance

    # Print differences for debugging
    print(f"Position Difference: {pos_dist}, Orientation Angle Difference: {np.rad2deg(angle_diff)} degrees")

    # Return True only if both position and orientation thresholds are satisfied
    return pos_dist < pos_threshold and angle_diff < angle_threshold

def get_motion_plan(
    node: Node, target_pose: Pose, linear: bool = True, attempts: int = 10
) -> RobotTrajectory | None:
    """Creates a motion plan to get from the robot's current pose
    to a target pose. Keep the default `linear` value, which uses
    the Pilz LIN planner.
    
    Add this method into your Node class. When you do, replace all instance of
    `node` with `self`."""
    node.get_logger().info("start of get motion plan")

    # TODO get FK
    current_pose = None # replace with pose from FK
    if current_pose is None:
        node.get_logger().error("Failed to get current pose")
    assert current_pose is not None # remove this line when done

    if check_same_pose(current_pose, target_pose):
        node.get_logger().warn("Same pose")
        return None

    current_joint_state_set, current_joint_state = common_motionplan_utilities.wait_for_message(
        JointState, node, node.joint_state_topic_, time_to_wait=1.0
    )
    if not current_joint_state_set:
        node.get_logger().error("Failed to get current joint state")
        return None

    current_robot_state = RobotState()
    current_robot_state.joint_state.position = current_joint_state.position
    node.get_logger().info("made robot state")

    # TODO get IK
    target_joint_state = None # replace with joint state from IK 
    if target_joint_state is None:
        node.get_logger().error("Failed to get target joint state")
        assert target_joint_state is not None # remove this line when done
        return None

    target_constraint = Constraints()
    for i in range(len(target_joint_state.position)):
        joint_constraint = JointConstraint()
        joint_constraint.joint_name = target_joint_state.name[i]
        joint_constraint.position = target_joint_state.position[i]
        joint_constraint.tolerance_above = 0.001
        joint_constraint.tolerance_below = 0.001
        joint_constraint.weight = 1.0
        target_constraint.joint_constraints.append(joint_constraint)

    node.get_logger().info("got constraints")

    request = GetMotionPlan.Request()
    request.motion_plan_request.group_name = node.move_group_name_
    request.motion_plan_request.start_state = current_robot_state
    request.motion_plan_request.goal_constraints.append(target_constraint)
    request.motion_plan_request.num_planning_attempts = 10
    request.motion_plan_request.allowed_planning_time = 5.0
    request.motion_plan_request.max_velocity_scaling_factor = 1.0
    request.motion_plan_request.max_acceleration_scaling_factor = 0.3

    node.get_logger().info("finished request")

    if linear:
        request.motion_plan_request.pipeline_id = "pilz_industrial_motion_planner"
        request.motion_plan_request.planner_id = "LIN"
    else:
        request.motion_plan_request.pipeline_id = "ompl"
        request.motion_plan_request.planner_id = "BFMTkConfigDefault"

    for _ in range(attempts):
        plan_future = node.plan_client_.call_async(request)
        rclpy.spin_until_future_complete(node, plan_future)

        if plan_future.result() is None:
            node.get_logger().error("Failed to get motion plan")

        response = plan_future.result()
        if response.motion_plan_response.error_code.val != MoveItErrorCodes.SUCCESS:
            node.get_logger().error(
                f"Failed to get motion plan: {response.motion_plan_response.error_code.val}"
            )
        else:
            return response.motion_plan_response.trajectory
        
    return None

def execute_trajectory(node: Node, target_pose: Pose) -> None:
    """Creates (with get_motion_plan()) and executes a trajectory to get from 
    the current pose to the target pose.
    
    Add this method into your Node class. When you do, replace all instance of
    `node` with `self`."""
    traj = node.get_motion_plan(target_pose)
    if traj:
        client = node.get_motion_execute_client()
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = traj

        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, future)
        
        goal_handle = future.result()
        if not goal_handle.accepted:
            node.get_logger().error("Failed to execute trajectory")
        else:
            node.get_logger().info("Trajectory accepted")

            result_future = goal_handle.get_result_async()

            expect_duration = traj.joint_trajectory.points[-1].time_from_start
            expect_time = time.time() + 2 * expect_duration.sec
            while not result_future.done() and time.time() < expect_time:
                time.sleep(0.01)

            node.get_logger().info("Trajectory executed")
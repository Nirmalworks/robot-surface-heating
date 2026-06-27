from launch import LaunchDescription
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import GroupAction, DeclareLaunchArgument

import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare("my_robot_cell_description"), "urdf", "my_robot_cell.urdf.xacro"]),
        ]
    )
    robot_description = {"robot_description": robot_description_content}
    
    srdf_path = "/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_moveit_config/config/my_robot_cell_ur10e.srdf"
    # srdf_path = "/home/tamp/Desktop/ur_bimanual/src/dual_moveit_config/config/my_robot_cell.srdf"
    with open(srdf_path, 'r') as f:
        robot_description_semantic_content = f.read()
    robot_description_semantic = {"robot_description_semantic": robot_description_semantic_content}

    robot_description_kinematics = PathJoinSubstitution(
        [FindPackageShare("thermal_moveit_config"), "config", "kinematics.yaml"]
    )

    robot_interface_params= os.path.join(
        get_package_share_directory('robot_interface'),
        'config',
        'robot_interface_parameters.yaml'
    )

    # MoveGroupInterface demo executable
    move_group_demo = Node(
        name="demo_robot_interface",
        package="robot_interface",
        executable="demo_robot_interface.py",
        # executable="moveit_test_v2",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            robot_interface_params,
        ],
    )

    # base collision table executable
    base_collision_table = Node(
        name="base_collision_table",
        package="robot_interface",
        executable="base_collision_table",
        output="screen",
        parameters=[],
    )

    ld = LaunchDescription()
    ld.add_action(move_group_demo)
    # ld.add_action(base_collision_table)
    return ld


    # return LaunchDescription([move_group_demo])
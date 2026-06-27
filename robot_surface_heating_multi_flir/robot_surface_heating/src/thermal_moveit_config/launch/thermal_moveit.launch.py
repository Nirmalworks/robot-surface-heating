# Copyright (c) 2021 PickNik, Inc.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the {copyright_holder} nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

#
# Author: Denis Stogl

import os
from moveit_configs_utils import MoveItConfigsBuilder
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from ur_moveit_config.launch_common import load_yaml

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.conditions import IfCondition
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution

from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def launch_setup(context, *args, **kwargs):

    moveit_package = LaunchConfiguration("moveit_package")
    description_package = LaunchConfiguration("description_package")
    description_file = LaunchConfiguration("description_file")
    moveit_config_package = LaunchConfiguration("moveit_config_package")
    moveit_joint_limits_file = LaunchConfiguration("moveit_joint_limits_file")
    moveit_config_file = LaunchConfiguration("moveit_config_file")
    warehouse_sqlite_path = LaunchConfiguration("warehouse_sqlite_path")
    prefix = LaunchConfiguration("tf_prefix")
    use_sim_time = LaunchConfiguration("use_sim_time")
    launch_rviz = LaunchConfiguration("launch_rviz")
    launch_servo = LaunchConfiguration("launch_servo")

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution([FindPackageShare(description_package), "urdf", description_file]),
        ]
    )
    robot_description = {"robot_description": robot_description_content}

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="both",
        parameters=[robot_description],
    )

    # MoveIt Configuration
    robot_description_semantic_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare(moveit_package), "config", moveit_config_file]
            ),
            " ",
            "name:=",
            # Also ur_type parameter could be used but then the planning group names in yaml
            # configs has to be updated!
            "ur",
            " ",
            "prefix:=",
            prefix,
            " ",
        ]
    )
    cnfg_path = get_package_share_directory('thermal_moveit_config')
    srdf_path = os.path.join(cnfg_path, 'config', 'my_robot_cell_ur10e.srdf')
    with open(srdf_path, 'r') as f:
        robot_description_semantic_content = f.read()

    robot_description_semantic = {"robot_description_semantic": robot_description_semantic_content}

    robot_description_kinematics = PathJoinSubstitution(
        [FindPackageShare(moveit_package), "config", "kinematics.yaml"]
    )

    robot_description_planning = {
        "robot_description_planning": load_yaml(
            str(moveit_package.perform(context)),
            os.path.join("config", str(moveit_joint_limits_file.perform(context))),
        )
    }


    # moveit_config = (
    #     MoveItConfigsBuilder("dual")
    #     .robot_description(
    #         file_path="config/my_robot_cell.urdf.xacro",
    #         mappings={
    #             "ros2_control_hardware_type": LaunchConfiguration(
    #                 "ros2_control_hardware_type"
    #             )
    #         },
    #     )
    #     .robot_description_semantic(file_path="config/my_robot_cell.srdf")
    #     .trajectory_execution(file_path="config/moveit_controllers.yaml")
    #     .planning_scene_monitor(
    #         publish_robot_description=True, publish_robot_description_semantic=True
    #     )
    #     .planning_pipelines(
    #         pipelines=["ompl", "pilz_industrial_motion_planner"]
    #     )
    #     .to_moveit_configs()
    # )

    # Planning Configuration
    ompl_planning_pipeline_config = {
        "planning_pipelines": ["ompl", "pilz_industrial_motion_planner"],
        "default_planning_pipeline": "pilz_industrial_motion_planner",
        "ompl": {
            "planning_plugin": "ompl_interface/OMPLPlanner",
            "request_adapters": """default_planner_request_adapters/AddTimeOptimalParameterization default_planner_request_adapters/FixWorkspaceBounds default_planner_request_adapters/FixStartStateBounds default_planner_request_adapters/FixStartStateCollision default_planner_request_adapters/FixStartStatePathConstraints""",
            "start_state_max_bounds_error": 0.1,
        },
        "pilz_industrial_motion_planner": {
            "planning_plugin": "pilz_industrial_motion_planner/CommandPlanner",
            "request_adapters": """""",
            "start_state_max_bounds_error": 0.1,
            "default_planner_config": "LIN",
            "cartesian_limits" :{
                "max_trans_vel": 1.0,
                "max_trans_acc": 2.25,
                "max_trans_dec": -5.0,
                "max_rot_vel": 1.57,
            },
        },
        "robot_description_planning":{},
    }
    ompl_planning_yaml = load_yaml("thermal_moveit_config", "config/ompl_planning.yaml")
    pilz_planning_yaml = load_yaml("thermal_moveit_config", "config/pilz_industrial_motion_planner_planning.yaml")
    ompl_planning_pipeline_config["ompl"].update(ompl_planning_yaml)
    ompl_planning_pipeline_config["pilz_industrial_motion_planner"].update(pilz_planning_yaml)

    pilz_planning_pipeline_config = {
        "planning_pipelines": ["ompl", "pilz_industrial_motion_planner"],
        "default_planning_pipeline": "ompl",
        "pilz_industrial_motion_planner": {},
        "move_group": {},
        "robot_description_planning":{},
    }
    pilz_planning_yaml = load_yaml("thermal_moveit_config", "config/pilz_industrial_motion_planner_planning.yaml")
    pilz_planning_pipeline_config["move_group"].update(pilz_planning_yaml)
    pilz_planning_pipeline_config["pilz_industrial_motion_planner"].update(pilz_planning_yaml) 

    pilz_cartesian_limits_yaml = load_yaml("thermal_moveit_config", "config/pilz_cartesian_limits.yaml")
    ompl_planning_pipeline_config["robot_description_planning"].update(pilz_cartesian_limits_yaml)

    # Trajectory Execution Configuration
    controllers_yaml = load_yaml("thermal_moveit_config", "config/moveit_controllers.yaml")
    # the scaled_joint_trajectory_controller does not work on fake hardware

    moveit_controllers = {
        "moveit_simple_controller_manager": controllers_yaml,
        "moveit_controller_manager": "moveit_simple_controller_manager/MoveItSimpleControllerManager",
    }

    trajectory_execution = {
        "moveit_manage_controllers": False,
        "trajectory_execution.allowed_execution_duration_scaling": 1.2,
        "trajectory_execution.allowed_goal_duration_margin": 0.5,
        "trajectory_execution.allowed_start_tolerance": 0.01,
    }

    planning_scene_monitor_parameters = {
        "publish_planning_scene": True,
        "publish_geometry_updates": True,
        "publish_state_updates": True,
        "publish_transforms_updates": True,
    }

    warehouse_ros_config = {
        "warehouse_plugin": "warehouse_ros_sqlite::DatabaseConnection",
        "warehouse_host": warehouse_sqlite_path,
    }

    move_group_capabilities = {"capabilities": "pilz_industrial_motion_planner/MoveGroupSequenceAction \
pilz_industrial_motion_planner/MoveGroupSequenceService",
                                                }

    # Start the actual move_group node/action server
    move_group_node = Node(
        package="moveit_ros_move_group",
        executable="move_group",
        output="screen",
        parameters=[
            robot_description,
            robot_description_semantic,
            robot_description_kinematics,
            robot_description_planning,
            ompl_planning_pipeline_config,
            trajectory_execution,
            moveit_controllers,
            planning_scene_monitor_parameters,
            {"use_sim_time": use_sim_time},
            warehouse_ros_config,
            move_group_capabilities,
            {"publish_robot_description": True},
            {"publish_robot_description_semantic": True},
            # {"publish_state_updates": True},
            # {"publish_transforms_updates": True},
            # {"publish_geometry_updates": True},
        ],

    )

    # rviz with moveit configuration
    rviz_config_file = PathJoinSubstitution(
        [FindPackageShare(moveit_package), "config", "moveit.rviz"]
    )
    rviz_node = Node(
        package="rviz2",
        condition=IfCondition(launch_rviz),
        executable="rviz2",
        name="rviz2_moveit",
        output="log",
        arguments=["-d", rviz_config_file],
        parameters=[
            robot_description,
            robot_description_semantic,
            ompl_planning_pipeline_config,
            robot_description_kinematics,
            robot_description_planning,
            warehouse_ros_config,
        ],
    )

    # UR10e joint state publisher
    ur10e_joint_states = Node(
        namespace= "", package='ur_robot_driver', executable='ur10e_joint_states.py', output='screen')

    nodes_to_start = [move_group_node, rviz_node, robot_state_publisher_node, ur10e_joint_states]

    return nodes_to_start


def generate_launch_description():

    declared_arguments = []
    # UR specific arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_fake_hardware",
            default_value="false",
            description="Indicate whether robot is running with fake hardware mirroring command to its states.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_limits",
            default_value="true",
            description="Enables the safety limits controller if true.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_pos_margin",
            default_value="0.15",
            description="The margin to lower and upper limits in the safety controller.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "safety_k_position",
            default_value="20",
            description="k-position factor in the safety controller.",
        )
    )
    # General arguments
    declared_arguments.append(
        DeclareLaunchArgument(
            "moveit_package",
            default_value="thermal_moveit_config",
            description="Description package with robot URDF/XACRO files. Usually the argument "
            "is not set, it enables use of a custom description.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_package",
            default_value="my_robot_cell_description",
            description="Description package with robot URDF/XACRO files. Usually the argument "
            "is not set, it enables use of a custom description.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "description_file",
            default_value="my_robot_cell.urdf.xacro",
            description="URDF/XACRO description file with the robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "moveit_config_package",
            default_value="ur_moveit_config",
            description="MoveIt config package with robot SRDF/XACRO files. Usually the argument "
            "is not set, it enables use of a custom moveit config.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "moveit_config_file",
            default_value="ur.srdf.xacro",
            description="MoveIt SRDF/XACRO description file with the robot.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "moveit_joint_limits_file",
            default_value="joint_limits.yaml",
            description="MoveIt joint limits that augment or override the values from the URDF robot_description.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "warehouse_sqlite_path",
            default_value=os.path.expanduser("~/.ros/warehouse_ros.sqlite"),
            description="Path where the warehouse database should be stored",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "use_sim_time",
            default_value="false",
            description="Make MoveIt to use simulation time. This is needed for the trajectory planing in simulation.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument(
            "tf_prefix",
            default_value='""',
            description="Prefix of the joint names, useful for "
            "multi-robot setup. If changed than also joint names in the controllers' configuration "
            "have to be updated.",
        )
    )
    declared_arguments.append(
        DeclareLaunchArgument("launch_rviz", default_value="true", description="Launch RViz?")
    )
    declared_arguments.append(
        DeclareLaunchArgument("launch_servo", default_value="true", description="Launch Servo?")
    )

    # configure UR10e driver launch
    ur_launch_dir = get_package_share_directory('ur_robot_driver')
    ur10e_driver_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ur_launch_dir, 'launch', 'ur10e_inter.launch.py')),
        )

    # return LaunchDescription([*declared_arguments, ur10e_driver_launch, OpaqueFunction(function=launch_setup)])
    return LaunchDescription([*declared_arguments, OpaqueFunction(function=launch_setup)])
    # return LaunchDescription(declared_arguments + [OpaqueFunction(function=launch_setup)] + [ur10e_driver_launch, ur10e_joint_states])

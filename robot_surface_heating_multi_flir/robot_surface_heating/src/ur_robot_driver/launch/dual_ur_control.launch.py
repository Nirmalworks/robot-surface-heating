import os
from ament_index_python.packages import get_package_share_directory
from launch.actions import IncludeLaunchDescription
from launch import LaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.substitutions import LaunchConfiguration

from launch.actions import GroupAction
from launch_ros.actions import PushRosNamespace
import launch
import launch_ros.actions
from launch_ros.actions import Node

def generate_launch_description():

    ur_type = LaunchConfiguration('ur_type') 
    arm_0_robot_ip = LaunchConfiguration('arm_0_robot_ip') 
    arm_0_controller_file = LaunchConfiguration('arm_0_controller_file') 
    arm_0_tf_prefix = LaunchConfiguration('arm_0_tf_prefix') 
    arm_0_script_command_port = LaunchConfiguration('arm_0_script_command_port')
    arm_0_trajectory_port = LaunchConfiguration('arm_0_trajectory_port')
    arm_0_reverse_port = LaunchConfiguration('arm_0_reverse_port')
    arm_0_script_sender_port = LaunchConfiguration('arm_0_script_sender_port')

    arm_1_robot_ip = LaunchConfiguration('arm_1_robot_ip') 
    arm_1_controller_file = LaunchConfiguration('arm_1_controller_file') 
    arm_1_tf_prefix = LaunchConfiguration('arm_1_tf_prefix') 
    arm_1_script_command_port = LaunchConfiguration('arm_1_script_command_port')
    arm_1_trajectory_port = LaunchConfiguration('arm_1_trajectory_port')
    arm_1_reverse_port = LaunchConfiguration('arm_1_reverse_port')
    arm_1_script_sender_port = LaunchConfiguration('arm_1_script_sender_port')

    # # UR specific arguments
    arm_0_robot_ip_arg = DeclareLaunchArgument(
            "arm_0_robot_ip",
            default_value='192.168.10.125',
            description="IP address by which the robot can be reached.",
    )
    arm_0_controller_file_arg = DeclareLaunchArgument(
            "arm_0_controller_file",
            default_value="arm_0_ur_controllers.yaml",
            description="YAML file with the controllers configuration.",
    )
    arm_0_tf_prefix_arg = DeclareLaunchArgument(
            "arm_0_tf_prefix",
            default_value="arm_0_",
            description="tf_prefix of the joint names, useful for \
            multi-robot setup. If changed, also joint names in the controllers' configuration \
            have to be updated.",
    )
    arm_0_script_command_port_arg =  DeclareLaunchArgument(
        "arm_0_script_command_port",
        default_value="50014",
        description="Port that will be opened to forward script commands from the driver to the robot",
    )
    arm_0_trajectory_port_arg = DeclareLaunchArgument(
        "arm_0_trajectory_port",
        default_value="50013",
        description="Port that will be opened to forward script commands from the driver to the robot",
    )
    arm_0_reverse_port_arg = DeclareLaunchArgument(
        "arm_0_reverse_port",
        default_value="50011",
        description="Port that will be opened to forward script commands from the driver to the robot",
    )
    arm_0_script_sender_port_arg = DeclareLaunchArgument(
        "arm_0_script_sender_port",
        default_value="50012",
        description="Port that will be opened to forward script commands from the driver to the robot",
    )

    arm_1_robot_ip_arg = DeclareLaunchArgument(
            "arm_0_robot_ip",
            default_value='192.168.10.112',
            description="IP address by which the robot can be reached.",
    )
    arm_1_controller_file_arg = DeclareLaunchArgument(
            "arm_1_controller_file",
            default_value="arm_1_ur_controllers.yaml",
            description="YAML file with the controllers configuration.",
    )
    arm_1_tf_prefix_arg = DeclareLaunchArgument(
            "arm_0_tf_prefix",
            default_value="arm_1_",
            description="tf_prefix of the joint names, useful for \
            multi-robot setup. If changed, also joint names in the controllers' configuration \
            have to be updated.",
    )
    arm_1_script_command_port_arg =  DeclareLaunchArgument(
        "arm_0_script_command_port",
        default_value="50004",
        description="Port that will be opened to forward script commands from the driver to the robot",
    )
    arm_1_trajectory_port_arg = DeclareLaunchArgument(
        "arm_0_trajectory_port",
        default_value="50003",
        description="Port that will be opened to forward script commands from the driver to the robot",
    )
    arm_1_reverse_port_arg = DeclareLaunchArgument(
        "arm_0_reverse_port",
        default_value="50001",
        description="Port that will be opened to forward script commands from the driver to the robot",
    )
    arm_1_script_sender_port_arg = DeclareLaunchArgument(
        "arm_0_script_sender_port",
        default_value="50002",
        description="Port that will be opened to forward script commands from the driver to the robot",
    )
    

    ur_launch_dir = get_package_share_directory('ur_robot_driver')
    moveit_launch_dir = get_package_share_directory('thermal_moveit_config')

    arm_0 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ur_launch_dir, 'launch', 'ur_control.launch.py')),
        launch_arguments={'ur_type': "ur5",
                          'robot_ip': arm_0_robot_ip,
                          'controllers_file': arm_0_controller_file,
                          'description_file': "ur.urdf.xacro",
                          'tf_prefix': arm_0_tf_prefix,
                          'script_command_port': arm_0_script_command_port,
                          'trajectory_port': arm_0_trajectory_port,
                          'reverse_port': arm_0_reverse_port,
                          'script_sender_port': arm_0_script_sender_port,
                          }.items())
    
    arm_0_with_namespace = GroupAction(
     actions=[
         PushRosNamespace('arm_0'),
         arm_0
      ]
    )

    arm_1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ur_launch_dir, 'launch', 'ur10e.launch.py')),
        )

    ur5_joint_states = Node(
        namespace= "", package='ur_robot_driver', executable='ur5_joint_states.py', output='screen')

    ur10e_joint_states = Node(
        namespace= "", package='ur_robot_driver', executable='ur10e_joint_states.py', output='screen')

    moveit_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(moveit_launch_dir, 'launch', 'dual_moveit.launch.py')),
    )

    
    return LaunchDescription(
        [
            arm_0_robot_ip_arg,
            arm_0_controller_file_arg,
            arm_0_tf_prefix_arg,
            arm_0_script_command_port_arg,
            arm_0_trajectory_port_arg,
            arm_0_reverse_port_arg,
            arm_0_script_sender_port_arg,

            arm_0_with_namespace,

        ] 
        +
        [
            arm_1
        ]
        + 
        [
            ur5_joint_states,
            ur10e_joint_states,
        ] 


    )

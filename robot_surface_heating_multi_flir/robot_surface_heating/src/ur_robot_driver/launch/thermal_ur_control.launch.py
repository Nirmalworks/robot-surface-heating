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

    ur_launch_dir = get_package_share_directory('ur_robot_driver')

    arm_1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(ur_launch_dir, 'launch', 'ur10e.launch.py')),
        # PythonLaunchDescriptionSource(os.path.join(ur_launch_dir, 'launch', 'ur5.launch.py')),
        )

    # ur10e_joint_states = Node(
    #     namespace= "", package='ur_robot_driver', executable='ur10e_joint_states.py', output='screen')

    
    return LaunchDescription(
        [
            arm_1
        ]
        # + 
        # [
        #     ur10e_joint_states,
        # ] 


    )

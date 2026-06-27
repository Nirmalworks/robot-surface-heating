import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([
        Node(
            package='fake_gripper_publisher',
            executable='fake_joint_state_publisher',
            name='fake_gripper_js_publisher',
        ),
])
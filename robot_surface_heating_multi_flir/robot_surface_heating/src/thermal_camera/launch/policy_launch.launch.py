from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # hottest/coldest visualizer node
    extrema_pub_node = Node(
        package='thermal_camera', 
        # executable='extrema_publisher',
        # name='extrema_publisher',
        executable='extrema_publisher_modular',
        name='extrema_publisher_modular',
    )

    return LaunchDescription([
        extrema_pub_node,
    ])
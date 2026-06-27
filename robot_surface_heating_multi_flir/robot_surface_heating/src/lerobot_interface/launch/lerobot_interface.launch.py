import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    lerobot_params = os.path.join(
        get_package_share_directory('lerobot_interface'),
        'config',
        'lerobot_interface_params.yaml'
        )

    lerobot_interface_node = Node(
        package='lerobot_interface',
        executable='lerobot_control_node',
        parameters=[lerobot_params]
    )

    return LaunchDescription([
        lerobot_interface_node,
    ])
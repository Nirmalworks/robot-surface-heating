import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    lepton_params = os.path.join(
        get_package_share_directory('thermal_camera'),
        'config',
        'camera_params.yaml'
        )

    visualizer_node = Node(
        package='thermal_camera',
        executable='uvc_visualize',
        parameters=[lepton_params]
    )

    return LaunchDescription([
        visualizer_node
    ])
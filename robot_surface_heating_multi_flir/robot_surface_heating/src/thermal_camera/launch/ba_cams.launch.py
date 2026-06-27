import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    ba_params = os.path.join(
        get_package_share_directory('thermal_camera'),
        'config',
        'pnp_calibration_config.yaml'
        )

    ba_node = Node(
        package='thermal_camera',
        executable='bundle_adjustment',
        name='bundle_adjustment',
        parameters=[ba_params],
        output='screen'
    )

    return LaunchDescription([
        ba_node
    ])
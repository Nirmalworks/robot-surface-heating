import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():

    pnp_params = os.path.join(
        get_package_share_directory('thermal_camera'),
        'config',
        'pnp_calibration_config.yaml'
        )

    pnp_node = Node(
        package='thermal_camera',
        executable='pnp_extrinsic',
        parameters=[pnp_params]
    )

    return LaunchDescription([
        pnp_node
    ])
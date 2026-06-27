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
        executable='thermal_camera_publisher',
        parameters=[lepton_params]
    )

    #    [ 0.78080308, -0.32230588,  0.5352247 ,  0.41100991],
    #    [-0.60980872, -0.57954072,  0.5406162 , -0.52491785],
    #    [ 0.13594073, -0.74849948, -0.64905212,  0.71021088],
    #    [ 0.        ,  0.        ,  0.        ,  1.        ]

    frame_broadcaster = Node(
        package='thermal_camera',
        executable='camera_frame_broadcaster',
        name='camera_frame_broadcaster',
        parameters=[lepton_params]
    )

    lepton0_transform = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        # arguments = ['--x', '0.42973184', '--y', '-0.53399323', '--z', '0.70900029', '--qx', '0.8625007', 
        #     '--qy', '-0.2891711', '--qz', '0.1952516', '--qw', '-0.3665372',
        #     '--frame-id', 'base_link', '--child-frame-id', 'thermal_camera_0']
        # arguments = ['--x', '0.41100991', '--y', '-0.52491785', '--z', '0.71021088', '--qx', '0.8733501', 
        #     '--qy', '-0.2705069', '--qz', '0.1553264', '--qw', '-0.3741114',
        #     '--frame-id', 'base_link', '--child-frame-id', 'thermal_camera_0']

        # best candidate
        arguments = ['--x', '0.40885124', '--y', '-0.51967428', '--z', '0.714809', '--qx', '0.872446', 
            '--qy', '-0.2625942', '--qz', '0.1749984', '--qw', '-0.3731727',
            '--frame-id', 'base_link', '--child-frame-id', 'thermal_camera_0']
    )

    lepton0_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        # arguments = ['0', '0', '0', '0', '-1.5708', '0', 'thermal_camera_0', 'thermal_camera_0_optical_frame']
        arguments = ['0', '0', '0', '0', '0', '0', 'thermal_camera_0', 'thermal_camera_0_optical_frame']
    )

    return LaunchDescription([
        # visualizer_node, lepton0_transform, lepton0_optical
        visualizer_node, frame_broadcaster
    ])
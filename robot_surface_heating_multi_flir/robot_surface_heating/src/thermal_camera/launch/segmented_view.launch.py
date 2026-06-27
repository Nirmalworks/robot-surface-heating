#!/usr/bin/env python3
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        # 1) Your virtual camera → publishes /simulated_camera_0/mask
        Node(
            package='thermal_camera',            # your package name
            executable='virtual_camera_viewer',  # your entry point for virtual_camera_viewer.py
            name='virtual_camera_viewer_0',
            output='screen',
            # if you need to pass in arguments or params, add them here
        ),

        # # 2) Segmentation node → publishes /thermal_camera_0/segmented
        # Node(
        #     package='thermal_camera',
        #     executable='segmentation_node',      # entry point for segmentation_node.py
        #     name='segmentation_node_0',
        #     output='screen',
        # ),

        # # 3) Show the segmented feed
        # Node(
        #     package='image_tools',
        #     executable='showimage',
        #     name='segmented_viewer_0',
        #     output='screen',
        #     remappings=[
        #         # showimage subscribes to “image” by default,
        #         # so remap it to your segmented topic:
        #         ('image', '/thermal_camera_0/segmented')
        #     ],
        # ),
    ])

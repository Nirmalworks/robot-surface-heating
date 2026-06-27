from launch import LaunchDescription
from launch_ros.actions import ComposableNodeContainer
from launch_ros.descriptions import ComposableNode
from launch_ros.actions import Node

def generate_launch_description():

    realsense_color = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments = ['--x', '-0.662784', '--y', '-0.254493', '--z', '1.26704', '--qx', '-0.0037613',
            '--qy', '0.9999708', '--qz', '-0.0049263', '--qw', '0.0044727',
            '--frame-id', 'world', '--child-frame-id', 'camera_color_optical_frame']

    )
    realsense_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        arguments = ['--x', '-0.67798973', '--y', '-0.2546055', '--z', '1.26728941', '--qx', '0.4953305', 
            '--qy', '0.4960405', '--qz', '-0.5017189', '--qw', '0.5068231',
            '--frame-id', 'world', '--child-frame-id', 'camera_link']
    )

    container = ComposableNodeContainer(
        name='depth_image_proc_container',
        namespace='',
        package='rclcpp_components',
        executable='component_container',
        composable_node_descriptions=[
            ComposableNode(
                package='depth_image_proc',
                plugin='depth_image_proc::RegisterNode',
                name='point_cloud_register_node',
                remappings=[
                    ('depth/image_rect', '/camera/camera/depth/image_rect_raw'),
                    ('depth/camera_info', '/camera/camera/depth/camera_info'),
                    ('rgb/camera_info', '/thermal_camera_0/camera_info'),
                ],
                parameters=[
                    {
                        'queue_size': 25,
                        'approximate_sync': True,
                        'use_system_default_qos': True
                    }
                ],
            ),
            ComposableNode(
                package='depth_image_proc',
                plugin='depth_image_proc::PointCloudXyzrgbNode',
                name='point_cloud_xyzrgb_node',
                remappings=[
                    ('rgb/image_rect_color', '/thermal_camera_0/image_raw'),
                    ('rgb/camera_info', '/thermal_camera_0/camera_info'),
                    ('points', '/thermal_camera_0/points'),
                ],
                parameters=[
                    {
                        'queue_size': 25,
                        'approximate_sync': True,
                        'reliability': 'reliable'
                    }
                ],
            ),
        ]
    )

    return LaunchDescription([container, realsense_color, realsense_base])
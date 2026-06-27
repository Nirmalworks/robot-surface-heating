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
        executable='extrema_publisher',
        name='extrema_publisher',
    )

    pcl_params = os.path.join(
        get_package_share_directory('thermal_camera'),
        'config',
        'pcl_params.yaml'
        )

    # pointcloud coloration node
    pointcloud_color_node = Node(
        package='thermal_camera', 
        executable='project_colored_cloud',
        parameters=[pcl_params],
    ) 

    pcl_selector_node = Node(
        package='thermal_camera',
        executable='o3d_visual_surface',
        name='interactive_surface_selector'
    )
    
    return LaunchDescription([
        # extrema_pub_node,
        pointcloud_color_node,
        pcl_selector_node
    ])
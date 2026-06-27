from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    cad_file_arg = DeclareLaunchArgument(
        'cad_file',
        # default_value='',
        # default_value='/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/test_rack_box.stl',
        # default_value='/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/test_big_box.stl',
        # default_value='/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/Demo_RSS_part.stl',
        # default_value='/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/Demo_RSS_part_v1.stl',
        default_value='/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/RSS_v3.stl',
        description='Full path to CAD file (STL or DAE format)'
    )
    
    frame_arg = DeclareLaunchArgument(
        'frame_id',
        # default_value='camera_depth_optical_frame',
        default_value='base_link',
        description='Frame to display CAD in'
    )
    
    x_arg = DeclareLaunchArgument('x', default_value='0.5')
    y_arg = DeclareLaunchArgument('y', default_value='0.0')
    z_arg = DeclareLaunchArgument('z', default_value='0.0')
    scale_arg = DeclareLaunchArgument('scale', default_value='0.001')   # convert mm to meters
    # scale_arg = DeclareLaunchArgument('scale', default_value='0.0001')   # convert mm to meters
    # scale_arg = DeclareLaunchArgument('scale', default_value='1.')
    
    # CAD visualizer node
    cad_viz_node = Node(
        package='thermal_camera', 
        executable='simple_cad_visualizer',
        name='cad_visualizer',
        parameters=[{
            'cad_file': LaunchConfiguration('cad_file'),
            'frame_id': LaunchConfiguration('frame_id'),
            'x': LaunchConfiguration('x'),
            'y': LaunchConfiguration('y'),
            'z': LaunchConfiguration('z'),
            'scale': LaunchConfiguration('scale')
        }],
        output='screen'
    )

    # hottest/coldest visualizer node
    extrema_pub_node = Node(
        package='thermal_camera', 
        executable='extrema_publisher',
        name='extrema_publisher',
        parameters=[{'cad_file':LaunchConfiguration('cad_file')}]
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
    
    return LaunchDescription([
        cad_file_arg,
        frame_arg,
        x_arg,
        y_arg,
        z_arg,
        scale_arg,
        cad_viz_node,
        extrema_pub_node,
        pointcloud_color_node,
    ])
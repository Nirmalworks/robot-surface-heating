import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def generate_launch_description():

    lepton_params = os.path.join(
        get_package_share_directory('thermal_camera'),
        'config',
        'camera_params.yaml'
        )

    thermal_publisher_node = Node(
        package='thermal_camera',
        executable='thermal_camera_publisher',
        parameters=[lepton_params],
        output='screen'
    )

    frame_broadcaster = Node(
        package='thermal_camera',
        executable='camera_frame_broadcaster',
        name='camera_frame_broadcaster',
        parameters=[lepton_params]
    )

    robot_mask_publisher_node = Node(
        package='thermal_camera',
        executable='robot_mask_pub',
        name='robot_mask_pub',
        output='screen'
    )

    cad_file_arg = DeclareLaunchArgument(
        'cad_file',
        # default_value='',
        # default_value='/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/test_rack_box.stl',
        # default_value='/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/test_big_box.stl',
        # default_value='/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/Demo_RSS_part.stl',
        # default_value='/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/Demo_RSS_part_v1.stl',
        default_value='/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/RSS_v3.stl',
        # default_value='/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/thermal_camera/thin_plate.STL',
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

    return LaunchDescription([
        # visualizer_node, lepton0_transform, lepton0_optical
        thermal_publisher_node, frame_broadcaster,
        robot_mask_publisher_node,
        cad_file_arg,
        frame_arg,
        x_arg,
        y_arg,
        z_arg,
        scale_arg,
        cad_viz_node,
    ])
import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node


def generate_launch_description():

    return LaunchDescription([
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            # arguments = ['--x', '-0.64995', '--y', '0.16439', '--z', '1.22083', '--qx', '-0.1513992', 
            #     '--qy', '0.975451', '--qz', '-0.1590221', '--qw', '-0.0169004',
            #     '--frame-id', 'world', '--child-frame-id', 'camera_link']
            # arguments = ['--x', '-0.64995', '--y', '0.16439', '--z', '1.22083', '--qx', '-0.1513992', 
            #     '--qy', '0.975451', '--qz', '-0.1590221', '--qw', '-0.0169004',
            #     '--frame-id', 'world', '--child-frame-id', 'camera_color_optical_frame']
            # arguments = ['--x', '0.638046', '--y', '-0.173092', '--z', '1.20169', '--qx', '0.984838', 
            #     '--qy', '-0.00241715', '--qz', '0.0129589', '--qw', '-0.172976',
            #     '--frame-id', 'world', '--child-frame-id', 'camera_color_optical_frame']
            # arguments = ['--x', '-0.681651', '--y', '0.148746', '--z', '0.90648', '--qx', '0.00474105', 
            #     '--qy', '0.98527', '--qz', '-0.169507', '--qw', '-0.022089',
            #     '--frame-id', 'world', '--child-frame-id', 'camera_color_optical_frame']
            # arguments = ['--x', '-0.65585', '--y', '0.168632', '--z', '0.846606', '--qx', '0.0400876',  # pre camera misplace arguments
            #     '--qy', '0.983331', '--qz', '-0.175354', '--qw', '-0.0265497',
            #     '--frame-id', 'world', '--child-frame-id', 'camera_color_optical_frame']
            # arguments = ['--x', '-0.656633', '--y', '0.163523', '--z', '0.852196', '--qx', '0.0174496',
            #     '--qy', '0.986067', '--qz', '-0.164629', '--qw', '-0.0162933',
            #     '--frame-id', 'world', '--child-frame-id', 'camera_color_optical_frame']

            ####### most recent calibration pre-9/26 workstation move #######
            # arguments = ['--x', '-0.646870971861243', '--y', '0.1882564411877606', '--z', '0.864622677996587', '--qx', '0.0174496',
            #     '--qy', '0.986067', '--qz', '-0.164629', '--qw', '-0.0162933',
            #     '--frame-id', 'world', '--child-frame-id', 'camera_color_optical_frame']

            ####### newest calibration following workstation move on 9/26 ########
            arguments = ['--x', '0.814', '--y', '-0.264', '--z', '1.290', '--qx', '0.010',
                '--qy', '1.000', '--qz', '-0.013', '--qw', '0.003',
                '--frame-id', 'base_link', '--child-frame-id', 'camera_color_optical_frame']

        ),
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            # arguments = ['--x', '-0.65077521', '--y', '0.17859553', '--z', '1.21608436', '--qx', '0.34314618', 
            #     '--qy', '0.38476658', '--qz', '-0.4999953', '--qw', '0.69585205',
            #     '--frame-id', 'world', '--child-frame-id', 'camera_link']
            # arguments = ['--x', '-0.65077521', '--y', '0.17859553', '--z', '1.21608436', '--qx', ' 0.0464961', 
            #     '--qy', '-0.6780158', '--qz', '0.7335753', '--qw', '0',
            #     '--frame-id', 'world', '--child-frame-id', 'camera_link']
            # arguments = ['--x', '-0.67682148', '--y', '0.14912606', '--z', '0.90748059', '--qx', '0.39947023', 
            #     '--qy', '0.41767666', '--qz', '-0.5913454', '--qw', '0.56176796',
            #     '--frame-id', 'world', '--child-frame-id', 'camera_link']
            # arguments = ['--x', '-0.67095788', '--y', '0.17009981', '--z', '0.8475477', '--qx', '0.41107642',  # pre camera misplace arguments 
            #     '--qy', '0.39832614', '--qz', '-0.61372025', '--qw', '0.54378296',
            #     '--frame-id', 'world', '--child-frame-id', 'camera_link']
            # arguments = ['--x', '-0.67180458', '--y', '0.16424819', '--z', '0.85296183', '--qx', '0.41157144', 
            #     '--qy', '0.41128136', '--qz', '-0.59331716', '--qw', '0.55626553',
            #     '--frame-id', 'world', '--child-frame-id', 'camera_link']

            ####### most recent calibration pre-9/26 workstation move #######
            # arguments = ['--x', '-0.662042551861243', '--y', '0.1889816311877606', '--z', '0.8653885079965871', '--qx', '0.41157144', 
            #     '--qy', '0.41128136', '--qz', '-0.59331716', '--qw', '0.55626553',
            #     '--frame-id', 'world', '--child-frame-id', 'camera_link']

            ####### newest calibration following workstation move on 9/26 ########
            arguments = ['--x', '0.799', '--y', '-0.264', '--z', '1.290', '--qx', '0.497', 
                '--qy', '0.486', '--qz', '-0.514', '--qw', '0.503',
                '--frame-id', 'base_link', '--child-frame-id', 'camera_link']
        ),
    ])

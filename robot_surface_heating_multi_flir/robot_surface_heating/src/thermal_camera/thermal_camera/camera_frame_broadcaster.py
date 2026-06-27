from geometry_msgs.msg import TransformStamped
from ament_index_python.packages import get_package_share_directory
import rclpy
from rclpy.node import Node
import os
from tf2_ros import TransformBroadcaster
import pickle
from scipy.spatial.transform import Rotation as R
import numpy as np

class FixedFrameBroadcaster(Node):

    def __init__(self):
        super().__init__('camera_frame_broadcaster')

        # declare and get parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_count', 1),
                ('extrinsics_path', ""),
                ('extrinsics_prefix', ""),
                ('extrinsics_file', ""),
            ]
        )
        self.camera_count = self.get_parameter("camera_count").get_parameter_value().integer_value
        self.extrinsics_path = self.get_parameter("extrinsics_path").get_parameter_value().string_value
        self.extrinsics_prefix = self.get_parameter("extrinsics_prefix").get_parameter_value().string_value
        self.extrinsics_file = self.get_parameter("extrinsics_file").get_parameter_value().string_value
        self.get_logger().info(f"Number of cameras: {self.camera_count}")
        self.get_logger().info(f"Extrinsics path: {self.extrinsics_path}")
        self.get_logger().info(f"Extrinsics prefix: {self.extrinsics_prefix}")
        self.get_logger().info(f"Extrinsics file: {self.extrinsics_file}")

        # read camera poses from extrinsic files
        if self.load_extrinsics():
            # create TF broadcaster and timer if successful
            self.tf_broadcaster = TransformBroadcaster(self)
            # self.timer = self.create_timer(1/30.0, self.broadcast_timer_callback) # WHAT RATE DO WE WANT TO PUBLISH?

            self.tf_timers = [self.create_timer(1/30.0, lambda i=i: self.broadcast_camera_transform(i)) for i in range(self.camera_count)]
            # for i in range(self.camera_count):
            #     timer = self.create_timer(1/30.0, lambda i=i: self.broadcast_camera_transform(i))
            #     self.tf_timers.append(timer)

    def load_extrinsics(self):
        self.camera_transforms: list[np.ndarray] = []
        for i in range(self.camera_count):
            self.pkg_path = '/'.join(get_package_share_directory('thermal_camera').split('/')[:-4]+['src','thermal_camera'])
            file_path = os.path.join(self.pkg_path, self.extrinsics_path, f'{self.extrinsics_prefix}{i}', f'{self.extrinsics_file}.pkl')
            try:
                if not os.path.exists(file_path):
                    self.get_logger().error(f"✗ Extrinsics file not found: {file_path}")
                    return None

                # get camera transform and convert rotation to quaternion
                with open(file_path, 'rb') as f:
                    calibration_data = pickle.load(f)
                t = calibration_data['camera_to_base_transform']
                quat = R.from_matrix(t[:3,:3]).as_quat()
                self.camera_transforms.append([t[:3,3],quat])

            except Exception as e:
                self.get_logger().error(f"✗ Error loading camera transforms: {e}")
                return False
        
        self.get_logger().info("Loaded all camera transforms.")
        return True

    def broadcast_camera_transform(self, i):
        # broadcasts a transform for the camera frame relative to the world frame to TF 
        now = (self.get_clock().now() - rclpy.duration.Duration(seconds=0.1)).to_msg()
        
        # Send base frame
        t_base = TransformStamped()
        t_base.header.stamp = now
        t_base.header.frame_id = 'base_link'
        t_base.child_frame_id = f'thermal_camera_{i}'
        t_base.transform.translation.x = self.camera_transforms[i][0][0]
        t_base.transform.translation.y = self.camera_transforms[i][0][1]
        t_base.transform.translation.z = self.camera_transforms[i][0][2]
        t_base.transform.rotation.x = self.camera_transforms[i][1][0]
        t_base.transform.rotation.y = self.camera_transforms[i][1][1]
        t_base.transform.rotation.z = self.camera_transforms[i][1][2]
        t_base.transform.rotation.w = self.camera_transforms[i][1][3]
        self.tf_broadcaster.sendTransform(t_base)

        # Send optical frame
        t_optical = TransformStamped()
        t_optical.header.stamp = now
        t_optical.header.frame_id = f'thermal_camera_{i}'
        t_optical.child_frame_id = f'thermal_camera_{i}_optical_frame'
        t_optical.transform.translation.x = 0.
        t_optical.transform.translation.y = 0.
        t_optical.transform.translation.z = 0.
        t_optical.transform.rotation.x = 0.
        t_optical.transform.rotation.y = 0.
        t_optical.transform.rotation.z = 0.
        t_optical.transform.rotation.w = 1.
        self.tf_broadcaster.sendTransform(t_optical)

    # def broadcast_timer_callback(self):
    #     # broadcasts a transform for the camera frame relative to the world frame to TF 
    #     t = TransformStamped()

    #     for i in range(self.camera_count):
    #         # send base frame
    #         t.header.stamp = self.get_clock().now().to_msg()
    #         t.header.frame_id = 'base_link'
    #         t.child_frame_id = f'thermal_camera_{i}'
    #         t.transform.translation.x = self.camera_transforms[i][0][0]
    #         t.transform.translation.y = self.camera_transforms[i][0][1]
    #         t.transform.translation.z = self.camera_transforms[i][0][2]
    #         t.transform.rotation.x = self.camera_transforms[i][1][0]
    #         t.transform.rotation.y = self.camera_transforms[i][1][1]
    #         t.transform.rotation.z = self.camera_transforms[i][1][2]
    #         t.transform.rotation.w = self.camera_transforms[i][1][3]

    #         self.tf_broadcaster.sendTransform(t)

    #         # send optical frame
    #         t.header.stamp = self.get_clock().now().to_msg()
    #         t.header.frame_id = f'thermal_camera_{i}'
    #         t.child_frame_id = f'thermal_camera_{i}_optical_frame'
    #         t.transform.translation.x = 0.
    #         t.transform.translation.y = 0.
    #         t.transform.translation.z = 0.
    #         t.transform.rotation.x = 0.
    #         t.transform.rotation.y = 0.
    #         t.transform.rotation.z = 0.
    #         t.transform.rotation.w = 1.

    #         self.tf_broadcaster.sendTransform(t)


def main():
    rclpy.init()
    node = FixedFrameBroadcaster()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    rclpy.shutdown()
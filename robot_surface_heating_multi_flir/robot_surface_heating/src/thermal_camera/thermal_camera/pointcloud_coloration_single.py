import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2, CameraInfo
from std_msgs.msg import Header
from sensor_msgs.msg import PointField

from cv_bridge import CvBridge
import message_filters
import numpy as np
import struct
import tf2_ros
import tf2_py
import tf2_geometry_msgs

from geometry_msgs.msg import TransformStamped
from sensor_msgs_py import point_cloud2

from scipy.spatial.transform import Rotation

class PointCloudProjector(Node):
    def __init__(self):
        super().__init__('image_projector')

        # Sync subscribers
        image_sub = message_filters.Subscriber(self, Image, '/thermal_camera_0/image_raw')
        info_sub = message_filters.Subscriber(self, CameraInfo, '/thermal_camera_0/camera_info')
        cloud_sub = message_filters.Subscriber(self, PointCloud2, '/cad_pointcloud')

        self.ts = message_filters.ApproximateTimeSynchronizer(
            [image_sub, info_sub, cloud_sub], queue_size=10, slop=0.1)
        self.ts.registerCallback(self.callback)

        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.pub = self.create_publisher(PointCloud2, '/colored_cad_pointcloud', 1)

        self.get_logger().info('PointCloud projector initialized.')

    def callback(self, image_msg, cam_info_msg, cloud_msg):
        try:
            # Get transform from point cloud frame to camera frame
            transform = self.tf_buffer.lookup_transform(
                target_frame=cam_info_msg.header.frame_id,
                source_frame=cloud_msg.header.frame_id,
                # time=cloud_msg.header.stamp,
                time=rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0)
            )
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {str(e)}")
            return

        fx = cam_info_msg.k[0]
        fy = cam_info_msg.k[4]
        cx = cam_info_msg.k[2]
        cy = cam_info_msg.k[5]
        intrinsics = (fx, fy, cx, cy)

        # Convert PointCloud2 to Nx3 numpy array
        points = []
        for pt in point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            points.append([pt[0], pt[1], pt[2]])
        points = np.array(points)

        if points.shape[0] == 0:
            self.get_logger().warn("Empty point cloud")
            return

        # Transform points to camera frame
        points_cam = self.transform_points(points, transform)

        # Convert image to OpenCV
        img = self.bridge.imgmsg_to_cv2(image_msg, desired_encoding='rgb8')

        # Project and color
        points_visible, colors = self.project_points_to_image(points_cam, img, intrinsics)

        # Publish colored point cloud
        cloud_colored = self.create_colored_pointcloud(points_visible, colors, cam_info_msg.header)
        self.pub.publish(cloud_colored)

    def transform_points(self, points, transform):
        # Extract rotation and translation
        t = transform.transform.translation
        q = transform.transform.rotation
        trans = np.array([t.x, t.y, t.z])
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
        R = rot.as_matrix()
        return (R @ points.T).T + trans

    def project_points_to_image(self, points, image, intrinsics):
        """Project the pointcloud onto the image."""
        fx, fy, cx, cy = intrinsics
        height, width = image.shape[:2]

        valid = points[:, 2] > 0
        points = points[valid]
        u = (fx * points[:, 0] / points[:, 2]) + cx
        v = (fy * points[:, 1] / points[:, 2]) + cy

        u = u.astype(int)
        v = v.astype(int)

        mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u = u[mask]
        v = v[mask]
        points = points[mask]

        colors = image[v, u]  # RGB values
        return points, colors

    def create_colored_pointcloud(self, points, colors, header_in):
        """Store pointcloud in a PointCloud2 message for publishing."""
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
        ]

        cloud_data = []
        for p, c in zip(points, colors):
            r, g, b = c
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
            cloud_data.append(struct.pack('fffI', *p, rgb))

        msg = PointCloud2()
        msg.header = header_in
        msg.height = 1
        msg.width = len(points)
        msg.fields = fields
        msg.is_bigendian = False
        msg.point_step = 16
        msg.row_step = msg.point_step * len(points)
        msg.data = b''.join(cloud_data)
        msg.is_dense = True
        return msg


def main(args=None):
    rclpy.init(args=args)
    node = PointCloudProjector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
import rclpy
import rclpy.node
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import numpy as np
import cv2
import tf_transformations
from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from geometry_msgs.msg import PoseArray, Pose
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

import open3d as o3d
import message_filters

class TIRPointCloudNode(rclpy.node.Node):
    def __init__(self):
        super().__init__('tir_pointcloud_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_count', 4),
                ('camera_intrinsics_path', ""),
                ('camera_extrinsics_path', ""),
                ('depth_baseline', 0.1),
            ]
        )
        
        self.camera_count = self.get_parameter('camera_count').get_parameter_value().integer_value
        self.bridge = CvBridge()

        self.intrinsics = self.load_intrinsics()
        self.extrinsics = self.load_extrinsics()

        # Subscribe to synchronized image topics
        # image_subs = [message_filters.Subscriber(self, Image, f"/{cid}/image_raw") for cid in range(self.camera_count)]
        # self.ts = message_filters.ApproximateTimeSynchronizer(image_subs, queue_size=10, slop=0.1)
        # self.ts.registerCallback(self.image_callback)

        depth_sub = message_filters.Subscriber(self, Image, "/depth/image_raw")
        thermal_sub = message_filters.Subscriber(self, Image, "/lepton0/image_raw")

        self.sync = message_filters.ApproximateTimeSynchronizer(
            [depth_sub, thermal_sub], queue_size=10, slop=0.1
        )
        self.sync.registerCallback(self.depth_thermal_callback)

        self.pc_pub = self.create_publisher(PointCloud2, "/thermal/pointcloud", 10)

    def load_intrinsics(self):
        rel_intrinsics_path = self.get_parameter('camera_intrinsics_path').get_parameter_value().string_value
        return eval(intrinsics)  # For prototype; use YAML safely in production

    def load_extrinsics(self):
        extrinsics = self.get_parameter('camera_extrinsics').get_parameter_value().string_value
        return eval(extrinsics)

    def image_callback(self, *images):
        # Convert to OpenCV format
        cv_images = [self.bridge.imgmsg_to_cv2(img, desired_encoding='mono8') for img in images]
        # Preprocess (CLAHE, etc.)
        cv_images = [cv2.equalizeHist(img) for img in cv_images]

        # Compute depth map via stereo or MVS
        depth_map = self.compute_depth_from_views(cv_images)

        # Backproject to point cloud
        pcd = self.backproject_to_pointcloud(depth_map)

        # Convert to PointCloud2 and publish
        ros_pc = self.open3d_to_ros(pcd, images[0].header.stamp)
        self.pc_pub.publish(ros_pc)

    def depth_thermal_callback(self, depth_msg, thermal_msg):
        depth = self.bridge.imgmsg_to_cv2(depth_msg, "passthrough")  # In meters
        thermal = self.bridge.imgmsg_to_cv2(thermal_msg, "mono8")

        # Resize thermal to match depth if needed
        if thermal.shape != depth.shape:
            thermal = cv2.resize(thermal, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Backproject depth to 3D
        K = self.depth_intrinsics  # 3x3 matrix
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        h, w = depth.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth.flatten()
        x = (u.flatten() - cx) * z / fx
        y = (v.flatten() - cy) * z / fy
        points = np.vstack((x, y, z)).T

        # apply 
        thermal_colored = cv2.applyColorMap(thermal, cv2.COLORMAP_JET)
        thermal_colored = cv2.cvtColor(thermal_colored, cv2.COLOR_BGR2RGB)
        thermal_color = thermal_colored.reshape(-1, 3) / 255.0

        # Normalize thermal image to [0,1] and use as grayscale color
        thermal_norm = thermal.astype(np.float32).flatten() / 255.0
        thermal_color = np.stack([thermal_norm]*3, axis=1)  # Grayscale RGB

        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(thermal_color)

        # Convert and publish
        ros_pc = self.open3d_to_ros(pcd, depth_msg.header.stamp)
        self.pc_pub.publish(ros_pc)

    def compute_depth_from_views(self, images):
        # Minimal stereo or plane sweep version (can be improved)
        # Placeholder: return dummy constant depth
        h, w = images[0].shape
        return np.ones((h, w), dtype=np.float32) * 1.0

    def backproject_to_pointcloud(self, depth_map):
        K = np.array(self.intrinsics[self.camera_ids[0]])
        fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
        h, w = depth_map.shape
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        z = depth_map.flatten()
        x = (u.flatten() - cx) * z / fx
        y = (v.flatten() - cy) * z / fy

        pts = np.vstack((x, y, z)).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        return pcd

    def open3d_to_ros(self, pcd, stamp):
        # Convert Open3D point cloud to ROS2 PointCloud2 message
        from sensor_msgs.msg import PointCloud2, PointField
        from std_msgs.msg import Header
        import struct

        points = np.asarray(pcd.points)
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        ]
        header = Header()
        header.stamp = stamp.to_msg()
        header.frame_id = 'map'
        data = b''.join([struct.pack('fff', *p) for p in points])

        return PointCloud2(
            header=header,
            height=1,
            width=len(points),
            fields=fields,
            is_bigendian=False,
            point_step=12,
            row_step=12 * len(points),
            is_dense=True,
            data=data
        )

def main(args=None):
    rclpy.init(args=args)
    node = TIRPointCloudNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, PointCloud2, CameraInfo, PointField
from std_msgs.msg import Header

from cv_bridge import CvBridge
import message_filters
import numpy as np
import struct
import tf2_ros
import tf2_py
import tf2_geometry_msgs

from sensor_msgs_py import point_cloud2
from geometry_msgs.msg import TransformStamped
from thermal_camera_interfaces.msg import RobotMask
from scipy.spatial.transform import Rotation
import cv2
import os
import time

class PointCloudProjector(Node):
    def __init__(self):
        super().__init__('pointcloud_projector_multi_camera')

        # Initialize stored data to None
        self.pcl_info = None
        self.camera_info = [None] * 4
        self.latest_robot_mask = None
        self.callback_run_once = False

        # Declare and read parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_count', 4),
                ('source_pointcloud_topic', '/cad_pointcloud'),
            ]
        )

        self.camera_count = self.get_parameter('camera_count').get_parameter_value().integer_value
        self.pcl_topic = self.get_parameter('source_pointcloud_topic').get_parameter_value().string_value
        self.bridge = CvBridge()

        # === OPTIMIZATION ===
        # Pointcloud subscriber
        self.pcl_sub = self.create_subscription(
            PointCloud2,
            self.pcl_topic,
            self.pcl_callback,
            10
        )
        
        # Camera info subscribers
        self.camera0_info_sub = self.create_subscription(
            CameraInfo,
            '/thermal_camera_0/camera_info',
            self.camera0_info_callback,
            10
        )

        self.camera1_info_sub = self.create_subscription(
            CameraInfo,
            '/thermal_camera_1/camera_info',
            self.camera1_info_callback,
            10
        )

        self.camera2_info_sub = self.create_subscription(
            CameraInfo,
            '/thermal_camera_2/camera_info',
            self.camera2_info_callback,
            10
        )
    
        self.camera3_info_sub = self.create_subscription(
            CameraInfo,
            '/thermal_camera_3/camera_info',
            self.camera3_info_callback,
            10
        )

        # robot_mask subscribe
        self.robot_mask_subscriber = self.create_subscription(
            RobotMask,
            "robot_mask",
            self.robot_mask_callback,
            10
        )   

        # Subscribers for each camera's image and camera info
        self.image_subs = [
            message_filters.Subscriber(self, Image, f'/thermal_camera_{i}/image_raw')
            for i in range(self.camera_count)
        ]
        # self.info_subs = [
        #     message_filters.Subscriber(self, CameraInfo, f'/thermal_camera_{i}/camera_info')
        #     for i in range(self.camera_count)
        # ]

        # Point cloud subscriber
        # self.cloud_sub = message_filters.Subscriber(self, PointCloud2, self.pcl_topic)

        # Synchronizer
        self.ts = message_filters.ApproximateTimeSynchronizer([
            *self.image_subs, 
            # *self.info_subs, 
            # self.cloud_sub
            ],
            queue_size=10,
            slop=0.2
        )
        # self.ts.registerCallback(self.callback)
        self.ts.registerCallback(self.callback_z_buffer)

        # TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Publisher for raw thermal point cloud
        self.thermal_pub = self.create_publisher(PointCloud2, '/raw_thermal_pointcloud', 1)

        # Publisher for masked image grid (for debug)
        self.masked_grid_pub = self.create_publisher(Image, "/thermal_masked_grid", 1)
        self.bridge = CvBridge()

        self.get_logger().info('Multi-camera PointCloud Projector initialized.')

    # Pointcloud callback
    def pcl_callback(self, msg: PointCloud2):
        # Store pointcloud data once
        if self.pcl_info is None:
            self.pcl_info = msg

            cloud_data = list(point_cloud2.read_points(
                msg,
                field_names=("x", "y", "z", "normal_x", "normal_y", "normal_z"),
                skip_nans=True
            ))

            self.points = np.array([[x, y, z] for x, y, z, _, _, _ in cloud_data])
            self.normals = np.array([[nx, ny, nz] for _, _, _, nx, ny, nz in cloud_data])
    
    # Camera info callbacks
    def camera0_info_callback(self, msg: CameraInfo):
        if self.camera_info[0] is None:
            self.camera_info[0] = msg

    def camera1_info_callback(self, msg: CameraInfo):
        if self.camera_info[1] is None:
            self.camera_info[1] = msg

    def camera2_info_callback(self, msg: CameraInfo):
        if self.camera_info[2] is None:
            self.camera_info[2] = msg

    def camera3_info_callback(self, msg: CameraInfo):
        if self.camera_info[3] is None:
            self.camera_info[3] = msg

    # Mask callback
    def robot_mask_callback(self, msg: RobotMask):
        self.latest_robot_mask = msg

    def callback(self, *msgs):
        # Split the messages
        images = msgs[:self.camera_count]
        infos = msgs[self.camera_count:self.camera_count*2]
        cloud_msg = msgs[-1]

        # Extract point cloud
        points = np.array([
            [pt[0], pt[1], pt[2]]
            for pt in point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        ])
        if points.shape[0] == 0:
            self.get_logger().warn("Empty point cloud")
            return

        # Prepare color assignment
        final_colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
        visibility_mask = np.zeros((points.shape[0]), dtype=bool)

        for i in range(self.camera_count):
            try:
                # cloud_time = cloud_msg.header.stamp
                # if not self.tf_buffer.can_transform(
                #     infos[i].header.frame_id,
                #     cloud_msg.header.frame_id,
                #     cloud_time,
                #     timeout=rclpy.duration.Duration(seconds=0.5)
                # ):
                #     self.get_logger().warn(f"Transform not available for camera {i}")
                #     continue

                # transform = self.tf_buffer.lookup_transform(
                #     infos[i].header.frame_id,
                #     cloud_msg.header.frame_id,
                #     cloud_time
                # )

                # transform = self.tf_buffer.lookup_transform(
                #     target_frame=infos[i].header.frame_id,
                #     source_frame=cloud_msg.header.frame_id,
                #     time=cloud_msg.header.stamp,
                #     timeout=rclpy.duration.Duration(seconds=0.5)
                # )

                transform = self.tf_buffer.lookup_transform(
                    target_frame=infos[i].header.frame_id,
                    source_frame=cloud_msg.header.frame_id,
                    time=rclpy.time.Time(),  # Use latest transform available
                    timeout=rclpy.duration.Duration(seconds=0.5)
                )
            except Exception as e:
                self.get_logger().warn(f"TF lookup failed for camera {i}: {e}")
                continue

            fx, fy, cx, cy = infos[i].k[0], infos[i].k[4], infos[i].k[2], infos[i].k[5]
            intrinsics = (fx, fy, cx, cy)

            img = self.bridge.imgmsg_to_cv2(images[i], desired_encoding='rgb8')
            points_cam = self.transform_points(points, transform)

            # Project points to image
            visible, uvs, mask = self.project(points_cam, img.shape[:2], intrinsics)

            # Assign colors only to points not already colored (or combine if needed)
            new_indices = (~visibility_mask) & mask
            if np.any(new_indices):
                color_values = img[uvs[new_indices, 1], uvs[new_indices, 0]]
                final_colors[new_indices] = color_values
                visibility_mask[new_indices] = True

        # Only keep visible points
        visible_points = points[visibility_mask]
        visible_colors = final_colors[visibility_mask]

        if visible_points.shape[0] == 0:
            self.get_logger().warn("No points visible in any camera.")
            return

        colored_cloud = self.create_colored_pointcloud(visible_points, visible_colors, cloud_msg.header)
        self.colored_pub.publish(colored_cloud)

    def callback_z_buffer(self, *msgs):
        # self.get_logger().info("callback_z_buffer")
        # Tracking time
        start = time.time()
        new_header = Header()
        # new_header.stamp = self.get_clock().now().to_msg()

        # Extract images
        images = msgs[:self.camera_count]
        new_header.stamp = images[0].header.stamp
        # infos = msgs[self.camera_count:self.camera_count*2]
        # cloud_msg = msgs[-1]
        
        # Extract camera info
        if self.camera_info is None:
            return
        infos = self.camera_info
        
        # Extract pointcloud 
        if self.pcl_info is None:
            return
        cloud_msg = self.pcl_info
        new_header.frame_id = cloud_msg.header.frame_id  # reuse the frame ID
        # Extract point cloud
        # points = np.array([
        #     [pt[0], pt[1], pt[2]]
        #     for pt in point_cloud2.read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True)
        # ])

        # Unpack pointcloud 
        points = self.points
        normals = self.normals
        if points.shape[0] == 0:
            self.get_logger().warn("Empty point cloud")
            return
        
        # Exrtract robot mask
        if self.latest_robot_mask is None:
            return
        robot_mask_msg = self.latest_robot_mask
        # Unpack robot mask
        robot_masks = [
            self.bridge.imgmsg_to_cv2(mask_msg, desired_encoding='mono8')
            for mask_msg in robot_mask_msg.masks
        ]

        # Prepare color assignment
        # final_colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
        # visibility_mask = np.zeros((points.shape[0]), dtype=bool)

        # color_accumulator = np.zeros((points.shape[0], 3), dtype=np.float32)
        thermal_accumulator = np.zeros(points.shape[0], dtype=np.float32)
        color_counts = np.zeros((points.shape[0],), dtype=np.int32)
    
        masked_images = [None] * self.camera_count

        for i in range(self.camera_count):
            try:
                transform = self.tf_buffer.lookup_transform(
                    target_frame=infos[i].header.frame_id,
                    source_frame=cloud_msg.header.frame_id,
                    time=rclpy.time.Time(),  # Use latest transform available
                    timeout=rclpy.duration.Duration(seconds=0.5)
                )
            except Exception as e:
                self.get_logger().warn(f"TF lookup failed for camera {i}: {e}")
                continue

            fx, fy, cx, cy = infos[i].k[0], infos[i].k[4], infos[i].k[2], infos[i].k[5]
            intrinsics = (fx, fy, cx, cy)

            # img = self.bridge.imgmsg_to_cv2(images[i], desired_encoding='rgb8')
            thermal_img = self.bridge.imgmsg_to_cv2(images[i], desired_encoding='mono16').astype(np.float32)
            points_cam = self.transform_points(points, transform)

            # Convert thermal image to float32 so we can assign NaNs
            # thermal_img_float = thermal_img.astype(np.float32)
            
            # Resize robot mask to match the thermal image shape
            robot_mask = robot_masks[i]
            robot_mask_resized = cv2.resize(
                robot_mask,
                (thermal_img.shape[1], thermal_img.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )

            # Invalidate robot pixels (where robot_mask == 0)
            thermal_img_masked = thermal_img.copy()
            thermal_img_masked[robot_mask_resized == 0] = np.nan

            # Publish masked image data for visualization
            # Ensure pointcloud is initialized before saving
            if self.pcl_info is not None:
                # timestamp = new_header.stamp.sec + new_header.stamp.nanosec * 1e-9
                # ts_str = f"{timestamp:.3f}"

                # scale = 3.5
                # # Colormap raw thermal image
                # thermal_img_8u = cv2.normalize(
                #     thermal_img,
                #     None,
                #     alpha=0,
                #     beta=255,
                #     norm_type=cv2.NORM_MINMAX,
                #     dtype=cv2.CV_8U
                # )

                # thermal_img_colormap = cv2.applyColorMap(thermal_img_8u, cv2.COLORMAP_JET)
                # masked_colormap = thermal_img_colormap.copy()
                # masked_colormap[robot_mask_resized == 0] = [0, 0 ,0]

                # Colormap mapped thermal image
                # thermal_img_8u = cv2.normalize(
                #     thermal_img_masked,
                #     None,
                #     alpha=0,
                #     beta=255,
                #     norm_type=cv2.NORM_MINMAX,
                #     dtype=cv2.CV_8U
                # )

                # thermal_img_masked_colormap = cv2.applyColorMap(thermal_img_8u, cv2.COLORMAP_JET)

                # Create mask of valid (non-NaN) pixels
                valid_mask = ~np.isnan(thermal_img_masked)

                # Fill NaNs temporarily with 0s to avoid OpenCV errors
                safe_img = thermal_img_masked.copy()
                safe_img[~valid_mask] = 0

                # Normalize using only valid pixels
                valid_pixels = thermal_img_masked[valid_mask]
                if valid_pixels.size == 0:
                    self.get_logger().warn(f"Camera {i}: all pixels masked out, skipping frame")
                    continue  # Skip this frame

                thermal_min = np.nanmin(valid_pixels)
                thermal_max = np.nanmax(valid_pixels)

                # Prevent divide-by-zero
                if thermal_max == thermal_min:
                    thermal_max += 1.0

                # Manual normalization (same as cv2.normalize would do)
                normalized = 255 * (safe_img - thermal_min) / (thermal_max - thermal_min)
                normalized = np.clip(normalized, 0, 255).astype(np.uint8)

                # Apply colormap
                thermal_img_masked_colormap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)

                # scaled_thermal_img_masked_colormap = cv2.resize(
                #     thermal_img_masked_colormap,
                #     dsize=None,
                #     fx=scale,
                #     fy=scale,
                #     interpolation=cv2.INTER_NEAREST
                # )

                # # Save to output directory
                # save_dir = f"/tmp/masked_frames/camera_{i}"
                # os.makedirs(save_dir, exist_ok=True)
                # save_path = os.path.join(save_dir, f"{ts_str}.png")
                # cv2.imwrite(save_path, scaled_thermal_img_masked_colormap)
                # Resize and store masked image for this camera
                masked_images[i] = cv2.resize(thermal_img_masked_colormap, (640, 480))



            def rotate_normals(normals, transform):
                q = transform.transform.rotation
                rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
                return rot.apply(normals)
            normals_cam = rotate_normals(normals, transform)

            # Project with occlusion filtering
            # uvs, mask = self.project_with_occlusion(points_cam, img.shape[:2], intrinsics)
            # uvs, mask = self.project_with_occlusion(points_cam, normals_cam, img.shape[:2], intrinsics)

            # uvs, mask = self.project_with_occlusion_backface(points_cam, normals_cam, thermal_img.shape[:2], intrinsics)
            uvs, mask = self.project_with_occlusion_backface(points_cam, normals_cam, thermal_img_masked.shape[:2], intrinsics)

            # Get indices of visible points in original point array
            indices_visible_in_original = np.nonzero(mask)[0]

            # # Only update colors for points that haven't been assigned yet
            # new_indices = ~visibility_mask[indices_visible_in_original]

            # if np.any(new_indices):
            #     selected_uvs = uvs[new_indices]
            #     selected_indices = indices_visible_in_original[new_indices]

            #     # Sample colors from image
            #     color_values = img[selected_uvs[:, 1], selected_uvs[:, 0]]
            #     final_colors[selected_indices] = color_values
            #     visibility_mask[selected_indices] = True

            ### average the colors across all cameras ###
            selected_uvs = uvs
            selected_indices = indices_visible_in_original

            # # Sample colors
            # color_values = img[selected_uvs[:, 1], selected_uvs[:, 0]].astype(np.float32)

            # # Accumulate color and count
            # color_accumulator[selected_indices] += color_values

            # Accumulate thermal and count
            # thermal_values = thermal_img[selected_uvs[:, 1], selected_uvs[:, 0]].astype(np.float32)

            thermal_values = thermal_img_masked[selected_uvs[:, 1], selected_uvs[:, 0]]
            valid = ~np.isnan(thermal_values)
            selected_uvs = selected_uvs[valid]
            selected_indices = selected_indices[valid]
            thermal_values = thermal_values[valid]

            thermal_accumulator[selected_indices] += thermal_values
            color_counts[selected_indices] += 1

        # Publish masked image grid for debugging
        if all(img is not None for img in masked_images): # Ensure all 4 cameras provided frames
            top_row = cv2.hconcat([masked_images[0], masked_images[1]])
            bottom_row = cv2.hconcat([masked_images[2], masked_images[3]])
            grid = cv2.vconcat([top_row, bottom_row])

            ros_img = self.bridge.cv2_to_imgmsg(grid, encoding="bgr8")
            ros_img.header.stamp = self.get_clock().now().to_msg()
            self.masked_grid_pub.publish(ros_img)

        # Only keep visible points
        # visible_points = points[visibility_mask]
        # visible_colors = final_colors[visibility_mask]

        valid_mask = color_counts > 0
        # averaged_colors = np.zeros_like(color_accumulator, dtype=np.uint8)
        # averaged_colors[valid_mask] = (color_accumulator[valid_mask] / color_counts[valid_mask, None]).astype(np.uint8)
        averaged_thermal = np.zeros_like(thermal_accumulator)
        averaged_thermal[valid_mask] = thermal_accumulator[valid_mask] / color_counts[valid_mask]

        visible_points = points[valid_mask]
        visible_thermal = averaged_thermal[valid_mask]

        valid_values = averaged_thermal[valid_mask]
        if valid_values.size == 0:
            self.get_logger().warn("No valid thermal data after filtering, skipping frame")
            return

        # colored_cloud = self.create_raw_thermal_pointcloud(visible_points, visible_thermal, cloud_msg.header)
        colored_cloud = self.create_raw_thermal_pointcloud(visible_points, visible_thermal, new_header)
        self.thermal_pub.publish(colored_cloud)
        # self.get_logger().warn(f'Processing_time: {time.time()-start}')

        # # Normalize to 0-255 for Jet colormap
        # valid_values = averaged_thermal[valid_mask]
        # if valid_values.size == 0:
        #     self.get_logger().warn("No valid thermal data after filtering, skipping frame")
        #     return
        # thermal_min, thermal_max = np.min(valid_values), np.max(valid_values)
        # # thermal_min, thermal_max = np.min(averaged_thermal[valid_mask]), np.max(averaged_thermal[valid_mask])
        # norm_thermal = np.zeros_like(averaged_thermal, dtype=np.uint8)
        # norm_thermal[valid_mask] = np.clip(
        #     255 * (averaged_thermal[valid_mask] - thermal_min) / (thermal_max - thermal_min),
        #     0, 255
        # ).astype(np.uint8)

        # # Apply colormap
        # jet_colors = np.zeros((len(points), 3), dtype=np.uint8)
        # bgr = cv2.applyColorMap(norm_thermal[valid_mask], cv2.COLORMAP_JET)
        # rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        # jet_colors[valid_mask] = rgb.reshape(-1, 3)

        # visible_points = points[valid_mask]
        # # visible_colors = averaged_colors[valid_mask]
        # visible_colors = jet_colors[valid_mask]

        # if visible_points.shape[0] == 0:
        #     self.get_logger().warn("No points visible in any camera.")
        #     return

        # colored_cloud = self.create_colored_pointcloud(visible_points, visible_colors, cloud_msg.header)
        # self.colored_pub.publish(colored_cloud)


    def transform_points(self, points, transform):
        """Apply transform to point cloud."""
        t = transform.transform.translation
        q = transform.transform.rotation
        trans = np.array([t.x, t.y, t.z])
        rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
        R = rot.as_matrix()
        return (R @ points.T).T + trans

    def project(self, points, image_shape, intrinsics):
        """Project 3D points to 2D image."""
        fx, fy, cx, cy = intrinsics
        height, width = image_shape

        valid_z = points[:, 2] > 0
        points = points[valid_z]

        u = (fx * points[:, 0] / points[:, 2] + cx).astype(int)
        v = (fy * points[:, 1] / points[:, 2] + cy).astype(int)

        mask = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        uvs = np.stack([u, v], axis=-1)

        full_mask = np.zeros((valid_z.shape[0],), dtype=bool)
        full_mask[valid_z] = mask

        uvs_full = np.zeros((valid_z.shape[0], 2), dtype=int)
        uvs_full[valid_z] = uvs

        return points[mask], uvs_full, full_mask

    def project_with_occlusion(self, points_cam, normals_cam, image_shape, intrinsics):
        fx, fy, cx, cy = intrinsics
        height, width  = image_shape

        # ---------- keep only points in front of the camera ----------
        z = points_cam[:, 2]
        valid_z       = z > 0
        points_valid  = points_cam[valid_z]
        if points_valid.size == 0:
            return np.empty((0, 2), dtype=int), np.zeros(points_cam.shape[0], dtype=bool)

        # ---------- pinhole projection ----------
        u = (fx * points_valid[:, 0] / points_valid[:, 2] + cx).astype(int)
        v = (fy * points_valid[:, 1] / points_valid[:, 2] + cy).astype(int)

        # ---------- image bounds filter ----------
        in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u, v, z = u[in_img], v[in_img], points_valid[in_img, 2]
        if u.size == 0:
            return np.empty((0, 2), dtype=int), np.zeros(points_cam.shape[0], dtype=bool)

        # indices in the original point array
        indices_in_points = np.nonzero(valid_z)[0][in_img]

        # ---------- vectorised Z-buffer ----------
        linear_idx = v * width + u
        order      = np.argsort(z)                 # nearest-first
        linear_idx_sorted = linear_idx[order]
        indices_sorted    = indices_in_points[order]
        u_sorted, v_sorted = u[order], v[order]

        # first (nearest) occurrence of every pixel
        _, first = np.unique(linear_idx_sorted, return_index=True)
        chosen_indices = indices_sorted[first]
        chosen_uvs     = np.column_stack((u_sorted[first], v_sorted[first]))

        # ---------- **sort by ascending point-index so rows match** ----------
        sort_rows           = np.argsort(chosen_indices)
        chosen_indices      = chosen_indices[sort_rows]
        chosen_uvs          = chosen_uvs[sort_rows]

        # full mask for caller
        mask = np.zeros(points_cam.shape[0], dtype=bool)
        mask[chosen_indices] = True

        return chosen_uvs, mask

    def project_with_occlusion_backface(self, points_cam, normals_cam, image_shape, intrinsics):
        fx, fy, cx, cy = intrinsics
        height, width  = image_shape

        z = points_cam[:, 2]
        valid_z = z > 0

        points_valid = points_cam[valid_z]
        normals_valid = normals_cam[valid_z]

        if points_valid.shape[0] == 0:
            return np.empty((0, 2), dtype=int), np.zeros(points_cam.shape[0], dtype=bool)

        # --- Compute view directions and apply backface culling ---
        view_dirs = -points_valid / np.linalg.norm(points_valid, axis=1, keepdims=True)
        dot = np.einsum('ij,ij->i', normals_valid, view_dirs)

        # facing_camera = dot > 0.1  # backface culling threshold
        facing_camera = dot > -1.0

        points_valid = points_valid[facing_camera]
        normals_valid = normals_valid[facing_camera]

        if points_valid.shape[0] == 0:
            return np.empty((0, 2), dtype=int), np.zeros(points_cam.shape[0], dtype=bool)

        # --- Project ---
        u = (fx * points_valid[:, 0] / points_valid[:, 2] + cx).astype(int)
        v = (fy * points_valid[:, 1] / points_valid[:, 2] + cy).astype(int)

        in_img = (u >= 0) & (u < width) & (v >= 0) & (v < height)
        u, v, z = u[in_img], v[in_img], points_valid[in_img, 2]
        if u.size == 0:
            return np.empty((0, 2), dtype=int), np.zeros(points_cam.shape[0], dtype=bool)

        # Indices in original points array
        original_indices = np.nonzero(valid_z)[0]
        indices_facing   = original_indices[facing_camera]
        indices_in_image = indices_facing[in_img]

        # Z-buffer
        linear_idx = v * width + u
        order = np.argsort(z)  # nearest first
        linear_idx_sorted = linear_idx[order]
        indices_sorted = indices_in_image[order]
        u_sorted, v_sorted = u[order], v[order]

        _, first = np.unique(linear_idx_sorted, return_index=True)
        chosen_indices = indices_sorted[first]
        chosen_uvs = np.column_stack((u_sorted[first], v_sorted[first]))

        # Sort to align rows
        sort_rows = np.argsort(chosen_indices)
        chosen_indices = chosen_indices[sort_rows]
        chosen_uvs = chosen_uvs[sort_rows]

        # Create full mask
        mask = np.zeros(points_cam.shape[0], dtype=bool)
        mask[chosen_indices] = True

        return chosen_uvs, mask

    # def create_colored_pointcloud(self, points, colors, header_in):
    #     """Store point cloud in PointCloud2 message format."""
    #     fields = [
    #         PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    #         PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    #         PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
    #         PointField(name='rgb', offset=12, datatype=PointField.UINT32, count=1),
    #     ]

    #     cloud_data = []
    #     for p, c in zip(points, colors):
    #         r, g, b = c
    #         rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 255))[0]
    #         cloud_data.append(struct.pack('fffI', *p, rgb))

    #     msg = PointCloud2()
    #     msg.header = header_in
    #     msg.height = 1
    #     msg.width = len(points)
    #     msg.fields = fields
    #     msg.is_bigendian = False
    #     msg.point_step = 16
    #     msg.row_step = msg.point_step * len(points)
    #     msg.data = b''.join(cloud_data)
    #     msg.is_dense = True
    #     return msg
    
    def create_raw_thermal_pointcloud(self, points, values, header_in):
        """Store point cloud in PointCloud2 message format."""
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='thermal', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        cloud_data = [
            struct.pack('ffff', *p, t)
            for p, t in zip(points, values)
        ]

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

# def main(args=None):
#     rclpy.init(args=args)
#     node = PointCloudProjector()
#     rclpy.spin(node)
#     node.destroy_node()
#     rclpy.shutdown()


class PointCloudColoredNode(Node):
    def __init__(self):
        super().__init__('pointcloud_coloration_node')

        # Publisher for colored point cloud
        self.colored_pub = self.create_publisher(PointCloud2, '/colored_cad_pointcloud', 1)

        self.declare_parameters(
            namespace='',
            parameters=[
                ('source_pointcloud_topic', '/cad_pointcloud'),
            ]
        )
        self.pcl_topic = self.get_parameter('source_pointcloud_topic').get_parameter_value().string_value
        self.subscription = self.create_subscription(
            PointCloud2,
            self.pcl_topic,
            self.pointcloud_callback,
            1
        )

        self.get_logger().info("Pointcloud heatmap visualization node initialized.")

    def pointcloud_callback(self, msg: PointCloud2):
        points = []
        temperatures = []
        start_time = time.time()

        # Extract and decode point cloud data
        for p in point_cloud2.read_points(msg, field_names=("x", "y", "z", "thermal"), skip_nans=True):
            x, y, z, temp = p
            points.append([x, y, z])
            temperatures.append(temp)
        points = np.asarray(points)
        temp_arr = np.array(temperatures)

        # Normalize to 0-255 for Jet colormap
        if points.shape[0] == 0:
            self.get_logger().warn("No valid thermal data, skipping frame")
            return
        thermal_min, thermal_max = np.min(temp_arr), np.max(temp_arr)
        # thermal_min, thermal_max = np.min(averaged_thermal[valid_mask]), np.max(averaged_thermal[valid_mask])
        norm_thermal = np.zeros_like(temp_arr, dtype=np.uint8)
        norm_thermal = np.clip(
            255 * (temp_arr - thermal_min) / (thermal_max - thermal_min),
            0, 255
        ).astype(np.uint8)

        # Apply colormap
        jet_colors = np.zeros((len(points), 3), dtype=np.uint8)
        bgr = cv2.applyColorMap(norm_thermal, cv2.COLORMAP_JET)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        jet_colors = rgb.reshape(-1, 3)

        visible_points = points
        # visible_colors = averaged_colors[valid_mask]
        visible_colors = jet_colors

        if visible_points.shape[0] == 0:
            self.get_logger().warn("No points visible in any camera.")
            return

        colored_cloud = self.create_colored_pointcloud(visible_points, visible_colors, msg.header)
        self.colored_pub.publish(colored_cloud)
        # self.get_logger().warn(f"Pointcloud colored: {time.time() - start_time}")

    def create_colored_pointcloud(self, points, colors, header_in):
        fields = [
            PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
            PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
            PointField(name='rgb', offset=12, datatype=PointField.FLOAT32, count=1),
        ]

        def rgb_to_float(r, g, b):
            rgb = (int(r) << 16) | (int(g) << 8) | int(b)
            return struct.unpack('f', struct.pack('I', rgb))[0]

        cloud_points = []
        for p, c in zip(points, colors):
            r, g, b = c
            cloud_points.append([p[0], p[1], p[2], rgb_to_float(r, g, b)])

        return point_cloud2.create_cloud(header_in, fields, cloud_points)

from rclpy.executors import MultiThreadedExecutor

def main(args=None):
    rclpy.init(args=args)

    raw_pcl = PointCloudProjector()
    colored_pcl = PointCloudColoredNode()

    # Create a MultiThreadedExecutor or SingleThreadedExecutor
    executor = MultiThreadedExecutor()
    executor.add_node(raw_pcl)
    executor.add_node(colored_pcl)

    try:
        executor.spin()
    finally:
        raw_pcl.destroy_node()
        colored_pcl.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()

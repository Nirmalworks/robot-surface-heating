import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Pose, PoseArray
from cv_bridge import CvBridge
import cv2
import numpy as np
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from scipy import stats

class CentroidPoseNode(Node):
    def __init__(self):
        super().__init__('realsense_calib_test')

        # extract parameters
        color_image_topic = "/camera/camera/color/image_raw"
        self.get_logger().info(f"Color image topic: {color_image_topic}")

        depth_image_topic = "/camera/camera/aligned_depth_to_color/image_raw"
        self.get_logger().info(f"Depth image topic: {depth_image_topic}")

        info_topic = "/camera/camera/color/camera_info"
        self.get_logger().info(f"Image info topic: {info_topic}")

        self.camera_frame = "camera_link"
        self.get_logger().info(f"Camera frame: {self.camera_frame}")

        # Initialize CvBridge
        self.bridge = CvBridge()

        # Initialize intrinsic parameters to None
        self.fx = self.fy = self.cx = self.cy = None
        self.depth_image = None

        # Create a subscriber for the image topic
        self.image_subscription = self.create_subscription(
            Image,
            color_image_topic,
            self.rgb_image_callback,
            10
        )

        self.depth_image_subscription = self.create_subscription(
            Image,
            depth_image_topic,  # Change this to your actual depth image topic
            self.depth_image_callback,
            10
        )

        self.camera_info_subscription = self.create_subscription(
            CameraInfo,
            info_topic,  # Change this to your actual CameraInfo topic
            self.camera_info_callback,
            10
        )

        self.image_subscription  # prevent unused variable warning
        self.depth_image_subscription  # prevent unused variable warning
        self.camera_info_subscription  # prevent unused variable warning

    def camera_info_callback(self, msg):
        # Update intrinsic parameters from CameraInfo message
        self.fx = msg.k[0]  # Focal length x
        self.fy = msg.k[4]  # Focal length y
        self.cx = msg.k[2]  # Principal point x
        self.cy = msg.k[5]  # Principal point y
        self.destroy_subscription(self.camera_info_subscription)
        self.get_logger().info(f"Camera info: fx {self.fx} fy {self.fy} cx {self.cx} cy {self.cy}")

    def depth_image_callback(self, msg):
        # Convert ROS Depth Image message to OpenCV image
        self.depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        # crop depth image
        # self.depth_image = self.depth_image[self.CONTOUR_Y_SEARCH_RANGE[0]:self.CONTOUR_Y_SEARCH_RANGE[1],
        #                                     self.CONTOUR_X_SEARCH_RANGE[0]:self.CONTOUR_X_SEARCH_RANGE[1]]

        # # visualizing depth image
        # depth_image_normalized = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX)
        # depth_image_normalized = np.uint8(depth_image_normalized)

        # # Convert to a color map for better visualization
        # depth_image_color = cv2.applyColorMap(depth_image_normalized, cv2.COLORMAP_JET)

        # # Display the images
        # cv2.imshow('Depth Image (Grayscale)', depth_image_normalized)
        # cv2.imshow('Depth Image (Color)', depth_image_color)


    def rgb_image_callback(self, msg):
        # if None in (self.fx, self.fy, self.cx, self.cy):
        if None in (self.fx, self.fy, self.cx, self.cy) or self.depth_image is None:
            self.get_logger().info('Camera intrinsic parameters or depth image not yet available.', once=True)
            return

        try:
            # Convert ROS RGB Image message to OpenCV image
            rgb_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # pattern_size = (3,9)
            pattern_size = (8,6)
            
            # Capture image
            thermal_img = rgb_image
            # thermal_8bit = raw_to_8bit(thermal_img)
            # thermal_8bit = cv2.normalize(thermal_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            # thermal_inv, _ = prepare_thermal_for_detection(thermal_img)
            # thermal_inv = 255 - thermal_8bit
            gray        = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2GRAY)

            # # Smooth out thermal sensor noise
            # blurred = cv2.GaussianBlur(gray, (5, 5), 0)

            # # Normalize intensities globally (thermal contrast is usually low)
            # normalized = cv2.normalize(blurred, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


            # thermal_inv = 255 - normalized
            thermal_inv = 255 - gray
            
            # # Check detection
            # ret, corners = cv2.findChessboardCorners(
            #     thermal_inv, pattern_size,
            #     flags=cv2.CALIB_CB_ADAPTIVE_THRESH
            # )

            params = cv2.SimpleBlobDetector_Params()
            params.filterByArea = True
            params.minArea = 5
            params.maxArea = 400
            # params.filterByCircularity = False
            params.filterByInertia = False
            params.filterByConvexity = False

            params.minArea = 30
            params.maxArea = 300
            params.minThreshold = 10
            params.maxThreshold = 255
            params.thresholdStep = 10
            params.filterByCircularity = True
            params.minCircularity = 0.3

            detector = cv2.SimpleBlobDetector_create(params)

            # Step 1: Contrast enhancement
            img_eq = cv2.equalizeHist(thermal_inv)

            # Step 2: Gaussian blur (to reduce noise)
            blurred = cv2.GaussianBlur(img_eq, (5, 5), 0)

            # Step 3: Adaptive thresholding (can help with uneven lighting)
            binary = cv2.adaptiveThreshold(
                blurred, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 
                11, 2
            )

            # Optional: Morphological ops to refine blobs
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

            # keypoints = detector.detect(binary)
            keypoints = detector.detect(thermal_inv)
            im_with_keypoints = cv2.drawKeypoints(
                # binary, keypoints, np.array([]), (0, 0, 255),
                thermal_inv, keypoints, np.array([]), (0, 0, 255),
                cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
            )
            cv2.imshow("Blobs", im_with_keypoints)
            cv2.waitKey(1)

            print(f"Detected {len(keypoints)} blobs")

            ret, corners = cv2.findCirclesGrid(
                rgb_image, pattern_size,
                # flags=cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING,
                # flags=cv2.CALIB_CB_ASYMMETRIC_GRID,
                # blobDetector=detector
            )
            
            if ret:
                print(f"✓ Captured")
                
                # Show preview
                # display_data = cv2.resize(thermal_inv[:,:], (640, 480))
                img_preview = cv2.cvtColor(thermal_inv, cv2.COLOR_GRAY2BGR)
                # corners_display = corners * 4 # 160x120 -> 640x480
                cv2.drawChessboardCorners(img_preview, pattern_size, corners, ret)
                cv2.imshow('Captured', img_preview)
                cv2.waitKey(1000)
            else:
                print("✗ Detection failed - try again")
                # display_data = cv2.resize(thermal_inv[:,:], (640, 480))
                cv2.imshow('Failed', thermal_inv)
                cv2.waitKey(1000)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = CentroidPoseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

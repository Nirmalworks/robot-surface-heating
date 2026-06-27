#!/usr/bin/env python
# -*- coding: utf-8 -*-

from thermal_camera.uvctypes import *
import time
import cv2
import numpy as np
try:
  from queue import Queue, Empty
except ImportError:
  from queue import Queue, Empty

import signal
import sys
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import Header
from cv_bridge import CvBridge
# from thermal_camera.msg import Extrema
from ament_index_python.packages import get_package_share_directory

########## Frame Transform Dependencies ##########
import asyncio
from tf2_ros import TransformException
from tf2_ros.buffer import Buffer
from tf2_ros.transform_listener import TransformListener
from scipy.spatial.transform import Rotation as R
########## ########## ##########

def make_callback(index, queue):
  """Set up libuvc callback with ctypes bindings."""
  def callback(frame, userptr):
    if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
      return
    array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
    data = np.frombuffer(array_pointer.contents, dtype=np.uint16).reshape(
      frame.contents.height, frame.contents.width
    )
    if not queue.full():
      queue.put(data)
  return CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(callback)

def ktof(val):
  return (1.8 * ktoc(val) + 32.0)

def ktoc(val):
  return (val - 27315) / 100.0

def raw_to_8bit(data):
  cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
  np.right_shift(data, 8, data)
  return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)

class ThermalCameraPublisher(Node):
    def __init__(self):
        super().__init__('thermal_camera_publisher')
        
        # Declare and read parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_count', 4),
                ('camera_intrinsics_path', ""),
                ('camera_extrinsics_path', ""),
                ('buffer_size', 2)
            ]
        )

        self.camera_count = self.get_parameter('camera_count').get_parameter_value().integer_value
        self.buffer_size = self.get_parameter('buffer_size').get_parameter_value().integer_value
        self.intrinsics_path = self.get_parameter('camera_intrinsics_path').get_parameter_value().string_value
        self.extrinsics_path = self.get_parameter('camera_extrinsics_path').get_parameter_value().string_value

        self.qs = [Queue(self.buffer_size) for _ in range(self.camera_count)]

        self.bridge = CvBridge()
        
        self.load_camera_intrinsics()

        # TF listener for camera extrinsics
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        # self.load_camera_extrinsics()


        # create publishers
        self.thermal_publishers = []
        self.info_publishers = []
        self.extrema_pubs = []
        for i in range(self.camera_count):
            pub_thermal = self.create_publisher(Image, f'/thermal_camera_{i}/image_raw', 10)
            self.thermal_publishers.append(pub_thermal)
            pub_info = self.create_publisher(CameraInfo, f'/thermal_camera_{i}/camera_info', 10)
            self.info_publishers.append(pub_info)
            # extrema_info = self.create_publisher()
            # self.extrema_pubs.append(extrema_info)

        self.timer = self.create_timer(1/30.0, self.publish_images) # WHAT RATE DO WE WANT TO PUBLISH?
        self.info_timer = self.create_timer(1/30., self.publish_info)

        self.camera_initialized = False
        self.init_cameras()
        
        self.get_logger().info(f'Publishing thermal heatmap {self.camera_count}')
        for i in range(self.camera_count):
            self.get_logger().info(f'\tPublishing: /thermal_camera_{i}/image_raw (rgb8 raw)')

    def init_cameras(self):
        """Get camera ports configured, opened, and streaming
        using the libuvc python binding API."""
        try:
            self.ctx = POINTER(uvc_context)()
            self.devs = POINTER(POINTER(uvc_device))()
            self.devh = (POINTER(uvc_device_handle) * self.camera_count)()
            self.ctrls = [uvc_stream_ctrl() for _ in range(self.camera_count)]
            self.callbacks = [make_callback(i, self.qs[i]) for i in range(self.camera_count)]

            res = libuvc.uvc_init(byref(self.ctx), 0)
            if res < 0:
                self.get_logger().error("uvc_init error")
                return False

            res = libuvc.uvc_find_devices(self.ctx, byref(self.devs), PT_USB_VID, PT_USB_PID, 0)
            if res < 0:
                self.get_logger().error("uvc_find_devices error")
                return False

            device_count = 0
            i = 0
            while self.devs[i]:
                device_count += 1
                i += 1

            self.get_logger().info(f'Found {device_count} thermal cameras')
            
            if device_count < self.camera_count:
                self.get_logger().warning(f'Expected {self.camera_count} cameras, found {device_count}')

            for i in range(min(self.camera_count, device_count)):
                res = libuvc.uvc_open(self.devs[i], byref(self.devh[i]))
                if res < 0:
                    self.get_logger().error(f"uvc_open camera {i} error")
                    return False
            for i in range(min(self.camera_count, device_count)):
                frame_format = uvc_get_frame_formats_by_guid(self.devh[i], VS_FMT_GUID_Y16)
                if len(frame_format) == 0:
                    self.get_logger().error(f"Camera {i} does not support Y16")
                    return False
                
                libuvc.uvc_get_stream_ctrl_format_size(
                    self.devh[i], byref(self.ctrls[i]), UVC_FRAME_FORMAT_Y16,
                    frame_format[0].wWidth, frame_format[0].wHeight, 
                    int(1e7 / frame_format[0].dwDefaultFrameInterval)
                )
            for i in range(min(self.camera_count, device_count)):
                res = libuvc.uvc_start_streaming(
                    self.devh[i], byref(self.ctrls[i]), self.callbacks[i], None, 0
                )
                if res < 0:
                    self.get_logger().error(f"uvc_start_streaming camera {i} failed: {res}")
                    return False
                time.sleep(0.5)

            self.camera_initialized = True
            self.get_logger().info('Cameras Initialized')
            return True

        except Exception as e:
            self.get_logger().error(f'Camera initialization failed: {e}')
            return False

    def load_camera_intrinsics(self):
        """Load intrinsic parameters into CameraInfo messages for
        publishing."""
        self.cam_info = [CameraInfo() for _ in range(self.camera_count)]
        for i in range(self.camera_count):
            pkg_path = '/'.join(get_package_share_directory("thermal_camera").split('/')[:-4]+['src','thermal_camera'])
            intrinsic_file = '/'.join([pkg_path, self.intrinsics_path+f'{i}', 'thermal_intrinsics.npz'])

            intrinsic_info = np.load(intrinsic_file)
            self.cam_info[i].header.frame_id = f"thermal_camera_{i}_optical_frame"
            self.cam_info[i].height = 120
            self.cam_info[i].width = 160
            self.cam_info[i].k = intrinsic_info['mtx'].flatten().tolist()  # 3x3 -> 9 values
            self.cam_info[i].d = intrinsic_info['dist'].flatten().tolist()    # 1x5 or 1x8
            self.cam_info[i].r = np.eye(3).flatten().tolist()
            
            # Default P from K
            P = np.zeros((3, 4))
            P[:3, :3] = intrinsic_info['mtx']
            self.cam_info[i].p = P.flatten().tolist()
            self.cam_info[i].distortion_model = "plumb_bob"

        self.get_logger().info("Camera intrinsic info obtained")

    def publish_info(self):
        """Publish camera intrinsic info for each camera."""
        timestamp = self.get_clock().now().to_msg()
        for i in range(self.camera_count):
            self.cam_info[i].header.stamp = timestamp
            self.info_publishers[i].publish(self.cam_info[i])

    def publish_images(self):
        """Convert all heat images to RGB and store in
        Image messages, then publish."""
        if not self.camera_initialized:
            return

        timestamp = self.get_clock().now().to_msg()
        
        for i in range(self.camera_count):
            try:
                data = self.qs[i].get_nowait()

                # # map thermal image to heat coloration
                # normalized = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
                # data_uint8 = np.uint8(normalized)
                # thermal_colored = cv2.applyColorMap(data_uint8, cv2.COLORMAP_JET)
                # thermal_colored = cv2.cvtColor(thermal_colored, cv2.COLOR_BGR2RGB)
                # thermal_color = thermal_colored

                # # publish image
                # header = Header()
                # header.stamp = timestamp
                # header.frame_id = f'thermal_camera_{i}_optical_frame'
                # thermal_msg = self.bridge.cv2_to_imgmsg(thermal_color, encoding='rgb8')
                # thermal_msg.header = header
                # self.thermal_publishers[i].publish(thermal_msg)

                # publish raw thermal image
                header = Header()
                header.stamp = timestamp
                header.frame_id = f'thermal_camera_{i}_optical_frame'
                thermal_msg = self.bridge.cv2_to_imgmsg(data, encoding='mono16')
                thermal_msg.header = header
                self.thermal_publishers[i].publish(thermal_msg)

                # # transform 2D extrema to 3D points
                # minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)

                # # publish extrema
                # extrema = Extrema()
                # header = Header()
                # header.stamp = timestamp
                # extrema.coldest_point
                # extrema.coldest_value = minVal
                # extrema.hottest_point 
                # extrema.hottest_value = maxVal
                
            except Empty:
                pass
        cv2.waitKey(1)

    def cleanup(self):
        """Graceful shutdown of UVC devices."""
        if hasattr(self, 'camera_initialized') and self.camera_initialized:
            try:
                for i in range(self.camera_count):
                    if hasattr(self, 'devh') and self.devh[i]:
                        libuvc.uvc_stop_streaming(self.devh[i])
                        libuvc.uvc_unref_device(self.devs[i])
                
                if hasattr(self, 'ctx'):
                    libuvc.uvc_exit(self.ctx)
                    
                self.get_logger().info('Camera cleanup completed')
            except Exception as e:
                self.get_logger().error(f'Cleanup error: {e}')

def main():
    rclpy.init()
    node = ThermalCameraPublisher()

    def signal_handler(sig, frame):
        node.cleanup()
        rclpy.shutdown()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        pass
    finally:
        node.cleanup()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

from thermal_camera.uvctypes import *
import time
import cv2
import numpy as np
try:
  from queue import Queue
except ImportError:
  from queue import Queue

import signal
import rclpy
import rclpy.node
from rcl_interfaces.msg import ParameterDescriptor, ParameterType

def make_callback(index, queue):
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

def display_temperature(img, val_k, loc, color):
  val = ktof(val_k)
  cv2.putText(img,"{0:.1f} degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
  x, y = loc
  cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
  cv2.line(img, (x, y - 2), (x, y + 2), color, 1)

def click_event(event, x, y, flags, param):
    print("click event")
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

class LeptonVisualizerNode(rclpy.node.Node):
    def __init__(self):
      super().__init__("uvc_visualize")

      # Declare and read parameters
      self.declare_parameter(
        name="camera_count",
        value=1,
        descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            description="Number of cameras in the cell.",
        ),
      )
      self.declare_parameter(
        name="buffer_size",
        value=2,
        descriptor=ParameterDescriptor(
            type=ParameterType.PARAMETER_DOUBLE,
            description="Buffer size for each camera.",
        ),
      )
      self.camera_count = (
          self.get_parameter("camera_count").get_parameter_value().integer_value
      )
      self.get_logger().info(f"Number of cameras: {self.camera_count}")
      self.buffer_size = (
          self.get_parameter("buffer_size").get_parameter_value().integer_value
      )
      self.get_logger().info(f"Buffer size: {self.buffer_size}")

      # initialize camera queues
      self.qs = [Queue(self.buffer_size) for _ in range(self.camera_count)]

      # configure visualization windows
      self.min_pixel_val, self.max_pixel_val = None, None
      self.cur_img = None
      for i in range(self.camera_count):
        cv2.namedWindow(f'Lepton Radiometry {i}')
        cv2.setMouseCallback(f'Lepton Radiometry {i}', self.get_limit_pixels, self)
        cv2.waitKey(1) 

      self.grayscale_buf = 10

      self.show_thermal_feed()

    def get_limit_pixels(self, event, x, y, flags, param):
      if event == cv2.EVENT_LBUTTONDOWN:
          if param.cur_img is None:
            self.get_logger().warn("Current image is None.")
          elif self.max_pixel_val is None:
            self.max_pixel_val = min(255, param.cur_img[y, x][0] + self.grayscale_buf)
            self.get_logger().info(f"Max pixel value set to {self.max_pixel_val}")
          elif self.min_pixel_val is None:
            self.min_pixel_val = max(0, param.cur_img[y, x][0] - self.grayscale_buf)
            self.get_logger().info(f"Min pixel value set to {self.min_pixel_val}")
          else:
            self.min_pixel_val = None
            self.max_pixel_val = None
            self.get_logger().warn("Reset max and min pixel values.")


    def show_thermal_feed(self):
      """Main visualizer method, sends the camera feed to
      its respective windows."""
      ctx = POINTER(uvc_context)()
      dev = POINTER(uvc_device)()
      devs = POINTER(POINTER(uvc_device))()
      devh = (POINTER(uvc_device_handle) * self.camera_count)()
      ctrls = [uvc_stream_ctrl() for _ in range(self.camera_count)]
      res = libuvc.uvc_init(byref(ctx), 0)
      callbacks = [make_callback(i, self.qs[i]) for i in range(self.camera_count)]

      if res < 0:
        print("uvc_init error")
        exit(1)

      try:
        # find the desired amount of cameras
        res = libuvc.uvc_find_devices(ctx, byref(devs), PT_USB_VID, PT_USB_PID, 0)
        print(res)
        i = 0
        while devs[i]:
            print(devs.contents[i])
            i += 1
        print('Number of devices: {}'.format(i))
        if res < 0:
          print("uvc_find_device error")
          print(res)
          exit(1)

        try:
          # open each camera
          for i in range(self.camera_count):
            res = libuvc.uvc_open(devs[i], byref(devh[i]))
            if res < 0:
              print(f"uvc_open {i} error")
              exit(1)
            print_device_info(devh[i])
            print_device_formats(devh[i])

          # configure stream format
          for i in range(self.camera_count):
            frame_format = uvc_get_frame_formats_by_guid(devh[i], VS_FMT_GUID_Y16)
            if len(frame_format) == 0:
              print(f"device {i} does not support Y16")
              exit(1)
            libuvc.uvc_get_stream_ctrl_format_size(devh[i], byref(ctrls[i]), UVC_FRAME_FORMAT_Y16,
              frame_format[0].wWidth, frame_format[0].wHeight, int(1e7 / frame_format[0].dwDefaultFrameInterval)
            )

          # start streaming for each camera
          for i in range(self.camera_count):
            res = libuvc.uvc_start_streaming(devh[i], byref(ctrls[i]), callbacks[i], None, 0)
            if res < 0:
              print("uvc_start_streaming {0} failed: {0}".format(i, res))
              exit(1)
            time.sleep(0.5)

          # configure graceful exit signal handler
          stop_flag = False
          def signal_handler(sig, frame):
              nonlocal stop_flag
              print("\nSIGINT caught. Cleaning up...")
              stop_flag = True
          signal.signal(signal.SIGINT, signal_handler)

          try:
            while not stop_flag:
              # get camera captures stored in queues
              datas = []
              for i in range(self.camera_count):
                try:
                  data = self.qs[i].get(timeout=2)
                except:
                  print(f"[Main Loop] Timed out waiting for camera {i}")
                  data = None
                datas.append(data)      
              if any(datas[i] is None for i in range(self.camera_count)):
                break

              # upscale and display camera feeds
              for i in range(self.camera_count):
                data = cv2.resize(datas[i][:,:], (640, 480))
                minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)
                img = raw_to_8bit(data)
                if self.min_pixel_val is not None and self.max_pixel_val is not None:
                  # clamp to selected pixel values if applicable
                  # self.get_logger().info(f"pre: max data {np.max(img)}, min data {np.min(img)}")
                  img = np.clip(img, self.min_pixel_val, self.max_pixel_val)
                  # self.get_logger().info(f"post: max data {np.max(img)}, min data {np.min(img)}")
                  # img_min = img.min()
                  # img_max = img.max()

                  # # Avoid division by zero if image is flat
                  # if img_max != img_min:
                  #     img = (img - img_min) / (img_max - img_min) * 255
                  # else:
                  #     img = np.zeros_like(img, dtype=np.uint8)
                display_temperature(img, minVal, minLoc, (255, 0, 0))
                display_temperature(img, maxVal, maxLoc, (0, 0, 255))
                self.cur_img = img.copy()
                # cv2.imshow(f'Lepton Radiometry {i}', img)
                cv2.imshow(f'Lepton Radiometry {i}', self.cur_img)
              cv2.waitKey(1)

            # cleanup
            cv2.destroyAllWindows()
          finally:
            for i in range(self.camera_count):
              if devh[i]:
                libuvc.uvc_stop_streaming(devh[i])
        finally:
          for i in range(self.camera_count):
            if devh[i]:
              libuvc.uvc_unref_device(devs[i])
      finally:
        libuvc.uvc_exit(ctx)

import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class LeptonSubscriberNode(rclpy.node.Node):
  def __init__(self):
    super().__init__("uvc_visualize")

    # Declare and read parameters
    self.declare_parameter(
      name="camera_count",
      value=1,
      descriptor=ParameterDescriptor(
          type=ParameterType.PARAMETER_DOUBLE,
          description="Number of cameras in the cell.",
      ),
    )
    self.camera_count = (
        self.get_parameter("camera_count").get_parameter_value().integer_value
    )
    self.get_logger().info(f"Number of cameras: {self.camera_count}")

    self.bridge = CvBridge()

    # configure visualization windows
    for i in range(self.camera_count):
      cv2.namedWindow(f'Lepton Radiometry {i}')
      cv2.waitKey(1)

    # Subscribers for each camera's image
    self.image_subs = [
        message_filters.Subscriber(self, Image, f'/thermal_camera_{i}/image_raw')
        for i in range(self.camera_count)
    ]

    self.image_raw_grid_pub = self.create_publisher(Image, "/image_raw_grid", 1)
    self.bridge = CvBridge()

    # Synchronizer
    self.ts = message_filters.ApproximateTimeSynchronizer(
        self.image_subs,
        queue_size=10,
        slop=0.1
    )
    self.ts.registerCallback(self.imshow_callback)

  def imshow_callback(self, *msgs):
    """ Separate windows for each image """

    # for i in range(self.camera_count):
    #   data = self.bridge.imgmsg_to_cv2(msgs[i], desired_encoding='mono16')

    #   normalized = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
    #   data_uint8 = np.uint8(normalized)
    #   thermal_colored = cv2.applyColorMap(data_uint8, cv2.COLORMAP_JET)
    #   # thermal_colored = cv2.cvtColor(thermal_colored, cv2.COLOR_BGR2RGB)
    #   thermal_color = thermal_colored

    #   img = cv2.resize(thermal_color[:,:], (640, 480))
    #   cv2.imshow(f'Lepton Radiometry {i}', img)
    #   cv2.waitKey(1)

    """ All images in 1 display: """
    """ 0 1"""
    """ 2 3"""

    resized_images = []

    for i in range(self.camera_count):
        data = self.bridge.imgmsg_to_cv2(msgs[i], desired_encoding='mono16')

        # Normalize to 8-bit and apply colormap
        normalized = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX)
        data_uint8 = np.uint8(normalized)
        thermal_colored = cv2.applyColorMap(data_uint8, cv2.COLORMAP_JET)

        # Resize and store
        img = cv2.resize(thermal_colored, (640, 480))
        resized_images.append(img)

    # Make 2x2 grid (assuming camera_count == 4)
    if len(resized_images) == 4:
        top_row = cv2.hconcat([resized_images[0], resized_images[1]])
        bottom_row = cv2.hconcat([resized_images[2], resized_images[3]])
        grid = cv2.vconcat([top_row, bottom_row])

        ros_img = self.bridge.cv2_to_imgmsg(grid, encoding="bgr8")
        ros_img.header.stamp = self.get_clock().now().to_msg()
        self.image_raw_grid_pub.publish(ros_img)

        cv2.imshow("Lepton Radiometry Grid", grid)
        cv2.waitKey(1)
    else:
        self.get_logger().warn(f"Expected 4 images, got {len(resized_images)}")

def main():
  rclpy.init()
  # node = LeptonVisualizerNode()
  node = LeptonSubscriberNode()
  rclpy.spin(node)

  node.destroy_node()
  rclpy.shutdown()

if __name__ == '__main__':
  main()


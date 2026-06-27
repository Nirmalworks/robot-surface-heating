#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from builtin_interfaces.msg import Time
import rclpy.time
from sensor_msgs.msg import PointCloud2  
from thermal_camera_interfaces.msg import RobotMask
from sensor_msgs.msg import Image
from collections import deque
import numpy as np
import time


class TimestampComparator(Node):
    def __init__(self):
        super().__init__('timestamp_comparator')

        self.latest_image_msg = None
        
        self.msg_type = PointCloud2
        # self.msg_type = RobotMask
        # self.msg_type = Image
        
        self.topic_name = '/raw_thermal_pointcloud'
        # self.topic_name = '/selected_surface_points'
        # self.topic_name = '/colored_cad_pointcloud'
        # self.topic_name = '/robot_mask'
        # self.topic_name = '/thermal_camera_0/image_raw'

        self.subscription = self.create_subscription(
            self.msg_type,
            self.topic_name,
            # self.callback,
            self.callback_time,
            10
        )

        # self.camera_subscriber = self.create_subscription(
        #     Image,
        #     '/thermal_camera_0/image_raw',
        #     self.camera_callback,
        #     1
        # )

        # self.pointcloud_subscriber = self.create_subscription(
        #     PointCloud2,
        #     '/colored_cad_pointcloud',
        #     self.pointcloud_callback,
        #     1
        # )

        self.all_deltas = [] # Stores all deltas
        # self.last_n_deltas = deque(maxlen=10) # Rolling buffer for last 10
        self.min_delta = 100.0
        self.max_delta = 0.0

    def callback(self, msg: PointCloud2):
    # def callback(self, msg: RobotMask):
        # ROS Timestamp
        msg_time = msg.header.stamp
        msg_stamp_sec = msg_time.sec + msg_time.nanosec * 1e-9

        # Wall clock time (seconds since epoch)

        wall_time_sec = time.time()

        # Store delta data
        delta = wall_time_sec - msg_stamp_sec
        self.all_deltas.append(delta)
        # self.last_n_deltas.append(delta)
        if(delta < self.min_delta):
            self.min_delta = delta
        if(delta > self.max_delta):
            self.max_delta = delta

        # Compute stats
        avg_total = np.mean(self.all_deltas)
        std_total = np.std(self.all_deltas)
        # avg_last_10 = np.mean(self.last_n_deltas)

        # self.get_logger().info(f"msg time: {msg_stamp_sec:.6f} s | Wall time: {wall_time_sec:.6f} s | Delta: {delta:.6f} s")
        self.get_logger().info(f"Topic: {self.topic_name}"
                               f"\n\tDelta: {delta:.4f} s | Total Avg: {avg_total:.4f} s | Min: {self.min_delta:.4f} s | Max: {self.max_delta:.4f} s | Std dev: {std_total:.4f} s")
        
    def callback_time(self, msg: PointCloud2):
    # def callback(self, msg: RobotMask):
        # ROS Timestamp
        msg_time = rclpy.time.Time.from_msg(msg.header.stamp)

        # Wall clock time (seconds since epoch)
        wall_time = self.get_clock().now()

        # Store delta data
        delta = wall_time - msg_time

        print(delta.nanoseconds*1e-9)
        # print(delta)

        # self.get_logger().info(f"Delta: {delta_sec:.6f} seconds")e

        # self.all_deltas.append(delta)
        # # self.last_n_deltas.append(delta)
        # if(delta < self.min_delta):
        #     self.min_delta = delta
        # if(delta > self.max_delta):
        #     self.max_delta = delta

        # # Compute stats
        # avg_total = np.mean(self.all_deltas)
        # std_total = np.std(self.all_deltas)
        # # avg_last_10 = np.mean(self.last_n_deltas)

        # # self.get_logger().info(f"msg time: {msg_stamp_sec:.6f} s | Wall time: {wall_time_sec:.6f} s | Delta: {delta:.6f} s")
        # self.get_logger().info(f"Topic: {self.topic_name}"
        #                        f"\n\tDelta: {delta:.4f} s | Total Avg: {avg_total:.4f} s | Min: {self.min_delta:.4f} s | Max: {self.max_delta:.4f} s | Std dev: {std_total:.4f} s")

    def camera_callback(self, msg: Image):
        self.latest_image_msg = msg
        self.get_logger().info(f"Camera : {msg.header.stamp}")

    def pointcloud_callback(self, msg: PointCloud2):
        # store image data
        self.get_logger().info(f"PCL : {msg.header.stamp}")
        # if self.latest_image_msg is None:
        #     return
        # camera_msg_time = self.latest_image_msg.header.stamp
        # camera_msg_stamp_sec = camera_msg_time.sec + camera_msg_time.nanosec * 1e-9

        # # store pointcloud data
        # pointcloud_msg_time = msg.header.stamp
        # pointcloud_msg_stamp_sec = pointcloud_msg_time.sec + pointcloud_msg_time.nanosec * 1e-9

        # # Store delta data
        # delta = camera_msg_stamp_sec - pointcloud_msg_stamp_sec
        # self.all_deltas.append(delta)
        # # self.last_n_deltas.append(delta)
        # if(delta < self.min_delta):
        #     self.min_delta = delta
        # if(delta > self.max_delta):
        #     self.max_delta = delta

        # # Compute stats
        # avg_total = np.mean(self.all_deltas)
        # std_total = np.std(self.all_deltas)
        # # avg_last_10 = np.mean(self.last_n_deltas)

        # # self.get_logger().info(f"msg time: {msg_stamp_sec:.6f} s | Wall time: {wall_time_sec:.6f} s | Delta: {delta:.6f} s")
        # self.get_logger().info(f"Topic: {self.topic_name}"
        #                        f"\n\tDelta: {delta:.4f} s | Total Avg: {avg_total:.4f} s | Min: {self.min_delta:.4f} s | Max: {self.max_delta:.4f} s | Std dev: {std_total:.4f} s")


def main(args=None):
    rclpy.init(args=args)
    node = TimestampComparator()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Float64
import numpy as np
from scipy.spatial import cKDTree
import time


def ktoc(val):
  return (val - 27315) / 100.0

def ktof(val):
  return (1.8 * ktoc(val) + 32.0)


class HottestTempNode(Node):
    
    def __init__(self):
        super().__init__('hottest_temp_node')

        # self.pointcloud_topic = 'raw_thermal_pointcloud'
        self.pointcloud_topic = 'selected_surface_points'

        self.hottest_temp_pub = self.create_publisher(
            Float64,
            "hottest_temp",
            10,
        )

        self.pointcloud_sub = self.create_subscription(
            PointCloud2,
            self.pointcloud_topic,
            self.pointcloud_callback,
            10,
        )

        self.get_logger().info(f"Monitoring hottest temp for topic: {self.pointcloud_topic}")


    def pointcloud_callback(self, msg: PointCloud2):
        
        points = []
        temperatures = []
        
        for p in point_cloud2.read_points(msg, field_names=("x", "y", "z", "thermal"), skip_nans=True):
            x, y, z, temp = p
            points.append([x, y, z])
            temperatures.append(ktoc(temp))        

        if not points:
            # self.get_logger().warn("Empty raw thermal point cloud.")
            self.get_logger().warn(self,"Empty raw thermal point cloud.")
            return

        points = np.array(points)
        temperatures = np.array(temperatures)            

        # Temperature smoothing
        tree = cKDTree(points)
        smoothed_temp = np.zeros(len(points))
        for i, pt in enumerate(points):
            idxs = tree.query_ball_point(pt, r=0.01)
            smoothed_temp[i] = np.mean(temperatures[idxs]) if idxs else temperatures[i]

        # Find hottest point
        hottest_idx = np.nanargmax(smoothed_temp)
        hottest_temp = Float64(data=temperatures[hottest_idx])
        
        # hottest_temp.data = 100.0
        # self.get_logger().info("Publishing 100.0")

        # # Publish hottest temp
        self.hottest_temp_pub.publish(hottest_temp)

        # time.sleep(5.0)

        # hottest_temp.data = 0.0
        # self.get_logger().info("Publishing 0.0")
        # self.hottest_temp_pub.publish(hottest_temp)

        # while 1:
        #     pass



def main(args=None):
    rclpy.init(args=args)
    node = HottestTempNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

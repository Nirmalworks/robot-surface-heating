#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

class ThermalGridVisualizer(Node):
    def __init__(self):
        super().__init__('thermal_grid_visualizer')
        self.bridge = CvBridge()
        self.subscription = self.create_subscription(
            Image,
            "/thermal_masked_grid",
            self.callback,
            10
        )
        self.subscription  # prevent unused warning

        self.get_logger().info("Displaying masked thermal images...")

    def callback(self, msg: Image):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            cv2.imshow("Masked Thermal Grid", img)
            cv2.waitKey(1)
        except Exception as e:
            self.get_logger().warn(f"Error displaying image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = ThermalGridVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
import sys

from robot_interface.srv import PoseToPlan
import rclpy
from rclpy.node import Node


class MinimalClientAsync(Node):

    def __init__(self):
        super().__init__('minimal_client_async')
        self.cli = self.create_client(PoseToPlan, '/ur5_move_to_pose')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = PoseToPlan.Request()

    def send_request(self):
        self.req.x = -0.6
        self.req.y = 0.0
        self.req.z = 0.5
        self.req.qx = 0.0
        self.req.qy = -0.707
        self.req.qz = 0.0
        self.req.qw = 0.707
        self.future = self.cli.call_async(self.req)


def main(args=None):
    rclpy.init(args=args)

    minimal_client = MinimalClientAsync()
    response = minimal_client.send_request()

    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
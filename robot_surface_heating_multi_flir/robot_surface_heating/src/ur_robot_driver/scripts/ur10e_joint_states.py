#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class MySubscriber(Node):

    def __init__(self):
        super().__init__('ur10e_joint_states')
        self.subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.listener_callback,
            10)
        self.publisher = self.create_publisher(JointState, '/arm_1/joint_states', 100)

    def listener_callback(self, msg):
        joint_positions = msg.position
        msg_formatted = JointState()
        msg_formatted.name = msg.name
        msg_formatted.header = msg.header
        msg_formatted.position = msg.position
        msg_formatted.velocity = msg.velocity
        msg_formatted.effort = msg.effort

        do_publish = True
        for name in msg_formatted.name:
            if('arm_0' in name):
                do_publish = False

        if(do_publish == True):
            self.publisher.publish(msg_formatted)

def main(args=None):
    rclpy.init(args=args)

    my_subscriber = MySubscriber()

    rclpy.spin(my_subscriber)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    my_subscriber.destroy_node()
    rclpy.shutdown()
if __name__ == '__main__':
    main()
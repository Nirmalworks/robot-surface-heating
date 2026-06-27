#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

class FakeJointStatePublisher(Node):
    def __init__(self):
        super().__init__('fake_joint_state_publisher')

        # initial joint state
        self.joint_state = JointState()
        self.joint_state.name = [
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"
        ]
        self.joint_state.position = [
            0.0, 0.0, 0.0,
            0.0, 0.0, 0.0
        ]

        # joint state publisher
        self.arm_publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.timer = self.create_timer(0.1, self.publish_joint_states)  # Publish every 0.1 seconds

        # joint state subscriber
        self.js_subscription = self.create_subscription(
            JointState,
            '/fake_joint_states',
            self.update_joint_state_callback,
            1,
        )
        self.js_subscription  # prevent unused variable warning

    def update_joint_state_callback(self, msg: JointState):
        self.joint_state = msg

    def publish_joint_states(self):
        msg = self.joint_state
        self.arm_publisher.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    fake_joint_state_publisher = FakeJointStatePublisher()
    rclpy.spin(fake_joint_state_publisher)
    fake_joint_state_publisher.destroy_node()
    rclpy.shutdown()
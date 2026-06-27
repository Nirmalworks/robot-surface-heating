#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
import time

from control_msgs.action import GripperCommand
from control_msgs.msg import GripperCommand as Gc_Msg


class GripperTestNode(Node):

    def __init__(self):
        super().__init__('gripper_test_node')
        
        self.declare_parameters(
            namespace='',
            parameters=[
                ("ur5_params.gripper_service", "/gripper_actuate"),
                ("ur10e_params.gripper_service", "/gripper_actuate"),
                ("ur5_params.gripper_grab_pos", 0.0),
                ("ur5_params.gripper_open_pos", 0.0),
                ("ur10e_params.gripper_grab_pos", 0.0),
                ("ur10e_params.gripper_open_pos", 0.0),
                ("ur5_params.gripper_default_effort", 0.0),
                ("ur10e_params.gripper_default_effort", 0.0),
            ]
        )

        # gripper constants
        self.GRIPPER_0_GRAB_POSITION = self.get_parameter("ur5_params.gripper_grab_pos").value
        self.GRIPPER_0_OPEN_POSITION = self.get_parameter("ur5_params.gripper_open_pos").value
        self.GRIPPER_0_DEFAULT_EFFORT = self.get_parameter("ur5_params.gripper_default_effort").value
        self.get_logger().info(f"UR5 parameters: grab={self.GRIPPER_0_GRAB_POSITION}, open={self.GRIPPER_0_OPEN_POSITION}, effort={self.GRIPPER_0_DEFAULT_EFFORT}")
        self.GRIPPER_1_GRAB_POSITION = self.get_parameter("ur10e_params.gripper_grab_pos").value
        self.GRIPPER_1_OPEN_POSITION = self.get_parameter("ur10e_params.gripper_open_pos").value
        self.GRIPPER_1_DEFAULT_EFFORT = self.get_parameter("ur10e_params.gripper_default_effort").value
        self.get_logger().info(f"UR10e parameters: grab={self.GRIPPER_1_GRAB_POSITION}, open={self.GRIPPER_1_OPEN_POSITION}, effort={self.GRIPPER_1_DEFAULT_EFFORT}")

        # gripper action client and storage members
        self.gripper0_srv_name = self.get_parameter("ur5_params.gripper_service").value
        self.gripper0_action_client_ = ActionClient(self, GripperCommand, self.gripper0_srv_name)
        self.gripper1_srv_name = self.get_parameter("ur10e_params.gripper_service").value
        self.gripper1_action_client_ = ActionClient(self, GripperCommand, self.gripper1_srv_name)
        self.gripper_result_position = None
        self.gripper_result_effort = None
        self.gripper_result_stalled = None
        self.gripper_result_reached_goal = None

        # loop gripper actuation
        while True:
            if input("open UR5 gripper (y/n)?  ").lower() == 'y':
                self.actuate_gripper_0(self.GRIPPER_0_OPEN_POSITION, self.GRIPPER_0_DEFAULT_EFFORT)
                time.sleep(3)
            if input("open UR10e gripper (y/n)?  ").lower() == 'y':
                self.actuate_gripper_1(self.GRIPPER_1_OPEN_POSITION, self.GRIPPER_1_DEFAULT_EFFORT)
                time.sleep(3)
            if input("close UR5 gripper (y/n)?  ").lower() == 'y':
                self.actuate_gripper_0(self.GRIPPER_0_GRAB_POSITION, self.GRIPPER_0_DEFAULT_EFFORT)
                time.sleep(3)
            if input("close UR10e gripper (y/n)?  ").lower() == 'y':
                self.actuate_gripper_1(self.GRIPPER_1_GRAB_POSITION, self.GRIPPER_1_DEFAULT_EFFORT)
                time.sleep(3)
    
    def actuate_gripper_0(self, position=0.0, max_effort=0.0):
        """Makes an action request to the associated gripper controller server to change the gripper's position"""
        gripper_goal = GripperCommand.Goal()
        gripper_goal.command = Gc_Msg()
        gripper_goal.command.position = position
        gripper_goal.command.max_effort = max_effort

        self.get_logger().info(f"ACTUATING GRIPPER: position={gripper_goal.command.position}, max_effort={gripper_goal.command.max_effort}")
        self.gripper0_action_client_.wait_for_server()
        self.goal_future = self.gripper0_action_client_.send_goal_async(gripper_goal)
        self.goal_future.add_done_callback(self.gripper_response_callback)

    def actuate_gripper_1(self, position=0.0, max_effort=0.0):
        """Makes an action request to the associated gripper controller server to change the gripper's position"""
        gripper_goal = GripperCommand.Goal()
        gripper_goal.command = Gc_Msg()
        gripper_goal.command.position = position
        gripper_goal.command.max_effort = max_effort

        self.get_logger().info(f"ACTUATING GRIPPER: position={gripper_goal.command.position}, max_effort={gripper_goal.command.max_effort}")
        self.gripper1_action_client_.wait_for_server()
        self.goal_future = self.gripper1_action_client_.send_goal_async(gripper_goal)
        self.goal_future.add_done_callback(self.gripper_response_callback)

    def gripper_response_callback(self, future):
        """Action response handler for actuate_gripper"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('GRIPPER GOAL REJECTED')
            return

        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.gripper_result_callback)
    
    def gripper_result_callback(self, future):
        """Action result handler for actuate_gripper"""
        result = future.result().result
        
        if result.stalled:
            self.get_logger().info(f"GRIPPER STOPPED DURING MOVEMENT: position={result.position}, effort={result.effort}")
        elif result.reached_goal:
            self.get_logger().info(f"GRIPPER COMPLETED MOVEMENT: position={result.position}, effort={result.effort}")
        else:
            self.get_logger().error(f"GRIPPER IN UNKNOWN STATE: position={result.position}, effort={result.effort}")
        
        # publish gripper result data -> maybe TODO
        self.gripper_result_position = result.position
        self.gripper_result_effort = result.effort
        self.gripper_result_stalled = result.stalled
        self.gripper_result_reached_goal = result.reached_goal

def main(args=None):
    
    rclpy.init(args=args)

    node = GripperTestNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
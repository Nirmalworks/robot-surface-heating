#!/usr/bin/env python
import rospy
import yaml
import numpy as np
from std_msgs.msg import String, Float32MultiArray, Int32MultiArray

class orchestrator:
    def __init__(self):
        self.heated_state_pub = rospy.Publisher('heated_state', Int32MultiArray, queue_size=10)
        self.heat_command_to_control_pub = rospy.Publisher('heat_command_to_control', Int32MultiArray, queue_size=10)
        self.heat_command_sub = rospy.Subscriber('heat_command', Int32MultiArray, self.heated_command_callback)
        self.heated_state_curr = Int32MultiArray()
        self.heated_state_curr.data = [-1, -1]
        self.freq = 10

    def heated_command_callback(self, data):
        self.heated_state_curr.data = data.data

    def main_loop(self):
        rate = rospy.Rate(self.freq)
        while not rospy.is_shutdown():
            self.heated_state_pub.publish(self.heated_state_curr)
            self.heat_command_to_control_pub.publish(self.heated_state_curr)
            rate.sleep()

if __name__ == '__main__': 
    rospy.init_node('orchestrator')
    orchestrator_obj = orchestrator()
    orchestrator_obj.main_loop()
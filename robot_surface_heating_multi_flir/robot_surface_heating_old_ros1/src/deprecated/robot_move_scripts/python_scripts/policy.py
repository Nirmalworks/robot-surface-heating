#!/usr/bin/env python
import rospy
import yaml
import numpy as np
from std_msgs.msg import String, Float32MultiArray, Int32MultiArray

class greedy:
    def __init__(self):
        self.heat_command_pub = rospy.Publisher('heat_command', Int32MultiArray, queue_size=10)
        self.temp_array_sub = rospy.Subscriber('temp_array_processed', Float32MultiArray, self.temp_array_callback)
        file = open('/home/cam/ND_ws/robot_surface_heating_ws/src/param.yaml', 'r')
        self.parameters = yaml.safe_load(file)
        self.node_num_x = self.parameters['node_num_x']
        self.node_num_y = self.parameters['node_num_y']
        self.switch_temp = self.parameters['switch_temp']
        self.temp_data = np.array([])
        self.policy_type = self.parameters['policy_type']
        if(self.policy_type=='greedy'):
            self.greedy_sub = rospy.Subscriber('heated_state', Int32MultiArray, self.greedy_callback)
        elif(self.policy_type=='zigzag'):
            self.zigzag_sub = rospy.Subscriber('heated_state', Int32MultiArray, self.zigzag_callback)
    
    def greedy_callback(self, data):
        if(len(self.temp_data)>0):
            # DON'T CHANGE THE PART BELOW
            reshaped_data = self.temp_data.reshape(self.node_num_x+2, self.node_num_y+2)
            temp_data_formatted = reshaped_data[1:-1, 1:-1]
            # DON'T CHANGE THE PART ABOVE

            min_index = np.argmin(temp_data_formatted)
            min_index_2d = np.unravel_index(min_index, temp_data_formatted.shape)

            msg = Int32MultiArray()
            
            if(data.data[0] == -1 and data.data[1] == -1):
                msg.data = [min_index_2d[0], min_index_2d[1]]
                self.heat_command_pub.publish(msg)
            elif(temp_data_formatted[data.data[0], data.data[1]] > self.switch_temp):
                msg.data = [min_index_2d[0], min_index_2d[1]]
                self.heat_command_pub.publish(msg)
            elif(temp_data_formatted[data.data[0], data.data[1]] < self.switch_temp):
                msg.data = [data.data[0], data.data[1]]
                self.heat_command_pub.publish(msg)
            print(msg)

    def zigzag_callback(self, data):
        msg = Int32MultiArray()
        msg.data = [-10, -10]
        self.heat_command_pub.publish(msg)
    
    def temp_array_callback(self, data):
        self.temp_data = np.array(data.data)

    def main_loop(self):
        rospy.spin()

if __name__ == '__main__':
    rospy.init_node('greedy_policy')
    greedy_obj = greedy()
    rospy.spin()
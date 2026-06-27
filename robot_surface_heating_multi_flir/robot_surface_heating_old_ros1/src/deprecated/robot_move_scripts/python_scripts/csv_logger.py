#!/usr/bin/python
import sys
import copy
import rospy
import moveit_commander
import yaml
from std_msgs.msg import Float32, Int32MultiArray, Float32MultiArray
import csv
import time
import os

class logger():
    
    def __init__(self):
        rospy.init_node("csv_logger_node", anonymous=True)
        file = open('/home/cam/ND_ws/robot_surface_heating_ws/src/param.yaml', 'r')
        self.parameters = yaml.safe_load(file)
        self.policy_type = self.parameters['policy_type']
        self.data_file_path ="/home/cam/ND_ws/robot_surface_heating_ws/src/data"
        self.dir_list = os.listdir(self.data_file_path)

        number = 0
        for file_name in self.dir_list:
            if(str(self.policy_type) in file_name):
                if(int(file_name[:-4][6:]) > number):
                    number = int(file_name[:-4][6:])

        self.csv_filename = f"{self.data_file_path}/{self.policy_type}"+str(number+1)+".csv"
        self.temp_array = None
        self.indices_to_heat = None
        self.temp_array_sub = rospy.Subscriber("temp_array_processed", Float32MultiArray, self.temp_sub_callback)
        self.controller_state_sub = rospy.Subscriber("controller_state", Int32MultiArray, self.controller_state_callback)
        
        self.file = open(self.csv_filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        self.node_num_x = self.parameters['node_num_x']
        self.node_num_y = self.parameters['node_num_y']
        self.writer.writerow(["Time_Seconds"]+[f"Node_{i+1}" for i in range( (self.node_num_x+2) * (self.node_num_y+2) )]+["Node_Index_x"]+["Node_Index_y"])
        self.start_time = time.time()


    def controller_state_callback(self,data):
        current_time = time.time() - self.start_time  # Calculate elapsed time in seconds
        print(data.data)
        if(data.data[0] != -1 and data.data[1] != -1):
            row_data = [current_time]+list(self.temp_array)+[data.data[0]+1, data.data[1]+1]
            self.writer.writerow(row_data)
        else:
            row_data = [current_time]+list(self.temp_array)+[-1, -1]
            self.writer.writerow(row_data)

    def temp_sub_callback(self,data):
        
        self.temp_array = data.data

    def indices_sub_callback(self,data):
        self.indices_to_heat = data.data


log = logger()
rospy.spin()
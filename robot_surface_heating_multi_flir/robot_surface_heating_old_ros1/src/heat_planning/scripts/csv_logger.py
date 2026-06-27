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
        rospy.init_node("csv_logger", anonymous=True)
        file = open('/home/cam/ND_ws/heat_ws/robot_surface_heating/src/param.yaml', 'r')
        self.parameters = yaml.safe_load(file)
        self.policy_type = self.parameters['policy_type']
        self.data_file_path ="/home/cam/ND_ws/heat_ws/robot_surface_heating/src/data"
        self.dir_list = os.listdir(self.data_file_path)
        if self.policy_type == "bnb":
            self.policy_type = "branchnb"
        number = 0
        for file_name in self.dir_list:
            if(str(self.policy_type) in file_name):
                if(self.policy_type == "one_step"):
                    if(int(file_name[:-4][8:])>number):
                        number = int(file_name[:-4][8:])
                elif(self.policy_type == "heat_cen"):
                    print(file_name)
                    if(int(file_name[:-4][8:])>number):
                        print(file_name[:-4][8:])
                        number = int(file_name[:-4][8:])
                elif(self.policy_type == "branchnb"):
                    print(file_name)
                    if(int(file_name[:-4][8:])>number):
                        print(file_name[:-4][3:])
                        number = int(file_name[:-4][8:])
                elif(int(file_name[:-4][6:]) > number):
                    number = int(file_name[:-4][6:])

        print(number)

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
        if(len(data.data)>=2 and len(list(self.temp_array)) > 0):
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
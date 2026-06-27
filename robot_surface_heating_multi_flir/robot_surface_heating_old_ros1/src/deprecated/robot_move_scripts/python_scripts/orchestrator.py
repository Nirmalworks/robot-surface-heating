#!/usr/bin/python

import rospy
from std_msgs.msg import Float32MultiArray, Int32MultiArray
import yaml
import numpy as np
from uvctypes import *
import time
import random
import cv2
import numpy as np
try:
  from queue import Queue
except ImportError:
  from queue import Queue
import platform
from datetime import datetime
import csv
import rospy
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray, Int32MultiArray
import yaml

global pub1, pub2, pub3
global x_values, y_values
global x_heated, y_heated, heated_node_temp

def callback(data):
    global x_values, y_values  # declare as global variables
    global x_heated, y_heated
    x_pos = x_values[data.data[0]]
    y_pos = y_values[data.data[1]]
    msg = Float32MultiArray() 

    if(x_heated == -1 and y_heated == -1):      
        print(x_heated)      
        msg.data = [x_pos, y_pos, 0.5]

        pub1.publish(msg)
        x_heated = data.data[0]
        y_heated = data.data[1]

        msg.data = [x_pos, y_pos]
        pub1.publish(msg)

    elif(x_heated != data.data[0] or y_heated != data.data[1]):
        x_pos_curr = x_values[x_heated]
        y_pos_curr = y_values[y_heated]
            
        msg.data = [x_pos_curr, y_pos_curr, 0.5]

        pub1.publish(msg)
        x_heated = data.data[0]
        y_heated = data.data[1]

        msg.data = [x_pos, y_pos]
        pub1.publish(msg)

def ktof(val):
  return (1.8 * ktoc(val) + 32.0)

def ktoc(val):
  return (val - 27315) / 100.0

def discretize(data):
    file = open('/home/cam/ND_ws/robot_surface_heating_ws/src/param.yaml', 'r')
    parameters = yaml.safe_load(file)

    node_num_x = parameters['node_num_x']
    node_num_y = parameters['node_num_y']

    data_numpy = np.array(data)

    reshaped_data = data_numpy.reshape(node_num_x+2, node_num_y+2)
    heated_nodes_data = reshaped_data[1:-1, 1:-1]

    return heated_nodes_data

def sensor_callback(data):
    # message = Float32MultiArray()
    # message.data = discretize(data.data)
    # pub2.publish(message)

    message1 = Int32MultiArray()
    message1.data = [x_heated, y_heated]
    pub3.publish(message1)
    
def main():
    global pub1, sub1, pub2, pub3, sub2, x_values, y_values  # declare as global variables
    global x_heated, y_heated, heated_node_temp
    x_heated = None
    y_heated = None

    x_heated = -1
    y_heated = -1
    heated_node_temp = -1

    param_file = open('/home/cam/ND_ws/robot_surface_heating_ws/src/param.yaml', 'r')
    parameters = yaml.safe_load(param_file)
    node_num_x = parameters['node_num_x']
    node_num_y = parameters['node_num_y']

    x_start = parameters['x_start_mp']
    x_end = parameters['x_end_mp']
    y_start = parameters['y_start_mp']
    y_end = parameters['y_end_mp']
    x_spacing = (x_end-x_start)/(node_num_x+1)  # Adjust this to change the spacing between points
    y_spacing = (y_end-y_start)/(node_num_y+1)  # Adjust this to change the spacing between points

    x_values = np.arange(x_start, x_end, x_spacing)
    y_values = np.arange(y_end, y_start, -y_spacing)
    x_values = np.delete(x_values, 0)
    y_values = np.delete(y_values, 0)

    print("This is X value", x_values)
    print("This is Y value", y_values)


    X, Y = np.meshgrid(x_values, y_values)
    num_rows, _ = X.shape
    _, num_columns = Y.shape
    rospy.init_node('orchestrator', anonymous=True)
    ## (x_pos, y_pos) are being sent from the orchestrator to the controller
    pub1 = rospy.Publisher('EE_node',Float32MultiArray,queue_size=2)

    ## indices of a node (x_low, y_low) being heated sent from the policy to orchestrator
    sub1 = rospy.Subscriber('node_indices_to_heat', Int32MultiArray, callback)

    ## 2d array of different temperatures being sent from the orchestrator to the policy
    pub2 = rospy.Publisher('temp_array_discretized',Float32MultiArray,queue_size=1)
    pub3 = rospy.Publisher('node_indices_heated', Int32MultiArray, queue_size=1)
    
    ## raw temperature 1d array from sensor node
    sub2 = rospy.Subscriber('temp_array_processed', Float32MultiArray, sensor_callback)

    rate = rospy.Rate(10) # 10hz
        
    rospy.spin()

if __name__ == "__main__":
    main()

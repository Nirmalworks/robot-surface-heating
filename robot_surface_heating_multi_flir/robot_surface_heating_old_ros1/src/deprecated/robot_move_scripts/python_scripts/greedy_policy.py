#!/usr/bin/python

import rospy
from std_msgs.msg import Float32MultiArray, Int32MultiArray
import yaml
import numpy as np

global pub
global x_heated, y_heated
global heated_nodes_data

def callback(data):
    global x_heated, y_heated, heated_nodes_data
    # DON'T CHANGE THIS PART BELOW
    heated_nodes_data = np.array(data.data)
    
def indices_callback(data):
    global x_heated, y_heated, heated_nodes_data
    x_heated_curr = data.data[0]
    y_heated_curr = data.data[1]

    # DON'T CHANGE THIS PART BELOW

    file = open('/home/cam/ND_ws/robot_surface_heating_ws/src/param.yaml', 'r')
    parameters = yaml.safe_load(file)
    node_num_x = parameters['node_num_x']
    node_num_y = parameters['node_num_y']
    reshaped_data = heated_nodes_data.reshape(node_num_x+2, node_num_y+2)
    heated_nodes_data_formatted = reshaped_data[1:-1, 1:-1]
    # ## DON'T CHANGE THE PART ABOVE
    
    min_index = np.argmin(heated_nodes_data_formatted)
    # Convert the 1D index to 2D index
    min_index_2d = np.unravel_index(min_index, heated_nodes_data_formatted.shape)

    msg = Int32MultiArray()

    print("Temperature: "+str(heated_nodes_data_formatted[x_heated_curr, y_heated_curr])+" at index: "+str(x_heated_curr) +", "+str(y_heated_curr))
    print()

    switch_temp = parameters['switch_temp']
    if(x_heated_curr == -1 and y_heated_curr == -1):
        msg.data = [min_index_2d[0], min_index_2d[1]]
        x_heated_curr = min_index_2d[0]
        y_heated_curr = min_index_2d[1]
        pub.publish(msg)
    elif(heated_nodes_data_formatted[x_heated_curr, y_heated_curr] > switch_temp):
        msg.data = [min_index_2d[0], min_index_2d[1]]
        x_heated_curr = min_index_2d[0]
        y_heated_curr = min_index_2d[1]
        pub.publish(msg)
    elif(heated_nodes_data_formatted[x_heated_curr, y_heated_curr] < switch_temp):
        msg.data = [x_heated_curr, y_heated_curr]
        pub.publish(msg)


def main():
    global pub
    global x_heated, y_heated
    x_heated = -1
    y_heated = -1
    rospy.init_node('greedy_policy', anonymous=True)
    pub = rospy.Publisher('node_indices_to_heat',Int32MultiArray,queue_size=1)
    sub = rospy.Subscriber('temp_array_processed', Float32MultiArray, callback)
    sub2 = rospy.Subscriber('node_indices_heated', Int32MultiArray, indices_callback)

    rate = rospy.Rate(10) # 10hz
    
    ######################### ENTER NODE NUMBER HERE ##################

    
    rospy.spin()

if __name__ == "__main__":
    main()

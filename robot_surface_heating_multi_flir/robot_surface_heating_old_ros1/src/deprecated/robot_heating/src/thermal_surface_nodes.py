#!/usr/bin/env python3

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
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray

#import rospy     
#from std_msgs.msg import Float64


class SurfaceTemperatureNodes(object):
    def __init__(self, parameters):

        rospy.Subscriber('thermal/transformed_temperature_data_F', Float32MultiArray, self.temperature_callback)
        rospy.Subscriber('thermal/transformed_temperature_image', Image, self.thermal_img_callback)

        self.temp_array_pub = rospy.Publisher('Temp_Array',Float32MultiArray,queue_size=10)
        rospy.init_node('Temp_Array_Node', anonymous=True)

        self.parameters = parameters    

            ################# CHANGE THIS VALUE FOR DENSITY OF NODES #######################
        # MAKE SURE THIS MATCHES THE NODE NUM IN ROBOT ARM SCRIPTS
        nodes_x = parameters['nodes_x']
        nodes_y = parameters['nodes_y']
        ################## CHANGE THIS VALUE FOR DENSITY OF NODES #######################

        x_start = parameters['x_start']
        x_end = parameters['x_end']
        y_start = parameters['y_start']
        y_end = parameters['y_end']
        
        x_spacing = x_end/(nodes_x+1)  # Adjust this to change the spacing between points
        y_spacing = y_end/(nodes_y+1)  # Adjust this to change the spacing between points

        # Create equally spaced points along X and Y axes
        x_values = np.arange(x_start, x_end, x_spacing)
        y_values = np.arange(y_start, y_end, y_spacing)

        # Add a value 999 at the end for edges
        #x_values = np.delete(x_values, 0)
        #y_values = np.delete(y_values, 0)
        x_values = np.append(x_values,[1999])
        y_values = np.append(y_values,[999])

        self.x_values = x_values
        self.y_values = y_values
        
        self.sheet_temperature_data = None

        # Create the mesh grid using np.meshgrid
        X, Y = np.meshgrid(x_values, y_values)
        num_rows, _ = X.shape
        _, num_columns = Y.shape
        
        
        self.log_csv = False
        
        self.start_time = None
        pass

    def temperature_callback(self,data):
        #  Get the dimensions from the layout
        # rows = data.layout.dim[0].size
        # cols = data.layout.dim[1].size
        # print(rows, cols)
        # input()
        print(data.layout)
        # # Reshape the data
        # self.sheet_temperature_data = np.reshape(data.data, (rows, cols))
        self.sheet_temperature_data = data.data

    def thermal_img_callback(self,data):
        self.thermal_img = data.data
   
    def log_csv_data(self, log = False, csv_filename = None):
        if log:
            self.log_csv = True
            self.csv_filename = csv_filename
        else:
            self.log_csv = False    
    
    def process_data(self):
        message = Float32MultiArray()
        rows, cols = (len(self.x_values), len(self.y_values))
        zero_arr = [0]*cols*rows
        message.data = zero_arr
        rospy.loginfo(message)
        self.temp_array_pub.publish(message)
        
        node_num_x = self.parameters['node_num_x']
        node_num_y = self.parameters['node_num_y']

        # Set up CSV output file
        if self.log_csv:
            with open(self.csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(["Time_Seconds"]+[f"Node_{i+1}" for i in range( node_num_x *  node_num_y )]+["Node_Index_x"]+["Node_Index_y"])
        # csv_filename = parameters["csv_filename"]
        # with open(csv_filename, mode='w', newline='') as file:
        #     writer = csv.writer(file)
        #     writer.writerow(["Time_Seconds"]+[f"Node_{i+1}" for i in range( node_num_x *  node_num_y )]+["Node_Index_x"]+["Node_Index_y"])
        
        

        if self.start_time is None:
            self.start_time = time.time()

        current_time = time.time() - self.start_time  # Calculate elapsed time in seconds
        if self.sheet_temperature_data is not None:
            data_transform = np.array(self.sheet_temperature_data)
            
            print(data_transform)
            print("This is the Shape", data_transform.shape)
            input()
        pass
        # input()

            # print("This is the Shape", data_transform.shape)
            # minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data_transform)
            
            # ############## EXTRACT DATA POINTS AT DESIRED LOCATIONS ################
            # # Loop through each X and Y point using nested loops
            # temperature_data = []
            # for i in range(len(x_values)):
            #     for j in range(len(y_values)):
            #         x_point = int(x_values[i])
            #         y_point = int(y_values[j])
            #         val = data_transform[y_point, x_point]
            #         temp_array[i,j] = float(ktof(val))
            #         color = generate_color(val)
            #         display_temperature(img_transform, val, (x_point,y_point), color)
            #         temperature_data.append(val)

            # data_numpy = np.array(temp_array)
            # reshaped_data = data_numpy.reshape(node_num_x+2, node_num_y+2)
            # heated_nodes_data = reshaped_data[1:-1, 1:-1]  # get only the heated nodes
            # avg_temp = np.mean(heated_nodes_data)
            # max_temp = np.max(heated_nodes_data)
            # std_dev_temp = np.std(heated_nodes_data)
            # index_data = node_index_subscriber.get_EE_node_data()
            # if index_data is None:
            #     index_data = [0, 0]  # Default to [0, 0] if None
            # print("Thissssssssssssssssssssssssssssss is the index", index_data)
            # x_index_adjusted = index_data[0] - 1        
            # y_index_adjusted = index_data[1] - 1

            # # Write temperature data to CSV for each frame
            # with open(csv_filename, mode='a', newline='') as file:
            #     writer = csv.writer(file)
            #     row_data = [current_time]+list(heated_nodes_data.flatten())+[x_index_adjusted, y_index_adjusted]
            #     writer.writerow(row_data)
            # print("Temp Array", heated_nodes_data)
            # print("Avg Temp: ", avg_temp)
            # print("Max Temp: ", max_temp)
            # print("STD DEV TEMP", std_dev_temp)


class EE_nodeSubscriber:

    def __init__(self):
        # Create a subscriber for the "/Temp_Array" topic
        self.EE_node_subscriber = rospy.Subscriber("/EE_node", Int32MultiArray, self.EE_node_callback)
        self.EE_node_data = None

    def EE_node_callback(self, data):
        # This function will be called every time a message is received on the "/Temp_Array" topic
        # Access the data from the received message
        self.EE_node_data = data.data

    def get_EE_node_data(self):
        # Function to get the latest temperature data
        return self.EE_node_data
        
    

def main():
    file = open('/home/cam/st_heat/src/param.yaml', 'r')
    parameters = yaml.safe_load(file)

    surface_temp_nodes = SurfaceTemperatureNodes(parameters)
    # surface_temp_nodes.log_csv_data(log = True, csv_filename = parameters["csv_filename"])

    while not rospy.is_shutdown():
        current_time = time.time()
        surface_temp_nodes.process_data()



# def main(): 
#     file = open('/home/cam/st_heat/src/param.yaml', 'r')
#     parameters = yaml.safe_load(file)

#     node_index_subscriber = EE_nodeSubscriber()

#     # Generate random X and Y coordinates with variable spacing
#     # Define the range and spacing for X and Y values

#     ################## CHANGE THIS VALUE FOR DENSITY OF NODES #######################
#     # MAKE SURE THIS MATCHES THE NODE NUM IN ROBOT ARM SCRIPTS
#     nodes_x = parameters['nodes_x']
#     nodes_y = parameters['nodes_y']
#     ################## CHANGE THIS VALUE FOR DENSITY OF NODES #######################

#     x_start = parameters['x_start']
#     x_end = parameters['x_end']
#     y_start = parameters['y_start']
#     y_end = parameters['y_end']
    
#     x_spacing = x_end/(nodes_x+1)  # Adjust this to change the spacing between points
#     y_spacing = y_end/(nodes_y+1)  # Adjust this to change the spacing between points

#     # Create equally spaced points along X and Y axes
#     x_values = np.arange(x_start, x_end, x_spacing)
#     y_values = np.arange(y_start, y_end, y_spacing)

#     # Add a value 999 at the end for edges
#     #x_values = np.delete(x_values, 0)
#     #y_values = np.delete(y_values, 0)
#     x_values = np.append(x_values,[1999])
#     y_values = np.append(y_values,[999])

#     # Create the mesh grid using np.meshgrid
#     X, Y = np.meshgrid(x_values, y_values)
#     num_rows, _ = X.shape
#     _, num_columns = Y.shape



#     ############## SET UP CSV OUTPUT ################
#     temp_array = np.zeros((len(x_values),len(y_values)))
#     timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")


#     ############## SET UP NODE OUTPUT ################
#     pub = rospy.Publisher('Temp_Array',Float32MultiArray,queue_size=10)
#     rospy.init_node('Temp_Array_Node', anonymous=True)

#     message = Float32MultiArray()
#     rows, cols = (len(x_values), len(y_values))
#     zero_arr = [0]*cols*rows
#     message.data = zero_arr
#     rospy.loginfo(message)
#     pub.publish(message)
    
#     node_num_x = parameters['node_num_x']
#     node_num_y = parameters['node_num_y']
#     # Constant joint velocity value
#     joint_velocity = 0.5

#     # Set up CSV output file
#     csv_filename = parameters["csv_filename"]
#     with open(csv_filename, mode='w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(["Time_Seconds"]+[f"Node_{i+1}" for i in range( node_num_x *  node_num_y )]+["Node_Index_x"]+["Node_Index_y"])
    
    
#     start_time = time.time()  # Start time for calculating elapsed time
#     while not rospy.is_shutdown():
#         current_time = time.time() - start_time  # Calculate elapsed time in seconds

#         print("This is the Shape", data_transform.shape)
#         minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data_transform)
        
#         ############## EXTRACT DATA POINTS AT DESIRED LOCATIONS ################
#         # Loop through each X and Y point using nested loops
#         temperature_data = []
#         for i in range(len(x_values)):
#             for j in range(len(y_values)):
#                 x_point = int(x_values[i])
#                 y_point = int(y_values[j])
#                 val = data_transform[y_point, x_point]
#                 temp_array[i,j] = float(ktof(val))
#                 color = generate_color(val)
#                 display_temperature(img_transform, val, (x_point,y_point), color)
#                 temperature_data.append(val)

#         data_numpy = np.array(temp_array)
#         reshaped_data = data_numpy.reshape(node_num_x+2, node_num_y+2)
#         heated_nodes_data = reshaped_data[1:-1, 1:-1]  # get only the heated nodes
#         avg_temp = np.mean(heated_nodes_data)
#         max_temp = np.max(heated_nodes_data)
#         std_dev_temp = np.std(heated_nodes_data)
#         index_data = node_index_subscriber.get_EE_node_data()
#         if index_data is None:
#             index_data = [0, 0]  # Default to [0, 0] if None
#         print("Thissssssssssssssssssssssssssssss is the index", index_data)
#         x_index_adjusted = index_data[0] - 1        
#         y_index_adjusted = index_data[1] - 1

#         # Write temperature data to CSV for each frame
#         with open(csv_filename, mode='a', newline='') as file:
#             writer = csv.writer(file)
#             row_data = [current_time]+list(heated_nodes_data.flatten())+[x_index_adjusted, y_index_adjusted]
#             writer.writerow(row_data)
#         print("Temp Array", heated_nodes_data)
#         print("Avg Temp: ", avg_temp)
#         print("Max Temp: ", max_temp)
#         print("STD DEV TEMP", std_dev_temp)
        
        


if __name__ == '__main__':
  main()

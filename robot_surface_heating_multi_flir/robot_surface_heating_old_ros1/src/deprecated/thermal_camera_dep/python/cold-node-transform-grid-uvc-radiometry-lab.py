#!/usr/bin/env python
# -*- coding: utf-8 -*-

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


#import rospy     
#from std_msgs.msg import Float64

BUF_SIZE = 2
q = Queue(BUF_SIZE)


def py_frame_callback(frame, userptr):

  array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
  data = np.frombuffer(
    array_pointer.contents, dtype=np.dtype(np.uint16)
  ).reshape(
    frame.contents.height, frame.contents.width
  ) # no copy


  if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
    return

  if not q.full():
    q.put(data)

PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)

def ktof(val):
  return (1.8 * ktoc(val) + 32.0)

def ktoc(val):
  return (val - 27315) / 100.0

def raw_to_8bit(data):
  cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
  np.right_shift(data, 8, data)
  return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)

def display_temperature(img, val_k, loc, color):
  val = ktof(val_k)
  cv2.putText(img,"{0:.1f} degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
  x, y = loc
  cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
  cv2.line(img, (x, y - 2), (x, y + 2), color, 1)

def generate_color(input_value):
    input_value = ktoc(input_value)
    max_val = 30
    min_val = 20
    input_value = max(min_val, min(max_val, input_value))
    # Define color mappings
    color_mapping = {
        20: [255, 0, 0],    # Blue
        25: [0, 255, 0],  # Yellow
        30: [0, 0, 255],    # Red
    }

    # Ensure the input value is in the color mapping, or interpolate if not
    if input_value in color_mapping:
        return color_mapping[input_value]
    elif 20 < input_value < 30:
        # Interpolate between blue and yellow for values between 20 and 25
        blue = color_mapping[20]
        yellow = color_mapping[25]
        interpolation_factor = (input_value - 20) / (25 - 20)
        interpolated_color = [int((1 - interpolation_factor) * c1 + interpolation_factor * c2) for c1, c2 in zip(blue, yellow)]
        return interpolated_color
    elif 25 < input_value < 30:
        # Interpolate between yellow and red for values between 25 and 30
        yellow = color_mapping[25]
        red = color_mapping[30]
        interpolation_factor = (input_value - 25) / (30 - 25)
        interpolated_color = [int((1 - interpolation_factor) * c1 + interpolation_factor * c2) for c1, c2 in zip(yellow, red)]
        return interpolated_color
    else:
        raise ValueError("Input value is outside the specified range.")

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
  ctx = POINTER(uvc_context)()
  dev = POINTER(uvc_device)()
  devh = POINTER(uvc_device_handle)()
  ctrl = uvc_stream_ctrl()
  node_index_subscriber = EE_nodeSubscriber()

  res = libuvc.uvc_init(byref(ctx), 0)

  #pub = rospy.Publisher('temp_nodes', Float64, queue_size=10)
  #rospy.init_node('temp_talker', anonymous=True)

  if res < 0:
    print("uvc_init error")
    exit(1)

  try:
    res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
    if res < 0:
      print("uvc_find_device error")
      exit(1)

    try:
      res = libuvc.uvc_open(dev, byref(devh))
      if res < 0:
        print("uvc_open error")
        exit(1)

      print("device opened!")

      print_device_info(devh)
      print_device_formats(devh)

      frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
      if len(frame_formats) == 0:
        print("device does not support Y16")
        exit(1)

      libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
        frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval)
      )

      res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
      if res < 0:
        print("uvc_start_streaming failed: {0}".format(res))
        exit(1)
      
      # Generate random X and Y coordinates with variable spacing
      # Define the range and spacing for X and Y values

      ################## CHANGE THIS VALUE FOR DENSITY OF NODES #######################
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

      # Create the mesh grid using np.meshgrid
      X, Y = np.meshgrid(x_values, y_values)
      num_rows, _ = X.shape
      _, num_columns = Y.shape
      print("HELLLLLLLOOOOOOOOOOOOOOOOOOOOOOOOOO")
      print(x_values)
      print(y_values)


      ################## CALIBRATE THESE VALUES FOR PERSPECTIVE TRANSFORM #######################
      # # Bottom Right Corner 
      # (578, 389)
      # # Top Right Corner 
      # (466, 306)
      # # Top Left Corner 
      # (190, 341)
      # # Bottom Left Corner 
      # (106, 438)

      # Top Left, Top Right, Bot Left, Bot Right
      orig_pts = np.float32(parameters['orig_pts'])
      dest_pts = np.float32(parameters['dest_pts'])

      ############## SET UP CSV OUTPUT ################
      temp_array = np.zeros((len(x_values),len(y_values)))
      timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")


      ############## SET UP NODE OUTPUT ################
      pub = rospy.Publisher('Temp_Array',Float32MultiArray,queue_size=10)
      rospy.init_node('Temp_Array_Node', anonymous=True)

      message = Float32MultiArray()
      rows, cols = (len(x_values), len(y_values))
      zero_arr = [0]*cols*rows
      message.data = zero_arr
      rospy.loginfo(message)
      pub.publish(message)
      
      node_num_x = parameters['node_num_x']
      node_num_y = parameters['node_num_y']
      # Constant joint velocity value
      joint_velocity = 0.5

      # Set up CSV output file
      csv_filename = parameters["csv_filename"]
      with open(csv_filename, mode='w', newline='') as file:
          writer = csv.writer(file)
          writer.writerow(["Time_Seconds"]+[f"Node_{i+1}" for i in range( node_num_x *  node_num_y )]+["Node_Index_x"]+["Node_Index_y"])
      try:
          start_time = time.time()  # Start time for calculating elapsed time
          while True:
            current_time = time.time() - start_time  # Calculate elapsed time in seconds

            data = q.get(True, 500)
            if data is None:
              break
            data = cv2.resize(data[:,:], (640, 480))

            data_cop = data.copy() # MAKE DATA COPY
            img = raw_to_8bit(data)
            img_lines = img.copy() # MAKE COPY OF IMAGE

            ############## TRANSFORM IMAGE #######################################
            cv2.line(img_lines, tuple(orig_pts[0]), tuple(orig_pts[1]), (255,0,0), 2)
            cv2.line(img_lines, tuple(orig_pts[1]), tuple(orig_pts[3]), (255,0,0), 2)
            cv2.line(img_lines, tuple(orig_pts[3]), tuple(orig_pts[2]), (255,0,0), 2)
            cv2.line(img_lines, tuple(orig_pts[2]), tuple(orig_pts[0]), (255,0,0), 2)
            # Get perspective transform M
            M = cv2.getPerspectiveTransform(orig_pts, dest_pts)
            # warp image with M
            # transform_x: 2000, transform_y: 1000
            img_transform = cv2.warpPerspective(img, M, (parameters["transform_x"], parameters["transform_y"]))
            data_transform = cv2.warpPerspective(data_cop, M, (parameters["transform_x"], parameters["transform_y"]))

            print("This is the Shape", data_transform.shape)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data_transform)
           
            ############## EXTRACT DATA POINTS AT DESIRED LOCATIONS ################
            # Loop through each X and Y point using nested loops
            temperature_data = []
            for i in range(len(x_values)):
              for j in range(len(y_values)):
                  x_point = int(x_values[i])
                  y_point = int(y_values[j])
                  val = data_transform[y_point, x_point]
                  temp_array[i,j] = float(ktof(val))
                  color = generate_color(val)
                  display_temperature(img_transform, val, (x_point,y_point), color)
                  temperature_data.append(val)

            data_numpy = np.array(temp_array)
            reshaped_data = data_numpy.reshape(node_num_x+2, node_num_y+2)
            heated_nodes_data = reshaped_data[1:-1, 1:-1]  # get only the heated nodes
            avg_temp = np.mean(heated_nodes_data)
            max_temp = np.max(heated_nodes_data)
            std_dev_temp = np.std(heated_nodes_data)
            index_data = node_index_subscriber.get_EE_node_data()
            if index_data is None:
              index_data = [0, 0]  # Default to [0, 0] if None
            print("Thissssssssssssssssssssssssssssss is the index", index_data)
            x_index_adjusted = index_data[0] - 1        
            y_index_adjusted = index_data[1] - 1

            # Write temperature data to CSV for each frame
            with open(csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                row_data = [current_time]+list(heated_nodes_data.flatten())+[x_index_adjusted, y_index_adjusted]
                writer.writerow(row_data)
            print("Temp Array", heated_nodes_data)
            print("Avg Temp: ", avg_temp)
            print("Max Temp: ", max_temp)
            print("STD DEV TEMP", std_dev_temp)
            
          
            ####################### WRITE DATA TO NODE ##########################
            message.data = np.reshape(temp_array,[len(x_values)*len(y_values)])
            pub.publish(message)

            print(float(ktof(maxVal)))
            display_temperature(img_transform, minVal, minLoc, (255, 0, 0))
            display_temperature(img_transform, maxVal, maxLoc, (0, 0, 255))
            ############################ SHOW DATA ################################
            cv2.imshow('Lepton Radiometry', img_transform)
            cv2.waitKey(1)

          cv2.destroyAllWindows()
      finally:
        libuvc.uvc_stop_streaming(devh)

      print("done")
    finally:
      libuvc.uvc_unref_device(dev)
  finally:
    libuvc.uvc_exit(ctx)

if __name__ == '__main__':
  main()

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
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


#import rospy     
#from std_msgs.msg import Float64

BUF_SIZE = 2
q = Queue(BUF_SIZE)
q_1 = Queue(BUF_SIZE)



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

def py_frame_callback_1(frame, userptr):

  array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
  data = np.frombuffer(
    array_pointer.contents, dtype=np.dtype(np.uint16)
  ).reshape(
    frame.contents.height, frame.contents.width
  ) # no copy

  # data = np.fromiter(
  #   frame.contents.data, dtype=np.dtype(np.uint8), count=frame.contents.data_bytes
  # ).reshape(
  #   frame.contents.height, frame.contents.width, 2
  # ) # copy

  if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
    return

  if not q_1.full():
    q_1.put(data)

PTR_PY_FRAME_CALLBACK_1 = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback_1)


def ktof(val):
  return (1.8 * ktoc(val) + 32.0)

def ktoc(val):
  return (val - 27315) / 100.0

def raw_to_8bit(data):
  cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
  np.right_shift(data, 8, data)
  return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2RGB)

def display_temperature(img, val, loc, color, F_conversion = True):
  if F_conversion:
    val = ktof(val)
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
  file = open('/home/cam/ND_ws/heat_ws/robot_surface_heating/src/param.yaml', 'r')
  parameters = yaml.safe_load(file)
  ctx = POINTER(uvc_context)()
  devs = POINTER(POINTER(uvc_device))()
  devh = (POINTER(uvc_device_handle) * 2)()
  ctrl = uvc_stream_ctrl()

  # node_index_subscriber = EE_nodeSubscriber() ########################

  res = libuvc.uvc_init(byref(ctx), 0)

  #pub = rospy.Publisher('temp_nodes', Float64, queue_size=10)
  #rospy.init_node('temp_talker', anonymous=True)

  if res < 0:
    print("uvc_init error")
    exit(1)

  try:
    res = libuvc.uvc_find_devices(ctx, byref(devs), PT_USB_VID, PT_USB_PID, 0)
    if res < 0:
      print("uvc_find_device error")
      exit(1)

    try:
      res = libuvc.uvc_open(devs[0], byref(devh[0]))
      res = libuvc.uvc_open(devs[1], byref(devh[1]))
      if res < 0:
        print("uvc_open error")
        exit(1)

      print("device opened!")

      # print_device_info(devh)
      # print_device_formats(devh)

      frame_formats = uvc_get_frame_formats_by_guid(devh[0], VS_FMT_GUID_Y16)
      frame_formats_1 = uvc_get_frame_formats_by_guid(devh[1], VS_FMT_GUID_Y16)

      if len(frame_formats) == 0:
        print("device does not support Y16")
        exit(1)

      libuvc.uvc_get_stream_ctrl_format_size(devh[0], byref(ctrl), UVC_FRAME_FORMAT_Y16,
        frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval)
      )

      libuvc.uvc_get_stream_ctrl_format_size(devh[1], byref(ctrl), UVC_FRAME_FORMAT_Y16,
        frame_formats_1[0].wWidth, frame_formats_1[0].wHeight, int(1e7 / frame_formats_1[0].dwDefaultFrameInterval)
      )


      res = libuvc.uvc_start_streaming(devh[0], byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
      res = libuvc.uvc_start_streaming(devh[1], byref(ctrl), PTR_PY_FRAME_CALLBACK_1, None, 0)

      if res < 0:
        print("uvc_start_streaming failed: {0}".format(res))
        exit(1)

      num_cameras = parameters['num_cameras']
      
      # # Generate random X and Y coordinates with variable spacing
      # # Define the range and spacing for X and Y values

      # ################## CHANGE THIS VALUE FOR DENSITY OF NODES #######################
      # # MAKE SURE THIS MATCHES THE NODE NUM IN ROBOT ARM SCRIPTS
      nodes_x = parameters['nodes_x']
      nodes_y = parameters['nodes_y']
      # ################## CHANGE THIS VALUE FOR DENSITY OF NODES #######################

      # x_start = parameters['x_start']
      # x_end = parameters['x_end']
      # y_start = parameters['y_start']
      # y_end = parameters['y_end']
      
      # x_spacing = x_end/(nodes_x+1)  # Adjust this to change the spacing between points
      # y_spacing = y_end/(nodes_y+1)  # Adjust this to change the spacing between points

      # # Create equally spaced points along X and Y axes
      # x_values = np.arange(x_start, x_end, x_spacing)
      # y_values = np.arange(y_start, y_end, y_spacing)

      # # Add a value 999 at the end for edges
      # #x_values = np.delete(x_values, 0)
      # #y_values = np.delete(y_values, 0)
      # x_values = np.append(x_values,[1999])
      # y_values = np.append(y_values,[999])

      # # Create the mesh grid using np.meshgrid
      # X, Y = np.meshgrid(x_values, y_values)
      # num_rows, _ = X.shape
      # _, num_columns = Y.shape
      # print("HELLLLLLLOOOOOOOOOOOOOOOOOOOOOOOOOO")
      # print(x_values)
      # print(y_values)


      ################## CALIBRATE THESE VALUES FOR PERSPECTIVE TRANSFORM #######################

      # Top Left, Top Right, Bot Left, Bot Right
      orig_pts = np.float32(parameters['orig_pts'])
      dest_pts = np.float32(parameters['dest_pts'])

      orig_pts_1 = np.float32(parameters['orig_pts_1'])
      dest_pts_1 = np.float32(parameters['dest_pts_1'])

      ############## SET UP CSV OUTPUT ################
      # temp_array = np.zeros((len(x_values),len(y_values)))
      # timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")


      # ############## SET UP NODE OUTPUT ################
      # pub = rospy.Publisher('Temp_Array',Float32MultiArray,queue_size=10)
      # rospy.init_node('Temp_Array_Node', anonymous=True)

      # message = Float32MultiArray()
      # rows, cols = (len(x_values), len(y_values))
      # zero_arr = [0]*cols*rows
      # message.data = zero_arr
      # rospy.loginfo(message)
      # pub.publish(message)
      
      # node_num_x = parameters['node_num_x']
      # node_num_y = parameters['node_num_y']
      # # Constant joint velocity value
      # joint_velocity = 0.5

      # # Set up CSV output file
      # csv_filename = parameters["csv_filename"]


      ############### Thermal Image Publisher ############################
      #  Initialize CvBridge
      bridge = CvBridge()

      # Set up publishers
      # pub_raw_image = rospy.Publisher('thermal/raw_temperature_image', Image, queue_size=10)
      # pub_raw_data = rospy.Publisher('thermal/raw_temperature_data_F', Float32MultiArray, queue_size=10)
      # pub_transformed_data = rospy.Publisher('thermal/temp_array_raw', Float32MultiArray, queue_size=10)
      # pub_transformed_image = rospy.Publisher('thermal/transformed_temperature_image', Image, queue_size=10)
      pub_temp_array = rospy.Publisher('temp_array_processed', Float32MultiArray, queue_size=1)
      file = open('/home/cam/ND_ws/heat_ws/robot_surface_heating/src/param.yaml', 'r')
      parameters = yaml.safe_load(file)
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
      x_values = np.append(x_values,[x_end-1])
      y_values = np.append(y_values,[y_end-1])

      orig_pts = np.float32(parameters['orig_pts'])
      dest_pts = np.float32(parameters['dest_pts'])

      orig_pts_1 = np.float32(parameters['orig_pts_1'])
      dest_pts_1 = np.float32(parameters['dest_pts_1'])

      rospy.init_node('Thermal_Sensor_Node', anonymous=True)
      
      def heat_command_callback(data):
        # Assuming 'data' is a Point message with 'x' and 'y' fields
        node_x = data.data[0] + 1
        node_y = data.data[1] + 1

        if node_x == -1 and node_y == -1:
           pass
        # # Convert node coordinates to pixel coordinates
        # # This depends on how your image and nodes are related
        heat_pixel_x = int(node_x * x_spacing)
        heat_pixel_y = int(node_y * y_spacing)

        # # Draw a big red dot on the image at the node being heated
        cv2.circle(img_transform, (heat_pixel_x, heat_pixel_y), 10, (0, 255, 0), -1)
       
      rospy.Subscriber('/controller_state', Int32MultiArray, heat_command_callback)
      # with open(csv_filename, mode='w', newline='') as file:
      #     writer = csv.writer(file)
      #     writer.writerow(["Time_Seconds"]+[f"Node_{i+1}" for i in range( node_num_x *  node_num_y )]+["Node_Index_x"]+["Node_Index_y"])
      try:
          start_time = time.time()  # Start time for calculating elapsed time
          while not rospy.is_shutdown():
            current_time = time.time() - start_time  # Calculate elapsed time in seconds

            data = q.get(True, 500)
            data_1 = q_1.get(True, 500)

            if data is None:
              print("Camera 0 is not receiving data")
              break
            if data_1 is None:
              print("Camera 1 is not receiving data")
              break

            # Relative calibration of two cameras
    
            if parameters['thermal_dynamic_offset'] is True:
               raise NotImplementedError("Dynamic Relative calibration of two cameras is not implemented yet")
            else:

                data = data + parameters['cam_0_static_offset']
                data_1 = data_1+ parameters['cam_1_static_offset']

            data = cv2.resize(data[:,:], (640, 480))
            data_1 = cv2.resize(data_1[:,:], (640, 480))

            data_cop = data.copy() # MAKE DATA COPY
            data_1_cop = data_1.copy() # MAKE DATA COPY

            img = raw_to_8bit(data)
            img_1 = raw_to_8bit(data_1)
            img_lines = img.copy() # MAKE COPY OF IMAGE
            img_lines_1 = img_1.copy() # MAKE COPY OF IMAGE

            ############## TRANSFORM IMAGE #######################################
            # cv2.line(img_lines, tuple(orig_pts[0]), tuple(orig_pts[1]), (255,0,0), 2)
            # cv2.line(img_lines, tuple(orig_pts[1]), tuple(orig_pts[3]), (255,0,0), 2)
            # cv2.line(img_lines, tuple(orig_pts[3]), tuple(orig_pts[2]), (255,0,0), 2)
            # cv2.line(img_lines, tuple(orig_pts[2]), tuple(orig_pts[0]), (255,0,0), 2)
            # Get perspective transform M
            M = cv2.getPerspectiveTransform(orig_pts, dest_pts)

            M_1 = cv2.getPerspectiveTransform(orig_pts_1, dest_pts_1)

            # Adjust translation of M matrix
            # x_offset = -20
            # y_offset = -20
            # translation_matrix = np.float32([[1, 0, -x_offset], [0, 1, -y_offset], [0, 0, 1]])
            # M = translation_matrix @ M

            # # Calculate the new size of the image after the perspective transformation
            # corners = np.array([[0, 0], [img.shape[1] - 1, 0], [0, img.shape[0] - 1], [img.shape[1] - 1, img.shape[0] - 1]], dtype=np.float32)
            # transformed_corners = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), M)
            # x_min, y_min = np.int0(transformed_corners.min(axis=0).ravel() - 0.5)
            # x_max, y_max = np.int0(transformed_corners.max(axis=0).ravel() + 0.5)

            # corners_1 = np.array([[0, 0], [img_1.shape[1] - 1, 0], [0, img_1.shape[0] - 1], [img_1.shape[1] - 1, img_1.shape[0] - 1]], dtype=np.float32)
            # transformed_corners_1 = cv2.perspectiveTransform(corners_1.reshape(-1, 1, 2), M_1)
            # x_min_1, y_min_1 = np.int0(transformed_corners_1.min(axis=0).ravel() - 0.5)
            # x_max_1, y_max_1 = np.int0(transformed_corners_1.max(axis=0).ravel() + 0.5)

            # # Calculate the translation needed to make all coordinates non-negative
            # x_translation = -x_min if x_min < 0 else 0
            # y_translation = -y_min if y_min < 0 else 0

            # x_translation_1 = -x_min_1 if x_min_1 < 0 else 0
            # y_translation_1 = -y_min_1 if y_min_1 < 0 else 0
            # # Create the translation matrix
            # translation_matrix = np.float32([[1, 0, x_translation], [0, 1, y_translation], [0, 0, 1]])

            # translation_matrix_1 = np.float32([[1, 0, x_translation_1], [0, 1, y_translation_1], [0, 0, 1]])

            # # Apply the translation to the perspective transformation matrix
            # M = translation_matrix @ M
            # M_1 = translation_matrix_1 @ M_1

            # # Calculate the new size of the image after the translation
            # transform_x = x_max - x_min + x_translation
            # transform_y = y_max - y_min + y_translation

            # transform_x_1 = x_max_1 - x_min_1 + x_translation_1
            # transform_y_1 = y_max_1 - y_min_1 + y_translation_1

            transform_x = 1000
            transform_y = 1000
            transform_x_1 = 1000
            transform_y_1 = 1000

            # Apply the perspective transformations
            # img_transform = cv2.warpPerspective(img, M, (transform_x, transform_y),flags=cv2.INTER_CUBIC)
            # data_transform = cv2.warpPerspective(data_cop, M, (transform_x, transform_y),flags=cv2.INTER_CUBIC)

            # img_transform_1 = cv2.warpPerspective(img_1, M_1, (transform_x_1, transform_y_1),flags=cv2.INTER_CUBIC)
            # data_transform_1 = cv2.warpPerspective(data_1_cop, M_1, (transform_x_1, transform_y_1),flags=cv2.INTER_CUBIC)

            img_transform = cv2.warpPerspective(img, M, (transform_x, transform_y))
            data_transform = cv2.warpPerspective(data_cop, M, (transform_x, transform_y))

            img_transform_1 = cv2.warpPerspective(img_1, M_1, (transform_x_1, transform_y_1))
            data_transform_1 = cv2.warpPerspective(data_1_cop, M_1, (transform_x_1, transform_y_1))


            if num_cameras == 2:
              img_transform_c = cv2.hconcat([img_transform, img_transform_1])
              data_transform_c = cv2.hconcat([data_transform, data_transform_1])
            elif num_cameras == 1:
              img_transform_c = img_transform
              data_transform_c = data_transform
            
            # img_transform = cv2.warpPerspective(img, M, (parameters["transform_x"], parameters["transform_y"]))
            # data_transform = cv2.warpPerspective(data_cop, M, (parameters["transform_x"], parameters["transform_y"]))
            
            # img_transform_1 = cv2.warpPerspective(img_1, M_1, (parameters["transform_x"], parameters["transform_y"]))
            # data_transform_1 = cv2.warpPerspective(data_1_cop, M_1, (parameters["transform_x"], parameters["transform_y"]))
            
            # # put rectangle on the image based on the dst corners
            # cv2.rectangle(img_transform, (0+x_translation, 0+y_translation), (1000+x_translation, 1000+y_translation), (0, 255, 0), 5)
            # cv2.rectangle(img_transform_1, (0+x_translation_1, 0+y_translation_1), (1000+x_translation_1, 1000+y_translation_1), (0, 255, 0), 5)

            # Resize both images while keeping the aspect ratio
            # Original image size
            original_height, original_width = img_transform.shape[:2]
            original_height_1, original_width_1 = img_transform_1.shape[:2]

            # Desired width
            new_width = 1000


            # Calculate the aspect ratio
            aspect_ratio = original_width / original_height
            aspect_ratio_1 = original_width_1 / original_height_1

            # Calculate the new height while maintaining the aspect ratio
            new_height = int(new_width / aspect_ratio)
            new_height_1 = int(new_width / aspect_ratio_1)

            # Resize the image
            # img_transform = cv2.resize(img_transform, (new_width, new_height))
            # img_transform_1 = cv2.resize(img_transform_1, (new_width, new_height_1))
            # data_transform = cv2.resize(data_transform, (640, 480))

            # print("This is the Shape", data_transform.shape)

            # print("This is the Shape", data_transform.shape)
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data_transform)
            minVal_1, maxVal_1, minLoc_1, maxLoc_1 = cv2.minMaxLoc(data_transform_1)
            minVal_c, maxVal_c, minLoc_c, maxLoc_c = cv2.minMaxLoc(data_transform_c)

            # Get Fareheit values
            data_transform_f = ktof(data_transform)
            data_transform_1_f = ktof(data_transform_1)
            data_transform_c_f = ktof(data_transform_c)


            # Get area params
            region_size = 50
            half_size = region_size // 2

          # 



            ################################################## RECOMMENT BACK#############
            temp_array = np.zeros((len(x_values),len(y_values)))
            ############## EXTRACT DATA POINTS AT DESIRED LOCATIONS ################
            # Loop through each X and Y point using nested loops
            temperature_data = []
        
            for i in range(len(x_values)):
              for j in range(len(y_values)):
                  x_point = int(x_values[i])
                  y_point = int(y_values[j])
                  # val = data_transform_c[y_point, x_point]
                  # temp_array[i,j] = float(ktof(val))

                  # val = data_transform_c_f[y_point, x_point]

                  #Calculate the start and end indices for both dimensions
                  y_start = max(0, y_point - half_size)
                  y_end = min(data_transform_c_f.shape[0], y_point + half_size + 1)
                  x_start = max(0, x_point - half_size)
                  x_end = min(data_transform_c_f.shape[1], x_point + half_size + 1)
                        
                  val = np.max(data_transform_c_f[y_start:y_end, x_start:x_end])


                  
                  temp_array[i,j] =float(val)
                  color = generate_color(val)
                  display_temperature(img_transform_c, val, (x_point,y_point), color,F_conversion = False)
                  temperature_data.append(val)
            #####################################################################



            # data_numpy = np.array(temp_array)
            # reshaped_data = data_numpy.reshape(node_num_x+2, node_num_y+2)
            # heated_nodes_data = reshaped_data[1:-1, 1:-1]  # get only the heated nodes
            # avg_temp = np.mean(heated_nodes_data)
            # max_temp = np.max(heated_nodes_data)
            # std_dev_temp = np.std(heated_nodes_data)
            # index_data = node_index_subscriber.get_EE_node_data()
            # if index_data is None:
            #   index_data = [0, 0]  # Default to [0, 0] if None
            # print("Thissssssssssssssssssssssssssssss is the index", index_data)
            # x_index_adjusted = index_data[0] - 1        
            # y_index_adjusted = index_data[1] - 1

            # Write temperature data to CSV for each frame
            # with open(csv_filename, mode='a', newline='') as file:
            #     writer = csv.writer(file)
            #     row_data = [current_time]+list(heated_nodes_data.flatten())+[x_index_adjusted, y_index_adjusted]
            #     writer.writerow(row_data)
            # print("Temp Array", heated_nodes_data)
            # print("Avg Temp: ", avg_temp)
            # print("Max Temp: ", max_temp)
            # print("STD DEV TEMP", std_dev_temp)
            
          
            ####################### WRITE DATA TO NODE ##########################
            # message.data = np.reshape(temp_array,[len(x_values)*len(y_values)])
            # pub.publish(message)
            # Assuming 'data_transform' is your thermal image
            # Normalize the data_transform array to 0-255

            # Assuming 'raw_image', 'raw_data', 'transformed_data', 'transformed_image' are your data
            

            # data_f = ktof(data_cop)
            # data_1_f = ktof(data_1_cop)


            # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # img_gray_1 = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)

            # raw_image_message = bridge.cv2_to_imgmsg(img_gray)
            # raw_data_message = Float32MultiArray(data=data_f.flatten())
            # transformed_data_message = Float32MultiArray(data=data_transform_f.flatten())
            # transformed_image_message = bridge.cv2_to_imgmsg(img_transform)

            # rospy.loginfo("Publishing data...")
            # pub_raw_image.publish(raw_image_message)
            # pub_raw_data.publish(raw_data_message)
            # pub_transformed_data.publish(transformed_data_message)
            # pub_transformed_image.publish(transformed_image_message)

            #############################################################
            message = Float32MultiArray()
            message.data = np.reshape(temp_array,[len(x_values)*len(y_values)])
            pub_temp_array.publish(message)
            #############################################################

            data_transform_normalized = cv2.normalize(data_transform_c_f, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

            # Apply the colormap
            # img_transform = cv2.applyColorMap(data_transform_normalized, cv2.COLORMAP_JET)

            display_temperature(img_transform, minVal, minLoc, (255, 0, 0))
            display_temperature(img_transform, maxVal, maxLoc, (0, 0, 255))

            display_temperature(img_transform_1, minVal_1, minLoc_1, (255, 0, 0))
            display_temperature(img_transform_1, maxVal_1, maxLoc_1, (0, 0, 255))

            display_temperature(img_transform_c, minVal_c, minLoc_c, (255, 0, 0))
            display_temperature(img_transform_c, maxVal_c, maxLoc_c, (0, 0, 255)) 




            ############################ SHOW DATA ################################
            # cv2.imshow('Lepton Radiometry', img_transform)
            # cv2.imshow('Lepton Radiometry_1', img_transform_1)
            cv2.imshow('Lepton Radiometry_C', img_transform_c)
            cv2.waitKey(1)

          cv2.destroyAllWindows()
      finally:
        libuvc.uvc_stop_streaming(devh)

      print("done")
    finally:
      libuvc.uvc_unref_device(devs[0])
      libuvc.uvc_unref_device(devs[1])
  finally:
    libuvc.uvc_exit(ctx)

if __name__ == '__main__':
  main()

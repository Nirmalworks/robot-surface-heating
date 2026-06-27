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
  from Queue import Queue
import platform
from datetime import datetime
import csv

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

  # data = np.fromiter(
  #   frame.contents.data, dtype=np.dtype(np.uint8), count=frame.contents.data_bytes
  # ).reshape(
  #   frame.contents.height, frame.contents.width, 2
  # ) # copy

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


def main():
  ctx = POINTER(uvc_context)()
  dev = POINTER(uvc_device)()
  devh = POINTER(uvc_device_handle)()
  ctrl = uvc_stream_ctrl()

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
      x_start = 0
      x_end = 1000
      y_start = 0
      y_end = 1000
      nodes = 5
      x_spacing = x_end/(nodes+1)  # Adjust this to change the spacing between points
      y_spacing = y_end/(nodes+1)  # Adjust this to change the spacing between points

      # Create equally spaced points along X and Y axes
      x_values = np.arange(x_start, x_end, x_spacing)
      y_values = np.arange(y_start, y_end, y_spacing)
      x_values = x_values[1:]
      y_values = y_values[1:]
      # Create the mesh grid using np.meshgrid
      X, Y = np.meshgrid(x_values, y_values)
      num_rows, num_columns = X.shape
      print(x_values)
      print(y_values)


      # Bottom Right Corner 
      # Bottom Right Corner 
      # (577, 354)
      # Top Right Corner 
      #(462, 270)
      # Top Left Corner 
      # (165, 254)
      # Bottom Left Corner 
      # (50, 330)
      y_offset = 0

      # Top Left, Top Right, Bot Left, Bot Right
      orig_pts = np.float32([[165, 254+y_offset], [462, 270+y_offset], [50, 330+y_offset],
                       [577, 354+y_offset]])
      dest_pts = np.float32([[0, 0], [1000, 0], [0, 1000], [1000, 1000]])

      temp_array = np.zeros((5,5))
      # Generate a timestamp
      timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")

      # Define the CSV file name
      file_name = f"array_{timestamp}.csv"
      start_time = time.time()


      try:
        # Write the NumPy array to the CSV file
        with open(file_name, mode='w', newline='') as file:
          writer = csv.writer(file)
          writer.writerows(temp_array)
          while True:
            data = q.get(True, 500)
            if data is None:
              break
            data = cv2.resize(data[:,:], (640, 480))

            data_cop = data.copy()
            img = raw_to_8bit(data)
            img_lines = img.copy()

            cv2.line(img_lines, tuple(orig_pts[0]), tuple(orig_pts[1]), (255,0,0), 2)
            cv2.line(img_lines, tuple(orig_pts[1]), tuple(orig_pts[3]), (255,0,0), 2)
            cv2.line(img_lines, tuple(orig_pts[3]), tuple(orig_pts[2]), (255,0,0), 2)
            cv2.line(img_lines, tuple(orig_pts[2]), tuple(orig_pts[0]), (255,0,0), 2)
            # Get perspective transform M
            M = cv2.getPerspectiveTransform(orig_pts, dest_pts)
            # warp image with M
            img_transform = cv2.warpPerspective(img, M, (1000, 1000))
            data_transform = cv2.warpPerspective(data_cop, M, (1000, 1000))

            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data_transform)
            #print(minLoc)
            #rospy.loginfo(maxVal)
            #pub.publish(maxVal)

            # Loop through each X and Y point using nested loops
            for i in range(num_rows):
              for j in range(num_columns):
                  x_point = int(X[i, j])
                  y_point = int(Y[i, j])
                  #print(x_point)
                  #print(y_point)
                  #print(np.shape(data_transform))
                  val = data_transform[x_point, y_point]
                  temp_array[i,j] = ktof(val)
                  color = generate_color(val)
                  display_temperature(img_transform, val, (y_point,x_point), color)

            #print(temp_array)
            # Write to a file
            elapsed_time = time.time() - start_time
            #print(elapsed_time)
            if elapsed_time > 0.1:
              start_time = time.time()
              curr_time = datetime.now()
              formatted_time = curr_time.strftime('%M:%S.%f')

              
              #timestamp = time.strftime('%Y-%m-%d %H:%M:%S.%f')
              #print(formatted_time)
              writer.writerow([])
              writer.writerow(formatted_time)
              writer.writerows(temp_array)


            display_temperature(img_transform, minVal, minLoc, (255, 0, 0))
            display_temperature(img_transform, maxVal, maxLoc, (0, 0, 255))



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

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
  val = ktoc(val_k)
  cv2.putText(img,"{0:.1f} degC".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
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
      x_start = 400
      x_end = 480
      y_start = 400
      y_end = 640
      x_spacing = 20  # Adjust this to change the spacing between points
      y_spacing = 20  # Adjust this to change the spacing between points

      # Create equally spaced points along X and Y axes
      x_values = np.arange(x_start, x_end + 1, x_spacing)
      y_values = np.arange(y_start, y_end + 1, y_spacing)

      # Create the mesh grid using np.meshgrid
      X, Y = np.meshgrid(x_values, y_values)

      # Bottom Right Corner 
      # (502, 378)
      # Top Right Corner 
      # (450, 265)
      # Top Left Corner 
      # (185, 254)
      # Bottom Left Corner 
      # (78, 354)
      orig_pts = np.float32([[185, 254], [450, 265], [78, 354],
                       [502, 378]])
      dest_pts = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])


      try:
        while True:
          data = q.get(True, 500)
          if data is None:
            break
          data = cv2.resize(data[:,:], (640, 480))

          img = raw_to_8bit(data)
          img_lines = img.copy()

          #cv2.line(img_lines, tuple(orig_pts[0]), tuple(orig_pts[1]), (255,0,0), 2)
          #cv2.line(img_lines, tuple(orig_pts[1]), tuple(orig_pts[3]), (255,0,0), 2)
          #cv2.line(img_lines, tuple(orig_pts[3]), tuple(orig_pts[2]), (255,0,0), 2)
          #cv2.line(img_lines, tuple(orig_pts[2]), tuple(orig_pts[0]), (255,0,0), 2)
          # Get perspective transform M
          M = cv2.getPerspectiveTransform(orig_pts, dest_pts)
          # warp image with M
          img_transform = cv2.warpPerspective(img_lines, M, (640, 480))

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

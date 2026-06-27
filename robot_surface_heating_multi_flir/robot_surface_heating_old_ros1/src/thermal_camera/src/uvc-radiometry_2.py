#!/usr/bin/env python
# -*- coding: utf-8 -*-

from uvctypes import *
import time
import cv2
import numpy as np
try:
  from queue import Queue
except ImportError:
  from queue import Queue
import platform
import struct
import csv

import signal
import sys

CAMERA_COUNT = 4
BUF_SIZE = 2
qs = [Queue(BUF_SIZE) for _ in range(CAMERA_COUNT)]
global_q_idx = 0

def make_callback(index, queue):
  def callback(frame, userptr):
    if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
      return
    array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
    data = np.frombuffer(array_pointer.contents, dtype=np.uint16).reshape(
      frame.contents.height, frame.contents.width
    )
    if not queue.full():
      queue.put(data)
  return CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(callback)

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
  cv2.putText(img,"{0:.1f} degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
  x, y = loc
  cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
  cv2.line(img, (x, y - 2), (x, y + 2), color, 1)

global cur_img
cur_img = None

# global minVal
# minVal = 0
# global minLoc
# minLoc = 0

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        # display_temperature(cur_img, minVal, minLoc, (255, 0, 0))
        x = int(x/4)
        y = int(y/4)
        global cur_img
        print(cur_img[y,x])
        print(ktof(cur_img[y,x]))

for i in range(CAMERA_COUNT):
  cv2.namedWindow(f'Lepton Radiometry {i}')
  cv2.setMouseCallback(f'Lepton Radiometry {i}', click_event)
  cv2.waitKey(1) 

def main():
  ctx = POINTER(uvc_context)()
  dev = POINTER(uvc_device)()
  devs = POINTER(POINTER(uvc_device))()
  devh = (POINTER(uvc_device_handle) * CAMERA_COUNT)()
  ctrls = [uvc_stream_ctrl() for _ in range(CAMERA_COUNT)]
  res = libuvc.uvc_init(byref(ctx), 0)
  callbacks = [make_callback(i, qs[i]) for i in range(CAMERA_COUNT)]

  if res < 0:
    print("uvc_init error")
    exit(1)

  try:
    # find the desired amount of cameras
    res = libuvc.uvc_find_devices(ctx, byref(devs), PT_USB_VID, PT_USB_PID, 0)
    print(res)
    i = 0
    while devs[i]:
        print(devs.contents[i])
        i += 1
    print('Number of devices: {}'.format(i))
    if res < 0:
      print("uvc_find_device error")
      print(res)
      exit(1)

    try:
      # open each camera
      for i in range(CAMERA_COUNT):
        res = libuvc.uvc_open(devs[i], byref(devh[i]))
        if res < 0:
          print(f"uvc_open {i} error")
          exit(1)
        print_device_info(devh[i])
        print_device_formats(devh[i])

      # configure stream format
      for i in range(CAMERA_COUNT):
        frame_format = uvc_get_frame_formats_by_guid(devh[i], VS_FMT_GUID_Y16)
        if len(frame_format) == 0:
          print(f"device {i} does not support Y16")
          exit(1)
        libuvc.uvc_get_stream_ctrl_format_size(devh[i], byref(ctrls[i]), UVC_FRAME_FORMAT_Y16,
          frame_format[0].wWidth, frame_format[0].wHeight, int(1e7 / frame_format[0].dwDefaultFrameInterval)
        )

      # start streaming for each camera
      for i in range(CAMERA_COUNT):
        res = libuvc.uvc_start_streaming(devh[i], byref(ctrls[i]), callbacks[i], None, 0)
        if res < 0:
          print("uvc_start_streaming {0} failed: {0}".format(i, res))
          exit(1)
        time.sleep(0.5)

      # configure graceful exit signal handler
      stop_flag = False
      def signal_handler(sig, frame):
          nonlocal stop_flag
          print("\nSIGINT caught. Cleaning up...")
          stop_flag = True
      signal.signal(signal.SIGINT, signal_handler)

      try:
        while not stop_flag:
          # get camera captures stored in queues
          datas = []
          for i in range(CAMERA_COUNT):
            try:
              data = qs[i].get(timeout=2)
            except:
              print(f"[Main Loop] Timed out waiting for camera {i}")
              data = None
            datas.append(data)      
          if any(datas[i] is None for i in range(CAMERA_COUNT)):
            # print(0, datas[0], 1, datas[1], 2, datas[2])
            # break
            continue

          # upscale and display camera feeds
          timestamp = time.time()
          for i in range(CAMERA_COUNT):
            data = cv2.resize(datas[i][:,:], (640, 480))
            # data = datas
            minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)
            img = raw_to_8bit(data)
            display_temperature(img, minVal, minLoc, (255, 0, 0))
            display_temperature(img, maxVal, maxLoc, (0, 0, 255))
            global cur_img
            cur_img = datas[i].copy()
            # cur_img = img.copy()
            cv2.imshow(f'Lepton Radiometry {i}', img)
          cv2.waitKey(1)

        # cleanup
        cv2.destroyAllWindows()
      finally:
        for i in range(CAMERA_COUNT):
          if devh[i]:
            libuvc.uvc_stop_streaming(devh[i])
    finally:
      for i in range(CAMERA_COUNT):
        if devh[i]:
          libuvc.uvc_unref_device(devs[i])
  finally:
    libuvc.uvc_exit(ctx)

if __name__ == '__main__':
  main()


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

import os
import rospkg
import glob

import sys

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
  cv2.putText(img,"{0:.1f} degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
  x, y = loc
  cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
  cv2.line(img, (x, y - 2), (x, y + 2), color, 1)

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)

cv2.namedWindow('Lepton Radiometry')
cv2.setMouseCallback('Lepton Radiometry', click_event)

def prepare_thermal_for_detection(thermal_bgr):
    gray = cv2.cvtColor(thermal_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(gray)

    # img_blur  = cv2.GaussianBlur(img_enhanced, (5,5), 0)
    img_inv   = 255 - img_enhanced
    _, img_thresh = cv2.threshold(img_inv, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return img_inv, img_thresh

# Simple example: check if rows of corners are mostly linear
def is_linearity_ok(corners, threshold=1.5):
    for i in range(0, len(corners), pattern_cols):
        row = corners[i:i+pattern_cols]
        [vx, vy, x0, y0] = cv2.fitLine(np.array(row), cv2.DIST_L2, 0, 0.01, 0.01)
        errors = [abs((vy*(x - x0) - vx*(y - y0))) for x, y in row]
        if np.mean(errors) > threshold:
            return False
    return True

def capture_calibration_images(img, step, pattern_size, output_dir='calibration_images'):
    # output_dir = '/'.join([rospkg.RosPack().get_path('thermal_camera'), output_dir])
    output_dir = '/'.join(["/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera", output_dir])
    os.makedirs(output_dir, exist_ok=True)
    
    positions = [
        "1. Frontal center",
        "2. Tilted left", 
        "3. Tilted right",
        "4. Tilted up",
        "5. Tilted down",
        "6. Top-left corner",
        "7. Top-right corner", 
        "8. Bottom-left corner",
        "9. Bottom-right corner",
        "10. Rotated clockwise",
        "11. Rotated counter-clockwise",
        "12. Close (60% frame)",
        "13. Far (30% frame)",
        "14. Left side",
        "15. Right side"
    ]
    
    # pattern_size = (6, 4)
    # pattern_size = (10, 7)
    
    # circle pattern
    # pattern_size = (4,11)
    # pattern_size = (4,8)

    # for position in positions:
    print(f"\nPosition: {positions[step]}")
    #     input("Arrange board and press Enter...")
      
    # Capture image
    thermal_img = img
    # thermal_8bit = raw_to_8bit(thermal_img)
    # thermal_8bit = cv2.normalize(thermal_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    # thermal_inv, _ = prepare_thermal_for_detection(thermal_img)
    # thermal_inv = 255 - thermal_8bit
    gray        = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2GRAY)

    # # Smooth out thermal sensor noise
    # blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # # Normalize intensities globally (thermal contrast is usually low)
    # normalized = cv2.normalize(blurred, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)


    # thermal_inv = 255 - normalized
    thermal_inv = 255 - gray
    
    # Check detection
    ret, corners = cv2.findChessboardCorners(
        thermal_inv, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH
    )

    # params = cv2.SimpleBlobDetector_Params()
    # params.filterByArea = True
    # params.minArea = 5
    # params.maxArea = 400
    # params.filterByCircularity = False
    # params.filterByInertia = False
    # params.filterByConvexity = False

    # params.minArea = 30
    # params.maxArea = 300
    # params.minThreshold = 10
    # params.maxThreshold = 255
    # params.thresholdStep = 10
    # # params.filterByCircularity = True
    # # params.minCircularity = 0.2

    # detector = cv2.SimpleBlobDetector_create(params)

    # # Step 1: Contrast enhancement
    # img_eq = cv2.equalizeHist(thermal_inv)

    # # Step 2: Gaussian blur (to reduce noise)
    # blurred = cv2.GaussianBlur(img_eq, (3, 3), 0)

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,6))
    # gray_eq = clahe.apply(img_eq)

    # laplacian = cv2.Laplacian(gray_eq, cv2.CV_8U)

    # # Step 3: Adaptive thresholding (can help with uneven lighting)
    # binary = cv2.adaptiveThreshold(
    #     blurred, 255, 
    #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #     cv2.THRESH_BINARY, 
    #     11, 2
    # )

    # # Optional: Morphological ops to refine blobs
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # # kernel = np.ones((3, 3))
    # binary = cv2.morphologyEx(laplacian, cv2.MORPH_OPEN, kernel)
    # binary = laplacian
    # keypoints = detector.detect(binary)
    # # keypoints = detector.detect(thermal_inv)
    # im_with_keypoints = cv2.drawKeypoints(
    #     binary, keypoints, np.array([]), (0, 0, 255),
    #     # thermal_inv, keypoints, np.array([]), (0, 0, 255),
    #     cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    # )
    # cv2.imshow("Blobs", im_with_keypoints)
    # cv2.waitKey(1)

    # print(f"Detected {len(keypoints)} blobs")

    # ret, corners = cv2.findCirclesGrid(
    #     thermal_inv, pattern_size,
    #     flags=cv2.CALIB_CB_ASYMMETRIC_GRID + cv2.CALIB_CB_CLUSTERING,
    #     # flags=cv2.CALIB_CB_ASYMMETRIC_GRID,
    #     # flags=cv2.CALIB_CB_CLUSTERING,
    #     # blobDetector=detector
    # )
    
    if ret:
        # Save image
        filename = f'thermal_calib_{step:03d}.png'
        cv2.imwrite(os.path.join(output_dir, filename), thermal_inv)
        print(f"✓ Captured {filename}")
        
        # Show preview
        # display_data = cv2.resize(thermal_inv[:,:], (640, 480))
        display_data = cv2.resize(thermal_inv[:,:], (640, 480))
        img_preview = cv2.cvtColor(display_data, cv2.COLOR_GRAY2BGR)
        corners_display = corners * 4 # 160x120 -> 640x480
        cv2.drawChessboardCorners(img_preview, pattern_size, corners_display, ret)
        cv2.imshow('Captured', img_preview)
        cv2.waitKey(1000)
    else:
        print("✗ Detection failed - try again")
        # display_data = cv2.resize(thermal_inv[:,:], (640, 480))
        display_data = cv2.resize(thermal_inv[:,:], (640, 480))
        cv2.imshow('Failed', display_data)
        cv2.waitKey(1000)
    
    print(f"\nTotal captured: {step} images")
    return True if ret else False, corners if ret else None

def calibrate_camera_from_images(pattern_size, calibration_dir='calibration_images'):
    """
    Run calibration on collected images
    """
    # square_size = 38.0  # mm
    # square_size = 34.5  # mm
    # square_size = 75.3  # mm
    # square_size = 61.6 # mm
    square_size = 49.97 # mm
    # pattern_size = (6, 4)

    # square_size = 33.7  # mm
    # pattern_size = (10, 7)

    # circle pattern
    # pattern_size = (11,3)
    
    # Termination criteria
    # criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    
    # Prepare object points
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1,2)
    objp *= square_size

    # # asymmetrical circular pattern
    # spacing = 19 # in mm
    # # spacing = 7.93 # in mm
    # objp[:, 0] *= spacing  # x scale (e.g., 20 mm)
    # objp[:, 1] *= spacing * np.sqrt(3)/2  # y scale for asymmetrical layout (triangular grid)
    
    objpoints = []
    imgpoints = []
    
    # calibration_dir = '/'.join([rospkg.RosPack().get_path('thermal_camera'), calibration_dir])
    calibration_dir = '/'.join(["/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera", calibration_dir])
    images = glob.glob(f'{calibration_dir}/*.png')
    
    print(f"\nProcessing {len(images)} images for calibration...")
    
    for fname in images:
        img = cv2.imread(fname, 0)
        
        ret, corners = cv2.findChessboardCorners(
            img, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
        )
        
        if ret:
            objpoints.append(objp)
            # corners2 = cv2.cornerSubPix(img, corners, (11,11), (-1,-1), criteria)
            corners2 = cv2.cornerSubPix(img, corners, (5,5), (-1,-1), criteria)
            imgpoints.append(corners2)
    
    # Calibrate
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, img.shape[::-1], None, None,
    )
    
    print(f"Calibration RMS error: {ret:.3f} pixels")
    
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            mtx,
            dist
        )
        # compute L2 error between detected and reprojected
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    mean_error = total_error / len(objpoints)
    print(f"Mean reprojection error: {mean_error:.3f} pixels")

    # Save calibration
    np.savez('/'.join([calibration_dir,'thermal_intrinsics.npz']),
             mtx=mtx,
             dist=dist,
             rvecs=rvecs,
             tvecs=tvecs,
             rms=ret,
             mean_error=mean_error)

    return mtx, dist, ret

def mean_error_check(mtx, dist, objpoints, imgpoints, rvecs, tvecs):
    total_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            objpoints[i],
            rvecs[i],
            tvecs[i],
            mtx,
            dist
        )
        # compute L2 error between detected and reprojected
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    mean_error = total_error / len(objpoints)
    print(f"Mean reprojection error: {mean_error:.3f} pixels")
    return mean_error

def intrinsic_visual_check(mtx, dist, raw):
  # assume objpoints, imgpoints, mtx, dist, rvecs, tvecs are from calibrateCamera
  h, w = raw.shape[:2]
  newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
  undistorted = cv2.undistort(raw_to_8bit(raw), mtx, dist, None, newcameramtx)


  # stack side by side
  comparison = np.hstack([raw_to_8bit(raw), undistorted])

  comparison = cv2.resize(
        comparison,
        (640, 480),
        interpolation=cv2.INTER_LINEAR
    )

  cv2.namedWindow('Comparison', cv2.WINDOW_NORMAL)
  cv2.imshow('Original (left) vs Undistorted (right)', comparison)

  # wait indefinitely for a keypress in the window
  cv2.waitKey(0)
  cv2.destroyWindow('Comparison')

def main():
  ctx = POINTER(uvc_context)()
  dev = POINTER(uvc_device)()
  devs = POINTER(POINTER(uvc_device))()
  # devh = (POINTER(uvc_device_handle) * 2)()
  # devh = (POINTER(uvc_device_handle) * 4)()
  devh = (POINTER(uvc_device_handle) * 1)()
  ctrl = uvc_stream_ctrl()
  # SN = b'\xbe\x95O\x01\x00\x00\x00\x00'
  # SN_string = SN.decode('utf-8', 'ignore')
  res = libuvc.uvc_init(byref(ctx), 0)

  if res < 0:
    print("uvc_init error")
    exit(1)

  try:
    # res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
    res = libuvc.uvc_find_devices(ctx, byref(devs), PT_USB_VID, PT_USB_PID, 0)
    print(res)
    i = 0
    while devs[i]:
        print(devs.contents[i])
        i += 1
    print('Number of devices: {}'.format(i))
    # input("Press Enter to continue...")
    if res < 0:
      print("uvc_find_device error")
      print(res)
      exit(1)

  # ctx = POINTER(uvc_context)()
  # dev = POINTER(uvc_device)()
  # devs = POINTER(uvc_device)()
  # devh = POINTER(uvc_device_handle)()
  # ctrl = uvc_stream_ctrl()
  # # SN = b'\xbe\x95O\x01\x00\x00\x00\x00'
  # # SN_string = SN.decode('utf-8', 'ignore')
  # res = libuvc.uvc_init(byref(ctx), 0)
  
  # if res < 0:
  #   print("uvc_init error")
  #   exit(1)

  # try:
  #   uvc_dev_array = uvc_device * 5
  #   arr = uvc_dev_array(uvc_device(), uvc_device(), uvc_device(), uvc_device(), uvc_device())
  #   devs_2 = POINTER(uvc_device)(arr)
  #   # devs_list = list(cast(devs_ptr, POINTER(uvc_device * 5)).contents)
  #   res_arrs = libuvc.uvc_get_device_list(ctx, byref(devs_2))
  #   # devs_ptr = POINTER(devs.contents.ctx)
  #   print("res", res_arrs, "uvc devices", )
  #   # res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
  #   res = libuvc.uvc_find_devices(ctx, byref(devs_2), PT_USB_VID, PT_USB_PID, 0)
  #   print(res)
  #   i = 0
  #   # while devs[i]:
  #   #     i += 1
  #   print('Number of devices: {}'.format(i))
  #   # input("Press Enter to continue...")
  #   if res < 0:
  #     print("uvc_find_device error")
  #     print(res)
  #     exit(1)

    # assign which camera to use
    target_cam = 0
    dev = devs[target_cam]
    devh = devh[target_cam]

    try:
      res = libuvc.uvc_open(dev, byref(devh))
      
      if res < 0:
        print("uvc_open error")
        exit(1)

      print("device opened!")

      print_device_info(devh)
      print_device_formats(devh)

      # input("Press Enter to continue...")

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
      # Logging function

      # # Open the CSV file for writing
      # csvfile = open('/home/cam/ND_ws/heat_ws/robot_surface_heating/src/data/log.csv', 'w', newline='')
      # # Create a CSV writer
      # writer = csv.writer(csvfile)
      # # Write the header row
      # writer.writerow(['Timestamp', 'Max Temperature'])
      step = 0
      corners_arr = []
      img_dir = "lepton4"
      pattern_size = (6,4)
      # pattern_size = (4,8)
      print("pattern size is", pattern_size)
      try:
        while True:
          if step > 12:
          # if step > 0:
            print("camera intrinsic calibration")
            mtx, dist, res = calibrate_camera_from_images(pattern_size, img_dir)
            print(mtx, dist, res)

            data = q.get(True, 500)
            if data is None:
              break
            # data = cv2.resize(data[:,:], (640, 480))
            # minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)
            # print(minLoc)
            
            # loc = (200,200)
            # val = data[loc[0],loc[1]]
            # img = raw_to_8bit(data)

            intrinsic_visual_check(mtx, dist, data)
            break

          data = q.get(True, 500)
          if data is None:
            break
          display_data = cv2.resize(data[:,:], (640, 480))
          minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)
          # print(minLoc)
          
          # loc = (200,200)
          # val = display_data[loc[0],loc[1]]
          display_img = raw_to_8bit(display_data)
          # # img = data
          # #display_temperature(img, val, loc, (0, 255, 0))
          display_temperature(display_img, minVal, minLoc, (255, 0, 0))
          display_temperature(display_img, maxVal, maxLoc, (0, 0, 255))

          # Get the current timestamp
          timestamp = time.time()

          # # Write a row to the CSV file
          # writer.writerow([timestamp, ktof(maxVal)])

          # log max val data to a csv file 


          cv2.imshow('Lepton Radiometry', display_img)
          cv2.waitKey(1)

          img = raw_to_8bit(data)
          status, corners = capture_calibration_images(img, step, pattern_size, output_dir=img_dir)
          if status:
            user_resp = input("Is this image good? (y/n): ")
            if user_resp == "y":
              corners_arr.append(corners)
              step += 1
            # pass

        cv2.destroyAllWindows()
      finally:
        libuvc.uvc_stop_streaming(devh)

      print("done")
    finally:
      libuvc.uvc_unref_device(dev)
      # csvfile.close()
  finally:
    libuvc.uvc_exit(ctx)

if __name__ == '__main__':
    # run this to calculate intrinsic parameters from an existing directory of images
    # calibrate_camera_from_images('refactored_test4')
    
    # run this to capture new images and then calculate intrinsic parameters
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UR10 Extrinsic Calibration - Integrated with your thermal camera system
Based on your intrinsic calibration code
"""

from thermal_camera.uvctypes import *
import time
import cv2
import numpy as np
import pickle
from rclpy.node import Node
import json
from datetime import datetime
from scipy.spatial.transform import Rotation as R
try:
    from queue import Queue
except ImportError:
    from queue import Queue
import platform
import struct
import os
# import rospkg
from rclpy.impl.implementation_singleton import rclpy_implementation as _rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.signals import SignalHandlerGuardCondition
from rclpy.utilities import timeout_sec_to_nsec
import rclpy
from moveit_msgs.srv import GetPositionFK, GetPositionIK
from sensor_msgs.msg import JointState
from std_msgs.msg import Header
from geometry_msgs.msg import Pose, Point, Quaternion
from moveit_msgs.msg import (
    RobotState,
    RobotTrajectory,
    MoveItErrorCodes,
    Constraints,
    JointConstraint,
    PlanningScene,
    CollisionObject
)
from typing import Union
from ament_index_python.packages import get_package_share_directory

def wait_for_message(
    msg_type,
    node: 'Node',
    topic: str,
    *,
    qos_profile: Union[QoSProfile, int] = 1,
    time_to_wait=-1
):
    """
    Wait for the next incoming message.

    :param msg_type: message type
    :param node: node to initialize the subscription on
    :param topic: topic name to wait for message
    :param qos_profile: QoS profile to use for the subscription
    :param time_to_wait: seconds to wait before returning
    :returns: (True, msg) if a message was successfully received, (False, None) if message
        could not be obtained or shutdown was triggered asynchronously on the context.
    """
    context = node.context
    wait_set = _rclpy.WaitSet(1, 1, 0, 0, 0, 0, context.handle)
    wait_set.clear_entities()

    sub = node.create_subscription(msg_type, topic, lambda _: None, qos_profile=qos_profile)
    try:
        wait_set.add_subscription(sub.handle)
        sigint_gc = SignalHandlerGuardCondition(context=context)
        wait_set.add_guard_condition(sigint_gc.handle)

        timeout_nsec = timeout_sec_to_nsec(time_to_wait)
        wait_set.wait(timeout_nsec)

        subs_ready = wait_set.get_ready_entities('subscription')
        guards_ready = wait_set.get_ready_entities('guard_condition')

        if guards_ready:
            if sigint_gc.handle.pointer in guards_ready:
                return False, None

        if subs_ready:
            if sub.handle.pointer in subs_ready:
                msg_info = sub.handle.take_message(sub.msg_type, sub.raw)
                if msg_info is not None:
                    return True, msg_info[0]
    finally:
        node.destroy_subscription(sub)

    return False, None

# Your existing camera setup
BUF_SIZE = 2
q = Queue(BUF_SIZE)

def py_frame_callback(frame, userptr):
    array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
    data = np.frombuffer(
        array_pointer.contents, dtype=np.dtype(np.uint16)
    ).reshape(
        frame.contents.height, frame.contents.width
    )
    
    if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
        return
    
    if not q.full():
        q.put(data)

PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)

# Your existing utility functions
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

def prepare_thermal_for_detection(thermal_bgr):
    """Your existing thermal preprocessing"""
    gray = cv2.cvtColor(thermal_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(gray)
    img_inv = 255 - img_enhanced
    _, img_thresh = cv2.threshold(img_inv, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return img_inv, img_thresh

class UR10ExtrinsicCalibration(Node):
    timeout_sec_ = 5.0

    move_group_name_ = "ur10e"
    namespace_ = ""

    joint_state_topic_ = "/arm_1/joint_states"
    plan_srv_name_ = "plan_kinematic_path"
    ik_srv_name_ = "compute_ik"
    fk_srv_name_ = "compute_fk"
    execute_action_name_ = "execute_trajectory"
    get_planning_scene_srv_name = "get_planning_scene"
    apply_planning_scene_srv_name = "apply_planning_scene"

    # # large bulb -> 97mm + 9.46mm to filament
    # eef_tool_offset = np.array([0., 0., 0.10646])  # in meters

    # # large bulb 2 -> 97mm + 9.2mm to filament
    # eef_tool_offset = np.array([0., 0., 0.1062])  # in meters
    
    # # large bulb 2 -> 97mm + 9.5mm to filament
    # eef_tool_offset = np.array([0., 0., 0.1065])  # in meters

    # large bulb 2 -> 97mm + 3.7mm to filament
    # eef_tool_offset = np.array([0., 0., 0.1007])  # in meters

    # large bulb 2 -> 97mm + 12.5mm to filament
    # eef_tool_offset = np.array([0., 0., 0.097 + 0.0125])  # in meters

    # ceramic bulb holder
    eef_tool_offset = np.array([0., 0., 0.138])  # in meters

    # small bulb -> 97mm + 1.7mm to filament
    # eef_tool_offset = np.array([0., 0., 0.0987])  # in meters

    num_contours = 1

    base_ = "base_link"
    end_effector_ = "tool0"
    def __init__(self):
        # super().__init__("UR10_Extrinsic_Calibration")
        super().__init__("pnp_extrinsic")

        # Declare and read YAML config parameters
        self.declare_parameters(
            namespace='',
            parameters=[
                ('camera_count', 1),
                ('intrinsics_dir_prefix', ""),
                ('extrinsics_dir_prefix', ""),
                ('calibration_backup_file', ""),
                ('extrinsics_file', "")
            ]
        )

        self.camera_count = self.get_parameter('camera_count').get_parameter_value().integer_value
        self.get_logger().info(f"camera_count {self.camera_count}")
        self.intrinsics_dir_pfx = self.get_parameter('intrinsics_dir_prefix').get_parameter_value().string_value
        self.get_logger().info(f"intrinsics_dir_prefix {self.intrinsics_dir_pfx}")
        self.extrinsics_dir_pfx = self.get_parameter('extrinsics_dir_prefix').get_parameter_value().string_value
        self.get_logger().info(f"extrinsics_dir_prefix {self.extrinsics_dir_pfx}")
        self.calibration_backup_file = self.get_parameter('calibration_backup_file').get_parameter_value().string_value
        self.get_logger().info(f"calibration_backup_file {self.calibration_backup_file}")
        self.extrinsics_file = self.get_parameter('extrinsics_file').get_parameter_value().string_value
        self.get_logger().info(f"extrinsics_file {self.extrinsics_file}")

        self.ik_client_ = self.create_client(GetPositionIK, self.ik_srv_name_)
        if not self.ik_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("IK service not available.")
            exit(1)

        self.fk_client_ = self.create_client(GetPositionFK, self.fk_srv_name_)
        if not self.fk_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("FK service not available.")
            exit(1)

        # Load your intrinsic calibration results
        self.load_intrinsic_parameters()
        
        # Calibration data storage
        self.pkg_path = '/'.join(get_package_share_directory('thermal_camera').split('/')[:-4]+['src','thermal_camera'])
        self.calibration_data = []
        self.centroid_point_data = [[] for _ in range(self.camera_count)]
        
        # Checkerboard parameters - MATCH YOUR INTRINSIC CALIBRATION
        self.CHECKERBOARD_SIZE = (6, 4)  # Same as your intrinsic calibration
        # self.SQUARE_SIZE = 0.0753  # 75.3mm converted to meters
        self.SQUARE_SIZE = 0.0616  # 61.6mm converted to meters
        
        # EEF to checkerboard offset - MEASURE THESE VALUES
        self.T_eef_to_board = self.define_eef_to_checkerboard_offset()
        
        # Create object points (same as your intrinsic calibration)
        self.objp = np.zeros((self.CHECKERBOARD_SIZE[0] * self.CHECKERBOARD_SIZE[1], 3), np.float32)
        self.objp[:,:2] = np.mgrid[0:self.CHECKERBOARD_SIZE[0], 0:self.CHECKERBOARD_SIZE[1]].T.reshape(-1,2)
        self.objp *= self.SQUARE_SIZE
        
        self.get_logger().info("UR10 Extrinsic Calibration initialized")
        self.get_logger().info(f"Checkerboard: {self.CHECKERBOARD_SIZE}, Square size: {self.SQUARE_SIZE*1000:.1f}mm")

    def get_joint_state(self):
        """Get current joint state (adapted from your robot interface)"""
        current_joint_state_set, current_joint_state = wait_for_message(
            JointState, self, self.joint_state_topic_, time_to_wait=1.0
        )
        if not current_joint_state_set:
            self.get_logger().error("Failed to get current joint state")
            return None
        
        return current_joint_state
    
    def get_fk(self) -> Pose | None:
        current_joint_state = self.get_joint_state()
        if current_joint_state is None:
            self.get_logger().error("Failed to get current joint state")
            return None

        current_robot_state = RobotState()
        current_robot_state.joint_state = current_joint_state

        request = GetPositionFK.Request()

        request.header.frame_id = self.base_
        request.header.stamp = self.get_clock().now().to_msg()

        request.fk_link_names.append(self.end_effector_)
        request.robot_state = current_robot_state

        future = self.fk_client_.call_async(request)

        rclpy.spin_until_future_complete(self, future)
        if future.result() is None:
            self.get_logger().error("Failed to get FK solution")
            return None

        response = future.result()
        if response.error_code.val != MoveItErrorCodes.SUCCESS:
            self.get_logger().error(
                f"Failed to get FK solution: {response.error_code.val}"
            )
            return None
        
        pose = response.pose_stamped[0].pose
        T = np.eye(4)
        T[:3, 3] = [pose.position.x, pose.position.y, pose.position.z]
        quat = [pose.orientation.x, pose.orientation.y, 
               pose.orientation.z, pose.orientation.w]
        T[:3, :3] = R.from_quat(quat).as_matrix()
        
        self.get_logger().info(f"EEF pose: [{pose.position.x:.3f}, {pose.position.y:.3f}, {pose.position.z:.3f}]")
        return T

    def load_intrinsic_parameters(self):
        """Load your intrinsic calibration results"""
        try:
            # Try to load from your ROS package structure
            # pkg_path = rospkg.RosPack().get_path('thermal_camera')
            pkg_path = "/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera"
            intrinsic_files = [
                os.path.join(pkg_path, f'calibration_data/intrinsic/{self.intrinsics_dir_pfx}0', 'thermal_intrinsics.npz'),
                os.path.join(pkg_path, f'calibration_data/intrinsic/{self.intrinsics_dir_pfx}1', 'thermal_intrinsics.npz'),
                os.path.join(pkg_path, f'calibration_data/intrinsic/{self.intrinsics_dir_pfx}2', 'thermal_intrinsics.npz'),
                os.path.join(pkg_path, f'calibration_data/intrinsic/{self.intrinsics_dir_pfx}3', 'thermal_intrinsics.npz'),
                # os.path.join(pkg_path, 'calibration_images', 'thermal_intrinsics.npz'),
                # 'thermal_intrinsics.npz'  # Current directory
            ]
            
            loaded = False
            self.camera_matrix_list = []
            self.dist_coeffs_list = []
            self.intrinsic_rms_error_list = []
            for filepath in intrinsic_files:
                if os.path.exists(filepath):
                    data = np.load(filepath)
                    self.camera_matrix_list.append(data['mtx'])
                    self.dist_coeffs_list.append(data['dist'])
                    self.intrinsic_rms_error_list.append(data['rms'])
                    
                    self.get_logger().info(f"✓ Loaded intrinsics from: {filepath}")
                    self.get_logger().info(f"  RMS error: {self.intrinsic_rms_error_list[-1]:.3f} pixels")
                    self.get_logger().info(f"  Camera matrix:\n{self.camera_matrix_list[-1]}")
                    self.get_logger().info(f"  Distortion coeffs: {self.dist_coeffs_list[-1].flatten()}")
                    # loaded = True
                    # break
            loaded = True
            
            if not loaded:
                raise FileNotFoundError("No intrinsic calibration file found")
                
        except Exception as e:
            self.get_logger().error(f"✗ ERROR loading intrinsics: {e}")
            self.get_logger().error("Run intrinsic calibration first!")
            raise

    def define_eef_to_checkerboard_offset(self):        
        x_offset = 0.00
        y_offset = 0.00   
        z_offset = 0.00  
        rx, ry, rz = 0, 0, 0  
        
        print(f"Translation: [{x_offset:+.3f}, {y_offset:+.3f}, {z_offset:+.3f}] meters")
        print(f"Rotation: [{rx}, {ry}, {rz}] degrees")
        
        T = np.eye(4)
        T[:3, 3] = [x_offset, y_offset, z_offset]
        T[:3, :3] = R.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
        return T

    def collect_point_centroid_in_thermal(self, thermal_datas):
        """Find hottest points of interest for thermal processing approach"""
        # cDictionary = []
        for data_idx in range(len(thermal_datas)):
            # Convert raw thermal to 8-bit RGB (your method)
            thermal_8bit_rgb = raw_to_8bit(thermal_datas[data_idx])
            
            # Convert to grayscale and invert (your method)
            gray = cv2.cvtColor(thermal_8bit_rgb, cv2.COLOR_BGR2GRAY)
            thermal_inv = 255 - gray

            # uninvert after applying thresholding
            _, thresh = cv2.threshold(thermal_inv, 127, 255, cv2.THRESH_BINARY)
            thresh = 255 - thresh

            assert thresh.dtype == np.uint8
            assert set(np.unique(thresh)).issubset({0, 255})

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            output = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            gray_upscaled = cv2.cvtColor(cv2.resize(gray, (640, 480)), cv2.COLOR_GRAY2BGR)
            thermal_inv_upscaled = cv2.cvtColor(cv2.resize(thermal_inv, (640, 480)), cv2.COLOR_GRAY2BGR)
            thresh_upscaled = cv2.resize(output, (640, 480))

            # Find the largest contour, assuming it's the hot spot
            if contours:
                largest = max(contours, key=cv2.contourArea)

                # Compute centroid
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    self.get_logger().info("centroid identified")
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # Draw the centroid on the image
                    cv2.circle(output, (cX, cY), 5, (0, 255, 0), -1)
                    cv2.circle(gray_upscaled, (cX*4, cY*4), 5, (0, 255, 0), -1)
                    cv2.circle(thermal_inv_upscaled, (cX*4, cY*4), 5, (0, 255, 0), -1)
                    cv2.circle(thresh_upscaled, (cX*4, cY*4), 5, (0, 255, 0), -1)
                else:
                    # Find coordinates of white pixels
                    ys, xs = np.where(thresh == 255)

                    self.get_logger().info(f"{len(thresh)}, {len(thresh[0])}")
                    self.get_logger().info(f"{len(ys)}, {len(xs)}")

                    # If none found, exit
                    if len(xs) == 0:
                        self.get_logger().warn("No white pixels found.")
                        return False, None, thermal_inv_upscaled
                    else:
                        self.get_logger().info("White pixels found")
                        # Compute centroid
                        cX = int(np.mean(xs))
                        cY = int(np.mean(ys))

                        cv2.circle(output, (cX, cY), 5, (0, 255, 0), -1)
                        cv2.circle(gray_upscaled, (cX*4, cY*4), 5, (0, 255, 0), -1)
                        cv2.circle(thermal_inv_upscaled, (cX*4, cY*4), 5, (0, 255, 0), -1)
                        cv2.circle(thresh_upscaled, (cX*4, cY*4), 5, (0, 255, 0), -1)
            
                cv2.imshow("Point Capture", thermal_inv_upscaled)
                # cv2.waitKey(1)

                self.get_logger().info("keep this point? [y/n]")   
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('y'):
                # usr_in = input("keep this point? [y/n]")
                # if usr_in == "y":

                    # store mapping of 2D projection and 3D pose
                    T_base_to_eef = self.get_fk()
                    world_pos = T_base_to_eef[:3, 3] + T_base_to_eef[:3, :3]@self.eef_tool_offset
                    world_T = np.eye(4)
                    world_T[:3,:3] = T_base_to_eef[:3,:3]
                    world_T[:3,3] = world_pos
                    self.centroid_point_data[data_idx].append({
                        'capture_id': len(self.centroid_point_data) + 1,
                        '2d_pt': np.array([np.float32(cX), np.float32(cY)]),
                        # '3d_pt': (T_base_to_eef[:3, 3] + T_base_to_eef[:3, :3]@self.eef_tool_offset),
                        '3d_pt': world_pos,
                        'T_3d_pt': world_T,
                        'timestamp': time.time()
                    })
                    # cDictionary.append({
                    #     'capture_id': len(self.centroid_point_data) + 1,
                    #     '2d_pt': np.array([np.float32(cX), np.float32(cY)]),
                    #     '3d_pt': (T_base_to_eef[:3, 3] + T_base_to_eef[:3, :3]@self.eef_tool_offset),
                    #     'timestamp': time.time()
                    # })



                    # self.get_logger().info("object point {}".format(cDictionary[-1]["3d_pt"]))
                    self.get_logger().info("object point {}".format(self.centroid_point_data[data_idx][-1]["3d_pt"]))
                    # self.get_logger().info("image point {}".format(cDictionary[-1]["2d_pt"]))
                    self.get_logger().info("image point {}".format(self.centroid_point_data[data_idx][-1]["2d_pt"]))
                
                    self.get_logger().info(f"✓ Pose {len( self.centroid_point_data[data_idx])} for Camera {data_idx} collected successfully!")
                    self.get_logger().info(f"  Total poses collected: {sum([len(arr) for arr in self.centroid_point_data])}")
                    
                    # Save image
                    # output_dir = "/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/extrinsic_calib_1/"
                    output_dir = '/'.join([self.pkg_path, f"calibration_data/extrinsic/{self.extrinsics_dir_pfx}{data_idx}"])
                    try:
                        if not os.path.isdir(output_dir):
                            os.mkdir(output_dir)
                    except:
                        self.get_logger().error(f"directory missing {output_dir}")
                    filename = f'thermal_calib_{len(self.centroid_point_data[data_idx]):03d}.png'
                    cv2.imwrite(os.path.join(output_dir, filename), thermal_datas[data_idx])
                    self.get_logger().info(f"✓ Captured thermal image {filename}")
                    filename = f'thermal_calib_capture_{len(self.centroid_point_data[data_idx]):03d}.png'
                    cv2.imwrite(os.path.join(output_dir, filename), thermal_inv_upscaled)
                    self.get_logger().info(f"✓ Captured point capture {filename}")

                    # Auto-save backup
                    self.save_calibration_backup(data_idx)

                    # self.get_logger().info("c to capture next image, n to end")   
                    # key = cv2.waitKey(0) & 0xFF
                    # if key == ord('c'):
                    #     continue
                
                self.get_logger().info(f"Pose dropped for Camera {data_idx}.")
            else:
                self.get_logger().info(f"Camera {data_idx} Contours not found.")
        self.get_logger().info("Total points:")
        for cam_idx in range(len(self.centroid_point_data)):
            self.get_logger().info(f"Camera {cam_idx}: {len(self.centroid_point_data[cam_idx])}")
        return True


    def ba_collect_point_centroid_in_thermal(self, thermal_datas):
        """Find hottest points of interest for thermal processing approach"""
        c_list = []
        for data_idx in range(len(thermal_datas)):
            # Convert raw thermal to 8-bit RGB (your method)
            thermal_8bit_rgb = raw_to_8bit(thermal_datas[data_idx])
            
            # Convert to grayscale and invert (your method)
            gray = cv2.cvtColor(thermal_8bit_rgb, cv2.COLOR_BGR2GRAY)
            thermal_inv = 255 - gray

            # uninvert after applying thresholding
            _, thresh = cv2.threshold(thermal_inv, 127, 255, cv2.THRESH_BINARY)
            thresh = 255 - thresh

            assert thresh.dtype == np.uint8
            assert set(np.unique(thresh)).issubset({0, 255})

            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            output = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            gray_upscaled = cv2.cvtColor(cv2.resize(gray, (640, 480)), cv2.COLOR_GRAY2BGR)
            thermal_inv_upscaled = cv2.cvtColor(cv2.resize(thermal_inv, (640, 480)), cv2.COLOR_GRAY2BGR)
            thresh_upscaled = cv2.resize(output, (640, 480))

            # Find the largest contour, assuming it's the hot spot
            if contours:
                largest = max(contours, key=cv2.contourArea)

                # Compute centroid
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    self.get_logger().info("centroid identified")
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    
                    # Draw the centroid on the image
                    cv2.circle(output, (cX, cY), 5, (0, 255, 0), -1)
                    cv2.circle(gray_upscaled, (cX*4, cY*4), 5, (0, 255, 0), -1)
                    cv2.circle(thermal_inv_upscaled, (cX*4, cY*4), 5, (0, 255, 0), -1)
                    cv2.circle(thresh_upscaled, (cX*4, cY*4), 5, (0, 255, 0), -1)
                else:
                    # Find coordinates of white pixels
                    ys, xs = np.where(thresh == 255)

                    self.get_logger().info(f"{len(thresh)}, {len(thresh[0])}")
                    self.get_logger().info(f"{len(ys)}, {len(xs)}")

                    # If none found, exit
                    if len(xs) == 0:
                        self.get_logger().warn("No white pixels found.")
                        return False, None, thermal_inv_upscaled
                    else:
                        self.get_logger().info("White pixels found")
                        # Compute centroid
                        cX = int(np.mean(xs))
                        cY = int(np.mean(ys))

                        cv2.circle(output, (cX, cY), 5, (0, 255, 0), -1)
                        cv2.circle(gray_upscaled, (cX*4, cY*4), 5, (0, 255, 0), -1)
                        cv2.circle(thermal_inv_upscaled, (cX*4, cY*4), 5, (0, 255, 0), -1)
                        cv2.circle(thresh_upscaled, (cX*4, cY*4), 5, (0, 255, 0), -1)
            
                cv2.imshow("Point Capture", thermal_inv_upscaled)
                # cv2.waitKey(1)

                self.get_logger().info("keep this point? [y/n]")   
                key = cv2.waitKey(0) & 0xFF
                
                if key == ord('y'):
                    # store point in local list -> don't keep any points if any
                    # image-centroid mapping is not approved

                    # store mapping of 2D projection and 3D pose
                    T_base_to_eef = self.get_fk()
                    world_pos = T_base_to_eef[:3, 3] + T_base_to_eef[:3, :3]@self.eef_tool_offset
                    world_T = np.eye(4)
                    world_T[:3,:3] = T_base_to_eef[:3,:3]
                    world_T[:3,3] = world_pos
                    # self.centroid_point_data[data_idx].append({
                    c_list.append({
                        'capture_id': len(self.centroid_point_data[data_idx]) + 1,
                        '2d_pt': np.array([np.float32(cX), np.float32(cY)]),
                        # '3d_pt': (T_base_to_eef[:3, 3] + T_base_to_eef[:3, :3]@self.eef_tool_offset),
                        '3d_pt': world_pos,
                        'T_3d_pt': world_T,
                        'timestamp': time.time()
                    })
                    # cDictionary.append({
                    #     'capture_id': len(self.centroid_point_data) + 1,
                    #     '2d_pt': np.array([np.float32(cX), np.float32(cY)]),
                    #     '3d_pt': (T_base_to_eef[:3, 3] + T_base_to_eef[:3, :3]@self.eef_tool_offset),
                    #     'timestamp': time.time()
                    # })

                    # self.get_logger().info("c to capture next image, n to end")   
                    # key = cv2.waitKey(0) & 0xFF
                    # if key == ord('c'):
                    #     continue

                else:
                    self.get_logger().info(f"Pose dropped for Camera {data_idx}.")
                    self.get_logger().info("Total points:")
                    for cam_idx in range(len(self.centroid_point_data)):
                        self.get_logger().info(f"Camera {cam_idx}: {len(self.centroid_point_data[cam_idx])}")
                    return False
            else:
                self.get_logger().info(f"Camera {data_idx} Contours not found.")

        self.get_logger().info(f"clist: {c_list}")

        for data_idx in range(len(c_list)):

            self.centroid_point_data[data_idx].append(c_list[data_idx])

            # self.get_logger().info("object point {}".format(cDictionary[-1]["3d_pt"]))
            self.get_logger().info("object point {}".format(self.centroid_point_data[data_idx][-1]["3d_pt"]))
            # self.get_logger().info("image point {}".format(cDictionary[-1]["2d_pt"]))
            self.get_logger().info("image point {}".format(self.centroid_point_data[data_idx][-1]["2d_pt"]))
        
            self.get_logger().info(f"✓ Pose {len( self.centroid_point_data[data_idx])} for Camera {data_idx} collected successfully!")
            self.get_logger().info(f"  Total poses collected: {sum([len(arr) for arr in self.centroid_point_data])}")
            
            # Save image
            # output_dir = "/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/extrinsic_calib_1/"
            output_dir = '/'.join([self.pkg_path, f"calibration_data/extrinsic/{self.extrinsics_dir_pfx}{data_idx}"])
            try:
                if not os.path.isdir(output_dir):
                    os.mkdir(output_dir)
            except:
                self.get_logger().error(f"directory missing {output_dir}")
            filename = f'thermal_calib_{len(self.centroid_point_data[data_idx]):03d}.png'
            cv2.imwrite(os.path.join(output_dir, filename), thermal_datas[data_idx])
            self.get_logger().info(f"✓ Captured thermal image {filename}")
            filename = f'thermal_calib_capture_{len(self.centroid_point_data[data_idx]):03d}.png'
            cv2.imwrite(os.path.join(output_dir, filename), thermal_inv_upscaled)
            self.get_logger().info(f"✓ Captured point capture {filename}")

            # Auto-save backup
            self.save_calibration_backup(data_idx)

        self.get_logger().info("Total points:")
        for cam_idx in range(len(self.centroid_point_data)):
            self.get_logger().info(f"Camera {cam_idx}: {len(self.centroid_point_data[cam_idx])}")
        return True


    def calculate_pnp(self):
        # Solve PnP for checkerboard pose relative to camera
        # self.centroid_point_data = [[{'2d_pt': np.array([0.,0.]), '3d_pt': self.get_fk()[:3, 3] + self.eef_tool_offset} for _ in range(6)] for _ in range(4)]
        T_cam2base_list = []
        for nCamera in range(self.camera_count):
            self.get_logger().info(f"Calculating PnP for camera {nCamera}/{self.camera_count}, {len(self.centroid_point_data)}, {len(self.centroid_point_data[nCamera])}")
            try:
                obj_pts = np.array([self.centroid_point_data[nCamera][sample_idx]['3d_pt'] for sample_idx in range(len(self.centroid_point_data[nCamera]))])
                img_pts = np.array([self.centroid_point_data[nCamera][sample_idx]['2d_pt'] for sample_idx in range(len(self.centroid_point_data[nCamera]))])
            except Exception as e:
                self.get_logger().error(f"Failed to get obj and img points: {e}")
                continue
            self.get_logger().info(f"objpts for camera {nCamera} {obj_pts} {obj_pts.shape}")
            self.get_logger().info(f"imgpts for camera {nCamera} {img_pts} {img_pts.shape}")
            success, rvec, tvec = cv2.solvePnP(
                obj_pts, 
                img_pts,
                self.camera_matrix_list[nCamera], self.dist_coeffs_list[nCamera]
            )
            if success:
                R_mat, _ = cv2.Rodrigues(rvec)

                R_obj2cam = R_mat
                t_obj2cam = tvec.flatten()

                # camera in object frame:
                R_cam2obj = R_obj2cam.T
                t_cam2obj = -R_obj2cam.T.dot(t_obj2cam)

                # if your “object” frame = robot base frame, then
                T_cam2base = np.eye(4)
                T_cam2base[:3,:3] = R_cam2obj
                T_cam2base[:3, 3] = t_cam2obj
                T_cam2base_list.append(T_cam2base)

                self.get_logger().info(f"final pnp for camera {nCamera}: {T_cam2base}")
            else:
                self.get_logger().error(f"pnp failure for camera {nCamera}")
                return None
        return T_cam2base_list 

    # def find_point_centroids_in_thermal(self, thermal_data):
    #     """Find hottest points of interest for thermal processing approach"""
    #     # Convert raw thermal to 8-bit RGB (your method)
    #     thermal_8bit_rgb = raw_to_8bit(thermal_data)
        
    #     # Convert to grayscale and invert (your method)
    #     gray = cv2.cvtColor(thermal_8bit_rgb, cv2.COLOR_BGR2GRAY)
    #     thermal_inv = 255 - gray

    #     _, thresh = cv2.threshold(thermal_inv, 127, 255, cv2.THRESH_BINARY)
    #     thresh_inv = thresh.copy()
    #     thresh = 255 - thresh

    #     print("Unique values:", np.unique(thresh))  # Should show [0 255]

    #     assert thresh.dtype == np.uint8
    #     assert set(np.unique(thresh)).issubset({0, 255})
    #     # blur = cv2.GaussianBlur(gray,(5,5),0)
    #     # alpha = 1.3  # Contrast control
    #     # beta = 40    # Brightness control

    #     # adjusted_image = cv2.convertScaleAbs(blur, alpha=alpha, beta=beta)
    #     # # _, binary_image = cv2.threshold(blur, 230, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    #     # binary_image = cv2.adaptiveThreshold(
    #     #     blur, 255, 
    #     #     cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #     #     cv2.THRESH_BINARY, 
    #     #     23, 2
    #     # )
        
    #     # num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_image)

    #     # # Find contours in the binary image
    #     # # contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #     # # contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9))
    #     # # dilation = cv2.dilate(binary_image,kernel,iterations = 1)
    #     # closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)

    #     # contours, _ = cv2.findContours(closing, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)

    #     # # Convert binary image to BGR to draw in color
    #     # thresh_bgr = cv2.cvtColor(binary_image, cv2.COLOR_GRAY2BGR)

    #     # gray_upscale_bgr = cv2.cvtColor(cv2.resize(gray, (640, 480)), cv2.COLOR_GRAY2BGR)

    #     # # Loop through contours and draw centroids
    #     # blurred = cv2.GaussianBlur(thermal_inv, (5, 5), 0)

    #     # # Threshold the image to find bright (hot) regions
    #     # _, thresh = cv2.threshold(blurred, 30, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #     # thresh_bgr = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)

    #     # Find contours
    #     contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #     output = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
    #     gray_upscaled = cv2.cvtColor(cv2.resize(gray, (640, 480)), cv2.COLOR_GRAY2BGR)
    #     thermal_inv_upscaled = cv2.cvtColor(cv2.resize(thermal_inv, (640, 480)), cv2.COLOR_GRAY2BGR)
    #     thresh_upscaled = cv2.resize(output, (640, 480))


    #     # Find the largest contour, assuming it's the hot spot
    #     if contours:
    #         largest = max(contours, key=cv2.contourArea)

    #         # Compute centroid
    #         M = cv2.moments(largest)
    #         if M["m00"] != 0:
    #             print("centroid identified")
    #             cX = int(M["m10"] / M["m00"])
    #             cY = int(M["m01"] / M["m00"])
                
    #             # Draw the centroid on the image
    #             cv2.circle(output, (cX, cY), 5, (0, 255, 0), -1)
    #             cv2.circle(gray_upscaled, (cX*4, cY*4), 5, (0, 255, 0), -1)
    #             cv2.circle(thermal_inv_upscaled, (cX*4, cY*4), 5, (0, 255, 0), -1)
    #             cv2.circle(thresh_upscaled, (cX*4, cY*4), 5, (0, 255, 0), -1)
    #         else:
    #             # Find coordinates of white pixels
    #             ys, xs = np.where(thresh == 255)

    #             print(len(thresh), len(thresh[0]))
    #             print(len(ys), len(xs))

    #             # If none found, exit
    #             if len(xs) == 0:
    #                 print("No white pixels found.")
    #                 return False, None, thermal_inv_upscaled, None
    #             else:
    #                 print("White pixels found")
    #                 # Compute centroid
    #                 cX = int(np.mean(xs))
    #                 cY = int(np.mean(ys))

    #                 cv2.circle(output, (cX, cY), 5, (0, 255, 0), -1)
    #                 cv2.circle(gray_upscaled, (cX*4, cY*4), 5, (0, 255, 0), -1)
    #                 cv2.circle(thermal_inv_upscaled, (cX*4, cY*4), 5, (0, 255, 0), -1)
    #                 cv2.circle(thresh_upscaled, (cX*4, cY*4), 5, (0, 255, 0), -1)
    #     # # for cnt in centroids:
    #     # #     M = cv2.moments(cnt)
    #     # #     if M["m00"] > 0:
    #     # #         cX = int(M["m10"] / M["m00"])
    #     # #         cY = int(M["m01"] / M["m00"])
    #     # #         # Draw red dot at the centroid
    #     # #         cv2.circle(thresh_bgr, (cX, cY), 4, (0, 0, 255), -1)
    #     # #         cv2.circle(gray_upscale_bgr, (cX*4, cY*4), 4, (0, 0, 255), -1)
                

    #     # # Show the binary image with centroids
    #     # cv2.imshow("Centroids on Binary Image", thresh_bgr)
    #     # cv2.imshow("Centroids on Gray Image", gray_upscale_bgr)
    #     # cv2.waitKey(1)

    #     # # --- Set up SimpleBlobDetector parameters ---
    #     # params = cv2.SimpleBlobDetector_Params()

    #     # # Filter by area (tune minArea to ignore noise)
    #     # params.filterByArea = True
    #     # params.minArea = 1
    #     # params.maxArea = 500  # adjust based on your image

    #     # # Detect bright blobs
    #     # params.blobColor = 255  # Looking for bright blobs (hot spots in inverted image)
    #     # # params.blobColor = max(230, np.max(thresh))

    #     # # Optional filters
    #     # params.filterByCircularity = False
    #     # params.filterByConvexity = False
    #     # params.filterByInertia = False
    #     # # params.filterByCircularity = True
    #     # # params.minCircularity = 0.3

    #     # # Create detector
    #     # detector = cv2.SimpleBlobDetector_create(params)

    #     # # Detect blobs
    #     # # keypoints = detector.detect(thermal_inv)
    #     # keypoints = detector.detect(thresh)

    #     # # Debug info
    #     # print(f"Detected {len(keypoints)} blobs.")
    #     # for i, kp in enumerate(keypoints):
    #     #     print(f"Blob {i}: (x={kp.pt[0]:.2f}, y={kp.pt[1]:.2f}), size={kp.size:.2f}")

    #     # # Draw detected blobs as red circles
    #     # output = thresh.copy()
    #     # output = cv2.drawKeypoints(output, keypoints, None, (0, 0, 255),
    #     #                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    #     # gray_upscaled = cv2.resize(gray, (640, 480))
    #     # keypoints_upscaled = [cv2.KeyPoint(
    #     #         x=kp.pt[0] * 4,
    #     #         y=kp.pt[1] * 4,
    #     #         size=kp.size * 4,  # optional, scale size too
    #     #         angle=kp.angle,
    #     #         response=kp.response,
    #     #         octave=kp.octave,
    #     #         class_id=kp.class_id
    #     #     ) for kp in keypoints]
    #     # gray_upscaled = cv2.drawKeypoints(gray_upscaled, keypoints_upscaled, None, (0, 0, 255),
    #     #                         cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #     # thermal_inv_upscaled = cv2.resize(thermal_inv, (640, 480))
    #     # thermal_inv_upscaled = cv2.drawKeypoints(thermal_inv_upscaled, keypoints_upscaled, None, (0, 0, 255),
    #     #                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #     # thresh_upscaled = cv2.resize(thresh, (640, 480))
    #     # thresh_upscaled = cv2.drawKeypoints(thresh_upscaled, keypoints_upscaled, None, (0, 0, 255),
    #     #                 cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    #     # Show result
    #     # cv2.imshow("Blobs", output)
    #     # cv2.imshow("Blobs Upscaled", gray_upscaled)
    #     # cv2.imshow("Blobs Inv Upscaled", thermal_inv_upscaled)
    #     # cv2.imshow("Blobs Thresh Upscaled", thresh_upscaled)
    #     # cv2.imshow("Thesh Inv", thresh_inv)
    #     # cv2.waitKey(1)

    #     return True, None, None, None

    # def validate_calibration(self, T_cam2base):
    #     """Validate calibration by computing reprojection errors"""
    #     self.get_logger().info("\n=== VALIDATION ===")

    #     for cam_idx in range(self.camera_count):
    #         errors = []
    #         for point_data in self.centroid_point_data[cam_idx]:
    #             # Transform EEF tool to camera using calibrated transform
    #             T_base2eef = point_data['3d_pt']
    #             if T_base2eef.shape[0] == 1 or T_base2eef.shape[0] == 3:
    #                 # convert 3D position vector to transform matrix
    #                 T_eye = np.eye(4)
    #                 T_eye[:3,3] = T_base2eef
    #                 T_base2eef = T_eye
    #             T_eef2base = np.linalg.inv(T_base2eef)
    #             self.get_logger().info(f"T_base2eef {T_eef2base.shape}")
    #             T_base2cam = np.linalg.inv(T_cam2base[cam_idx])
    #             self.get_logger().info(f"T_base2cam {T_base2cam.shape}")
    #             T_eef2cam_pred = T_eef2base@T_base2cam
    #             self.get_logger().info("got transform")

    #             pt_in_base = point_data['3d_pt']  # shape (3,)
    #             pt_in_base_h = np.append(pt_in_base, 1.0)  # make it homogeneous

    #             T_base2cam = np.linalg.inv(T_cam2base[cam_idx])
    #             pt_in_cam = T_base2cam @ pt_in_base_h
    #             pt_in_cam = pt_in_cam[:3].reshape(1, 3)

    #             # Get rvec, tvec (camera pose relative to base)
    #             rvec, _ = cv2.Rodrigues(np.eye(3))  # Or use the actual rotation if available
    #             tvec = np.zeros((3, 1))             # Or set to proper tvec if needed

    #             projected_points, _ = cv2.projectPoints(
    #                 pt_in_cam, np.zeros((3, 1)), np.zeros((3, 1)),
    #                 self.camera_matrix_list[cam_idx],
    #                 self.dist_coeffs_list[cam_idx]
    #             )

    #             # Project to image
    #             rvec_pred, _ = cv2.Rodrigues(T_eef2cam_pred[:3, :3])
    #             tvec_pred = T_eef2cam_pred[:3, 3]

    #             self.get_logger().info(f"Camera {cam_idx} Predicted pose:\n{T_eef2cam_pred}")
    #             self.get_logger().info(f"Camera {cam_idx} tvec_pred: {tvec_pred}")
    #             self.get_logger().info(f"Camera {cam_idx} rvec_pred: {rvec_pred}")

    #             projected_points, _ = cv2.projectPoints(
    #                 point_data['3d_pt'], rvec_pred, tvec_pred,
    #                 self.camera_matrix_list[cam_idx],
    #                 self.dist_coeffs_list[cam_idx]
    #             )

    #             # Compare with observed point
    #             observed = point_data['2d_pt']
    #             error = np.sqrt(np.sum(
    #                 (projected_points.reshape(-1, 2) - observed.reshape(-1, 2))**2, axis=1))
    #             errors.extend(error)

    #             self.get_logger().info("")
            
    #         if errors:
    #             mean_error = np.mean(errors)
    #             max_error = np.max(errors)
                
    #             self.get_logger().info(f"Camera {cam_idx} Mean reprojection error: {mean_error:.2f} pixels")
    #             self.get_logger().info(f"Camera {cam_idx} Max reprojection error:  {max_error:.2f} pixels")
                
    #             if mean_error < 2.0:
    #                 self.get_logger().info("Quality: EXCELLENT ✓✓")
    #             elif mean_error < 3.0:
    #                 self.get_logger().info("Quality: GOOD ✓")
    #             elif mean_error < 5.0:
    #                 self.get_logger().info("Quality: ACCEPTABLE ⚠")
    #             else:
    #                 self.get_logger().warn("Quality: POOR ✗")

    #     # for pose_data in self.calibration_data:
    #     #     # Transform checkerboard to camera using calibrated transform
    #     #     T_base_to_board = pose_data['T_base_to_board']
    #     #     T_base2cam = np.linalg.inv(T_cam2base)
    #     #     T_board2cam_pred = T_base2cam @ T_base_to_board

    #     #     # Project to image
    #     #     rvec_pred, _ = cv2.Rodrigues(T_board2cam_pred[:3, :3])
    #     #     tvec_pred = T_board2cam_pred[:3, 3]
            
    #     #     print("Predicted pose:\n", T_board2cam_pred)
    #     #     print("tvec_pred:", tvec_pred)
    #     #     print("rvec_pred:", rvec_pred)

    #     #     projected_points, _ = cv2.projectPoints(
    #     #         self.objp, rvec_pred, tvec_pred,
    #     #         self.camera_matrix, self.dist_coeffs
    #     #     )
            
    #     #     # Compare with observed corners
    #     #     observed = pose_data['corners']
    #     #     error = np.sqrt(np.sum(
    #     #         (projected_points.reshape(-1, 2) - observed.reshape(-1, 2))**2, axis=1))
    #     #     errors.extend(error)
        
    #     # if errors:
    #     #     mean_error = np.mean(errors)
    #     #     max_error = np.max(errors)
            
    #     #     print(f"Mean reprojection error: {mean_error:.2f} pixels")
    #     #     print(f"Max reprojection error:  {max_error:.2f} pixels")
            
    #     #     if mean_error < 2.0:
    #     #         print("Quality: EXCELLENT ✓✓")
    #     #     elif mean_error < 3.0:
    #     #         print("Quality: GOOD ✓")
    #     #     elif mean_error < 5.0:
    #     #         print("Quality: ACCEPTABLE ⚠")
    #     #     else:
    #     #         print("Quality: POOR ✗")

    def validate_calibration(self, T_cam2base):
        """Validate calibration by computing reprojection errors"""
        self.get_logger().info("\n=== VALIDATION ===")

        for cam_idx in range(self.camera_count):
            errors = []

            for point_idx in range(len(self.centroid_point_data[cam_idx])):
                # point_data['3d_pt'] is a 4x4 transform: T_base2eef
                T_base2eef = self.centroid_point_data[cam_idx][point_idx]['T_3d_pt']
                # T_base2eef = point_data['3d_pt']
                # T_eye = np.eye(4)
                # T_eye[:3,3] = T_base2eef
                # T_base2eef = T_eye

                # Define the fixed point in the EEF frame (e.g., tool tip or marker location)
                pt_eef = np.append(self.eef_tool_offset, 1.0)

                # Compute point in base frame
                pt_in_base = T_base2eef @ pt_eef

                # Transform to camera frame
                T_base2cam = np.linalg.inv(T_cam2base[cam_idx])
                pt_in_cam = T_base2cam @ pt_in_base
                pt_in_cam = pt_in_cam[:3].reshape(1, 3)  # Drop homogeneous coordinate

                # Project point using camera intrinsics
                rvec = np.zeros((3, 1))  # No rotation needed (already in camera frame)
                tvec = np.zeros((3, 1))  # No translation needed
                projected_points, _ = cv2.projectPoints(
                    pt_in_cam,
                    rvec,
                    tvec,
                    self.camera_matrix_list[cam_idx],
                    self.dist_coeffs_list[cam_idx]
                )

                # Compare with observed 2D point
                observed = self.centroid_point_data[cam_idx][point_idx]['2d_pt'].reshape(1, 2)
                error = np.linalg.norm(projected_points.reshape(1, 2) - observed)
                errors.append(error)

                # draw the two points onto the associated image
                try:
                    compare_img_dir = '/'.join([self.pkg_path, f"calibration_data/extrinsic/{self.extrinsics_dir_pfx}{cam_idx}"])
                    compare_img_path = os.path.join(
                        compare_img_dir,
                        f'thermal_calib_capture_00{point_idx+1}.png',
                    )
                    compare_img = cv2.imread(compare_img_path, cv2.IMREAD_COLOR)
                    cv2.circle(compare_img, projected_points.reshape(2).astype(np.int32)*4, 5, (255, 0, 0), -1)    # estimate
                    cv2.circle(compare_img, observed.reshape(2).astype(np.int32)*4, 5, (0, 0, 255), -1)    # observed
                    filename = f'thermal_validate_extr_{point_idx+1:03d}.png'
                    self.get_logger().info(f"file path {os.path.join(compare_img_dir, filename)}")
                    cv2.imwrite(os.path.join(compare_img_dir, filename), compare_img)
                except Exception as e:
                    self.get_logger().error(f"Comparison image creation failed: {e}")

            # Report errors
            if errors:
                mean_error = np.mean(errors)
                max_error = np.max(errors)
                
                self.get_logger().info(f"Camera {cam_idx} Mean reprojection error: {mean_error:.2f} pixels")
                self.get_logger().info(f"Camera {cam_idx} Max reprojection error:  {max_error:.2f} pixels")
                
                if mean_error < 2.0:
                    self.get_logger().info("\tQuality: EXCELLENT ✓✓")
                elif mean_error < 3.0:
                    self.get_logger().info("\tQuality: GOOD ✓")
                elif mean_error < 5.0:
                    self.get_logger().info("\tQuality: ACCEPTABLE ⚠")
                else:
                    self.get_logger().warn("\tQuality: POOR ✗")

    def save_calibration_backup(self, idx):
        """Save backup of calibration data"""
        try:
            # pkg_path = rospkg.RosPack().get_path('thermal_camera')
            # pkg_path = "/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera"
            backup_file = os.path.join(self.pkg_path, f'calibration_data/extrinsic/{self.extrinsics_dir_pfx}{idx}/{self.calibration_backup_file}.pkl')
        except:
            self.get_logger().error(f"cannot make file {backup_file}")

        with open(backup_file, 'wb') as f:
            pickle.dump(self.centroid_point_data, f)

    def save_extrinsic_results(self, T_cam2base_list):
        """Save final calibration results"""
        for idx in range(len(T_cam2base_list)):
            package = {
                'camera_to_base_transform': T_cam2base_list[idx],
                'eef_tool_translation': self.eef_tool_offset,
                'intrinsic_calibration': {
                    'camera_matrix': self.camera_matrix_list[idx],
                    'dist_coeffs': self.dist_coeffs_list[idx],
                    'rms_error': self.intrinsic_rms_error_list[idx]
                },
                'metadata': {
                    'method': 'opencv_hand_eye_tsai',
                    'num_poses': len(self.centroid_point_data),
                    'date': datetime.now().isoformat(),
                }
            }
            
            try:
                # pkg_path = rospkg.RosPack().get_path('thermal_camera')
                pkg_path = "/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera"
                result_file = os.path.join(self.pkg_path, f'calibration_data/extrinsic/{self.extrinsics_dir_pfx}{idx}/{self.extrinsics_file}.pkl')
                self.get_logger().info(f"Saving extrinsic results for Camera {idx} to {result_file}")
                # result_file = os.path.join(pkg_path, f'calibration_data/extrinsic/{self.extrinsics_dir_pfx}{idx}/{self.extrinsics_file}.pkl')
            except:
                self.get_logger().error("Result file not found.")
            
            try:
                with open(result_file, 'wb') as f:
                    pickle.dump(package, f)
                self.get_logger().info("Save success")
            except Exception as e:
                self.get_logger().error(f"Saving failed: {e}")
        

    def print_instructions(self):
        """Print calibration instructions"""
        barrier_str = "="*60
        self.get_logger().info(f"\n{barrier_str}")
        self.get_logger().info("UR10e THERMAL CAMERA EXTRINSIC CALIBRATION")
        self.get_logger().info(barrier_str)
        self.get_logger().info("SETUP:")
        self.get_logger().info("  1. Attach the incandescent bulb tool to the UR10e end effector.")
        self.get_logger().info("  2. Turn on incandescent bulb on the tool.")
        self.get_logger().info("  3. Put UR10e in teach mode")
        self.get_logger().info("  4. Move robot manually to different poses")
        self.get_logger().info("")
        self.get_logger().info("COLLECTION STRATEGY:")
        self.get_logger().info("  • Vary position and distance from the camera")
        self.get_logger().info("  • Collect 10-15 good poses")
        self.get_logger().info("  • Ensure incandescent bulb is clearly visible")
        self.get_logger().info("")
        self.get_logger().info("COMMANDS:")
        self.get_logger().info("  'c' - Capture current pose")
        self.get_logger().info("  's' - Compute calibration")
        self.get_logger().info("  'f' - Show current robot EEF forward kinematics as a 4x4 matrix")
        self.get_logger().info("  'l' - Load from the calibration backup and compute calibration")
        self.get_logger().info("  'r' - Reset data")
        self.get_logger().info("  'v' - Visual check (show detection)")
        self.get_logger().info("  'q' - Quit")
        self.get_logger().info(barrier_str)

    def load_calibration_backup(self, file_path=None):
        """
        Load saved calibration poses from a pickle file.

        :param file_path: Optional path to the .pkl file.
                        If None, default path in the thermal_camera package is used.
        :return: List of calibration pose dictionaries or None if loading fails.
        """
        try:
            if file_path is None:
                # Default path from your project structure
                file_path = os.path.join(self.pkg_path, f'calibration_data/extrinsic/{self.extrinsics_dir_pfx}{self.camera_count-1}/{self.calibration_backup_file}.pkl')
            

            if not os.path.exists(file_path):
                self.get_logger().error(f"✗ Calibration file not found: {file_path}")
                return None

            with open(file_path, 'rb') as f:
                calibration_data = pickle.load(f)

            self.get_logger().info(f"✓ Loaded calibration data from {file_path}")
            self.centroid_point_data = calibration_data

            self.get_logger().info("Total points:")
            for cam_idx in range(len(self.centroid_point_data)):
                self.get_logger().info(f"Camera {cam_idx}: {len(self.centroid_point_data[cam_idx])}")

            return calibration_data

        except Exception as e:
            self.get_logger().error(f"✗ Error loading calibration poses: {e}")
            return None

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

def main():
    """Main function - integrated with your camera system"""
    rclpy.init()
    # Initialize calibration system
    calibrator = UR10ExtrinsicCalibration()
    calibrator.print_instructions()

    # calibrator.load_calibration_poses()

    # if len(calibrator.calibration_data) >= 3:
    #     T_cam2base = calibrator.compute_hand_eye_calibration()
    #     if T_cam2base is not None:
    #         calibrator.validate_calibration(T_cam2base)
    #         calibrator.save_extrinsic_results(T_cam2base)
    #         print("Completed")
    # else:
    #     print(f"Need ≥3 poses, have {len(calibrator.calibration_data)}")

    # return

    
    camera_count = calibrator.camera_count
    buffer_size = 2
    qs = [Queue(buffer_size) for _ in range(camera_count)]
    ctx = POINTER(uvc_context)()
    dev = POINTER(uvc_device)()
    devs = POINTER(POINTER(uvc_device))()
    devh = (POINTER(uvc_device_handle) * camera_count)()
    ctrls = [uvc_stream_ctrl() for _ in range(camera_count)]
    res = libuvc.uvc_init(byref(ctx), 0)
    callbacks = [make_callback(i, qs[i]) for i in range(camera_count)]


    if res < 0:
        calibrator.get_logger().error()("uvc_init error")
        exit(1)

    try:
        res = libuvc.uvc_find_devices(ctx, byref(devs), PT_USB_VID, PT_USB_PID, 0)
        calibrator.get_logger().info(f"{res}")
        i = 0
        while devs[i]:
            calibrator.get_logger().info(f"{devs.contents[i]}")
            i += 1
        calibrator.get_logger().info('Number of devices: {}'.format(i))
        if res < 0:
          calibrator.get_logger().error(f"uvc_find_device error {res}")
          exit(1)

        # open each camera
        for i in range(camera_count):
            res = libuvc.uvc_open(devs[i], byref(devh[i]))
            if res < 0:
                calibrator.get_logger().error(f"uvc_open {i} error")
                exit(1)
            # calibrator.get_logger().info(print_device_info(devh[i]))
            # calibrator.get_logger().info(print_device_formats(devh[i]))

        calibrator.get_logger().info("Thermal camera opened!")
        # configure stream format
        for i in range(camera_count):
            frame_format = uvc_get_frame_formats_by_guid(devh[i], VS_FMT_GUID_Y16)
            if len(frame_format) == 0:
                calibrator.get_logger().error(f"device {i} does not support Y16")
                exit(1)
            libuvc.uvc_get_stream_ctrl_format_size(devh[i], byref(ctrls[i]), UVC_FRAME_FORMAT_Y16,
                frame_format[0].wWidth, frame_format[0].wHeight, int(1e7 / frame_format[0].dwDefaultFrameInterval)
            )

        # start streaming for each camera
        for i in range(camera_count):
            res = libuvc.uvc_start_streaming(devh[i], byref(ctrls[i]), callbacks[i], None, 0)
            if res < 0:
              calibrator.get_logger().error("uvc_start_streaming {0} failed: {0}".format(i, res))
              exit(1)
            time.sleep(0.5)
        calibrator.get_logger().info("Streaming started! Move robot and press 'c' to capture poses...")
        
        visual_mode = False
        
        try:
            while rclpy.ok():
                rclpy.spin_once(calibrator, timeout_sec=0.01)
                datas = []
                for i in range(camera_count):
                    try:
                        data = qs[i].get(timeout=2)
                    except:
                        calibrator.get_logger().warn(f"[Main Loop] Timed out waiting for camera {i}")
                        data = None
                    datas.append(data)      
                if any(datas[i] is None for i in range(camera_count)):
                    break
                
                if not visual_mode:
                    # upscale and display camera feeds
                    for i in range(camera_count):
                        data = cv2.resize(datas[i][:,:], (640, 480))
                        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)
                        img = raw_to_8bit(data)
                        display_temperature(img, minVal, minLoc, (255, 0, 0))
                        display_temperature(img, maxVal, maxLoc, (0, 0, 255))
                        cv2.imshow(f'Lepton Radiometry {i}', img)

                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('c'):
                    # success = calibrator.collect_point_centroid_in_thermal(datas)
                    success = calibrator.ba_collect_point_centroid_in_thermal(datas)
                    if success:
                        calibrator.get_logger().info(f"point collected, {len(calibrator.centroid_point_data)} total points captured")
                    else:
                        # calibrator.get_logger().error("centroid collection failed")
                        calibrator.get_logger().warn(f"point not collected, {len(calibrator.centroid_point_data)} total points captured")
                    # success = calibrator.collect_calibration_pose(data)
                    # if success:
                    #     print(f"Poses collected: {len(calibrator.calibration_data)}")
                    #     print("")
                
                elif key == ord('s'):
                    # if len(calibrator.centroid_point_data) >= 6:
                    if any(len(cam_arr) >= 6 for cam_arr in calibrator.centroid_point_data):
                    # if len(calibrator.centroid_point_data) >= 0:
                        T_cam2base = calibrator.calculate_pnp()
                        if T_cam2base is not None:
                            calibrator.validate_calibration(T_cam2base)
                            calibrator.save_extrinsic_results(T_cam2base)
                            calibrator.get_logger().info("completed")
                            break
                    else:
                        calibrator.get_logger().warn(f"All cameras need ≥6 points, have {[len(cam_arr) for cam_arr in calibrator.centroid_point_data]}")

                    # if len(calibrator.calibration_data) >= 3:
                    #     T_cam2base = calibrator.compute_hand_eye_calibration()
                    #     if T_cam2base is not None:
                    #         calibrator.validate_calibration(T_cam2base)
                    #         calibrator.save_extrinsic_results(T_cam2base)
                    #         print("Completed")
                    #         break
                    # else:
                    #     print(f"Need ≥3 poses, have {len(calibrator.calibration_data)}")

                elif key == ord('l'):
                    calibrator.load_calibration_backup()
                    # if any(len(cam_arr) >= 6 for cam_arr in calibrator.centroid_point_data):
                    #     T_cam2base = calibrator.calculate_pnp()
                    #     if T_cam2base is not None:
                    #         calibrator.validate_calibration(T_cam2base)
                    #         calibrator.save_extrinsic_results(T_cam2base)
                    #         calibrator.get_logger().info("completed")
                    #         break
                    # else:
                    #     calibrator.get_logger().warn(f"All cameras need ≥6 points, have {[len(cam_arr) for cam_arr in calibrator.centroid_point_data]}")

                elif key == ord('f'):
                    calibrator.get_logger().info(f"FK: {calibrator.get_fk()}")

                elif key == ord('r'): 
                    calibrator.calibration_data = []
                    calibrator.get_logger().info("Calibration data reset")
                
                elif key == ord('v'): 
                    visual_mode = not visual_mode
                    if visual_mode:
                        calibrator.get_logger().info("Visual detection mode ON")
                    else:
                        calibrator.get_logger().info("Visual detection mode OFF")
                        cv2.destroyWindow('Extrinsic Calibration')
                
                elif key == ord('i'): 
                    calibrator.print_instructions()
                
                elif key == ord('q'): 
                    break
                
                if visual_mode:
                    ret, corners, img_display, T_board_to_camera = calibrator.find_checkerboard_in_thermal(data)
                    cv2.imshow('Detection Check', img_display)

        finally:
            libuvc.uvc_stop_streaming(devh)
        
        calibrator.get_logger().info("Calibration completed!")
        
    finally:
        libuvc.uvc_unref_device(dev)
        libuvc.uvc_exit(ctx)
        cv2.destroyAllWindows()
        rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("Calibration interrupted")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        try:
            rclpy.shutdown()
        except:
            pass
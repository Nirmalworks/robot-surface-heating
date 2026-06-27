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

class CADRegistration(Node):
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
                ('eef_pointy_offset', []),
                ('dest_dir', ""),
                ('world_pts_offsets', [])
            ]
        )

        self.eef_pointy_offset = self.get_parameter('eef_pointy_offset').get_parameter_value().double_array_value
        self.get_logger().info(f"eef_pointy_offset {self.eef_pointy_offset}")
        self.dest_dir = self.get_parameter('dest_dir').get_parameter_value().string_value
        self.get_logger().info(f"dest_dir {self.dest_dir}")
        self.world_pts_offsets = self.get_parameter('world_pts_offsets').get_parameter_value().double_array_value
        self.get_logger().info(f"world_pts_offsets {self.world_pts_offsets}")

        self.ik_client_ = self.create_client(GetPositionIK, self.ik_srv_name_)
        if not self.ik_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("IK service not available.")
            exit(1)

        self.fk_client_ = self.create_client(GetPositionFK, self.fk_srv_name_)
        if not self.fk_client_.wait_for_service(timeout_sec=self.timeout_sec_):
            self.get_logger().error("FK service not available.")
            exit(1)

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
        
        self.get_logger().info("CAD registration initialized")
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
        self.get_logger().info("  1. Attach the pointy tool to the UR10e end effector.")
        self.get_logger().info("  2. Put UR10e in teach mode")
        self.get_logger().info("  3. Move robot manually to different poses")
        self.get_logger().info("")
        self.get_logger().info("COLLECTION STRATEGY:")
        self.get_logger().info("  • Collect 4 poses associated with known points on the CAD model.")
        self.get_logger().info("  • Make sure the pointy tool tip is in direct contact with the target point.")
        self.get_logger().info("")
        self.get_logger().info("COMMANDS:")
        self.get_logger().info("  'c' - Capture FK")
        self.get_logger().info("  's' - Save accumulated FK Data")
        self.get_logger().info("  'f' - Show current robot EEF forward kinematics as a 4x4 matrix")
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
    calibrator = CADRegistration()
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

    
    # camera_count = calibrator.camera_count
    # buffer_size = 2
    # qs = [Queue(buffer_size) for _ in range(camera_count)]
    # ctx = POINTER(uvc_context)()
    # dev = POINTER(uvc_device)()
    # devs = POINTER(POINTER(uvc_device))()
    # devh = (POINTER(uvc_device_handle) * camera_count)()
    # ctrls = [uvc_stream_ctrl() for _ in range(camera_count)]
    # res = libuvc.uvc_init(byref(ctx), 0)
    # callbacks = [make_callback(i, qs[i]) for i in range(camera_count)]


    # if res < 0:
    #     calibrator.get_logger().error()("uvc_init error")
    #     exit(1)

    # try:
    #     res = libuvc.uvc_find_devices(ctx, byref(devs), PT_USB_VID, PT_USB_PID, 0)
    #     calibrator.get_logger().info(f"{res}")
    #     i = 0
    #     while devs[i]:
    #         calibrator.get_logger().info(f"{devs.contents[i]}")
    #         i += 1
    #     calibrator.get_logger().info('Number of devices: {}'.format(i))
    #     if res < 0:
    #       calibrator.get_logger().error(f"uvc_find_device error {res}")
    #       exit(1)

    #     # open each camera
    #     for i in range(camera_count):
    #         res = libuvc.uvc_open(devs[i], byref(devh[i]))
    #         if res < 0:
    #             calibrator.get_logger().error(f"uvc_open {i} error")
    #             exit(1)
    #         # calibrator.get_logger().info(print_device_info(devh[i]))
    #         # calibrator.get_logger().info(print_device_formats(devh[i]))

    #     calibrator.get_logger().info("Thermal camera opened!")
    #     # configure stream format
    #     for i in range(camera_count):
    #         frame_format = uvc_get_frame_formats_by_guid(devh[i], VS_FMT_GUID_Y16)
    #         if len(frame_format) == 0:
    #             calibrator.get_logger().error(f"device {i} does not support Y16")
    #             exit(1)
    #         libuvc.uvc_get_stream_ctrl_format_size(devh[i], byref(ctrls[i]), UVC_FRAME_FORMAT_Y16,
    #             frame_format[0].wWidth, frame_format[0].wHeight, int(1e7 / frame_format[0].dwDefaultFrameInterval)
    #         )

    #     # start streaming for each camera
    #     for i in range(camera_count):
    #         res = libuvc.uvc_start_streaming(devh[i], byref(ctrls[i]), callbacks[i], None, 0)
    #         if res < 0:
    #           calibrator.get_logger().error("uvc_start_streaming {0} failed: {0}".format(i, res))
    #           exit(1)
    #         time.sleep(0.5)
    #     calibrator.get_logger().info("Streaming started! Move robot and press 'c' to capture poses...")
        
    #     visual_mode = False
        
    try:
        while rclpy.ok():
            rclpy.spin_once(calibrator, timeout_sec=0.01)
            datas = []
            # for i in range(camera_count):
            #     try:
            #         data = qs[i].get(timeout=2)
            #     except:
            #         calibrator.get_logger().warn(f"[Main Loop] Timed out waiting for camera {i}")
            #         data = None
            #     datas.append(data)      
            # if any(datas[i] is None for i in range(camera_count)):
            #     break
            
            # if not visual_mode:
            #     # upscale and display camera feeds
            #     for i in range(camera_count):
            #         data = cv2.resize(datas[i][:,:], (640, 480))
            #         minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)
            #         img = raw_to_8bit(data)
            #         display_temperature(img, minVal, minLoc, (255, 0, 0))
            #         display_temperature(img, maxVal, maxLoc, (0, 0, 255))
            #         cv2.imshow(f'Lepton Radiometry {i}', img)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
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

            elif key == ord('f'):
                calibrator.get_logger().info(f"FK: {calibrator.get_fk()}")

            elif key == ord('i'): 
                calibrator.print_instructions()
            
            elif key == ord('q'): 
                break

        # finally:
        #     libuvc.uvc_stop_streaming(devh)
        
        calibrator.get_logger().info("Calibration completed!")
        
    finally:
        # libuvc.uvc_unref_device(dev)
        # libuvc.uvc_exit(ctx)
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
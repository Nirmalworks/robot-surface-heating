#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UR10 Extrinsic Calibration - Integrated with your thermal camera system
Based on your intrinsic calibration code

DEPRECATED -> see `pnp_extrinsic.py` for currently used calibration script
"""

from uvctypes import *
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

    base_ = "base_link"
    end_effector_ = "tool0"
    def __init__(self):
        super().__init__("UR10_Extrinsic_Calibration")

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
        self.calibration_data = []
        
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
        
        
        
        print("UR10 Extrinsic Calibration initialized")
        print(f"Checkerboard: {self.CHECKERBOARD_SIZE}, Square size: {self.SQUARE_SIZE*1000:.1f}mm")

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
            pkg_path = "/home/cam/ur_bimanual/src/thermal_camera"
            intrinsic_files = [
                os.path.join(pkg_path, 'lepton0', 'thermal_intrinsics.npz'),
                # os.path.join(pkg_path, 'calibration_images', 'thermal_intrinsics.npz'),
                # 'thermal_intrinsics.npz'  # Current directory
            ]
            
            loaded = False
            for filepath in intrinsic_files:
                if os.path.exists(filepath):
                    data = np.load(filepath)
                    self.camera_matrix = data['mtx']
                    self.dist_coeffs = data['dist']
                    self.intrinsic_rms_error = data['rms']
                    
                    print(f"✓ Loaded intrinsics from: {filepath}")
                    print(f"  RMS error: {self.intrinsic_rms_error:.3f} pixels")
                    print(f"  Camera matrix:\n{self.camera_matrix}")
                    print(f"  Distortion coeffs: {self.dist_coeffs.flatten()}")
                    loaded = True
                    break
            
            if not loaded:
                raise FileNotFoundError("No intrinsic calibration file found")
                
        except Exception as e:
            print(f"✗ ERROR loading intrinsics: {e}")
            print("Run intrinsic calibration first!")
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

    def find_checkerboard_in_thermal(self, thermal_data):
        """Find checkerboard using your thermal processing approach"""
        # Convert raw thermal to 8-bit RGB (your method)
        thermal_8bit_rgb = raw_to_8bit(thermal_data)
        
        # Convert to grayscale and invert (your method)
        gray = cv2.cvtColor(thermal_8bit_rgb, cv2.COLOR_BGR2GRAY)
        thermal_inv = 255 - gray
        
        # Alternative: Use your enhanced preprocessing
        # thermal_inv, thermal_thresh = prepare_thermal_for_detection(thermal_8bit_rgb)
        
        # Find checkerboard corners (same as your intrinsic calibration)
        ret, corners = cv2.findChessboardCorners(
            thermal_inv, 
            self.CHECKERBOARD_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH
        )
        
        if ret:
            # Refine corners to sub-pixel accuracy (same as your intrinsic)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
            corners = cv2.cornerSubPix(thermal_inv, corners, (5, 5), (-1, -1), criteria)
            
            # Solve PnP for checkerboard pose relative to camera
            success, rvec, tvec = cv2.solvePnP(
                self.objp, corners,
                self.camera_matrix, self.dist_coeffs
            )
            
            if success:
                # Convert to transformation matrix
                R_mat, _ = cv2.Rodrigues(rvec)
                T_board_to_camera = np.eye(4)
                T_board_to_camera[:3, :3] = R_mat
                T_board_to_camera[:3, 3] = tvec.flatten()
                
                # Create visualization (resize to match your display)
                display_img = cv2.resize(thermal_inv, (640, 480))
                img_preview = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
                corners_display = corners * (640.0 / thermal_inv.shape[1])  # Scale corners
                cv2.drawChessboardCorners(img_preview, self.CHECKERBOARD_SIZE, corners_display, ret)
                cv2.putText(img_preview, "DETECTED + POSE OK", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                return True, corners, img_preview, T_board_to_camera
            else:
                # Corners found but pose failed
                display_img = cv2.resize(thermal_inv, (640, 480))
                img_preview = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
                corners_display = corners * (640.0 / thermal_inv.shape[1])
                cv2.drawChessboardCorners(img_preview, self.CHECKERBOARD_SIZE, corners_display, ret)
                cv2.putText(img_preview, "CORNERS OK, POSE FAILED", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        else:
            # No corners detected
            display_img = cv2.resize(thermal_inv, (640, 480))
            img_preview = cv2.cvtColor(display_img, cv2.COLOR_GRAY2BGR)
            cv2.putText(img_preview, "NO CHECKERBOARD DETECTED", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return False, None, img_preview, None

    def collect_calibration_pose(self, thermal_data):
        print(f"\n Collecting Pose -> {len(self.calibration_data) + 1} ===")
        
        # Get current robot pose
        T_base_to_eef = self.get_fk()
        if T_base_to_eef is None:
            print("✗ Failed to get robot pose")
            return False

        # Calculate checkerboard pose in robot base frame
        T_base_to_board = T_base_to_eef @ self.T_eef_to_board
        
        # Find checkerboard in thermal image
        ret, corners, img_display, T_board_to_camera = self.find_checkerboard_in_thermal(thermal_data)
        
        # Display detection result
        cv2.imshow('Extrinsic Calibration', img_display)
        cv2.waitKey(100)
        
        if ret:
            # Store calibration data
            calibration_pose = {
                'pose_id': len(self.calibration_data) + 1,
                'T_base_to_eef': T_base_to_eef,
                'T_base_to_board': T_base_to_board,
                'T_board_to_camera': T_board_to_camera,
                'corners': corners,
                'timestamp': time.time() 
            }

            # print("calibration pose", calibration_pose)

            res = input("keep pose? [y/n]")
            if res != "y":
                print("pose dropped")
                return True
            
            self.calibration_data.append(calibration_pose)
            
            print(f"✓ Pose {len(self.calibration_data)} collected successfully!")
            print(f"  Total poses collected: {len(self.calibration_data)}")
            
            # Save image
            output_dir = "/home/cam/ur_bimanual/src/thermal_camera/extrinsic_calib_0/"
            filename = f'thermal_calib_{len(self.calibration_data):03d}.png'
            cv2.imwrite(os.path.join(output_dir, filename), thermal_data)
            print(f"✓ Captured {filename}")

            # Auto-save backup
            self.save_calibration_backup()
            return True
        else:
            print("✗ Checkerboard not detected or pose estimation failed")
            return False

    def compute_hand_eye_calibration(self):
        """Compute hand-eye calibration using OpenCV"""
        min_poses = 3
        if len(self.calibration_data) < min_poses:
            print(f"Need at least {min_poses} poses, have {len(self.calibration_data)}")
            return None
        
        print(f"\n=== COMPUTING HAND-EYE CALIBRATION ===")
        print(f"Using {len(self.calibration_data)} collected poses...")
        
        # Collect transformations
        A_matrices = []  # Robot base to EEF
        B_matrices = []  # Target to camera
        
        for pose_data in self.calibration_data:
            # A_matrices.append(pose_data['T_base_to_eef'])
            A_matrices.append(np.linalg.inv(pose_data['T_base_to_eef']))
            B_matrices.append(pose_data['T_board_to_camera'])
        
        # Extract rotation and translation for OpenCV
        R_gripper2base = [A[:3, :3] for A in A_matrices]
        t_gripper2base = [A[:3, 3].reshape(3, 1) for A in A_matrices]
        R_target2cam = [B[:3, :3] for B in B_matrices]
        t_target2cam = [B[:3, 3].reshape(3, 1) for B in B_matrices]
        
        # Convert to numpy arrays
        R_gripper2base = [np.array(R, dtype=np.float64) for R in R_gripper2base]
        t_gripper2base = [np.array(t, dtype=np.float64) for t in t_gripper2base]
        R_target2cam = [np.array(R, dtype=np.float64) for R in R_target2cam]
        t_target2cam = [np.array(t, dtype=np.float64) for t in t_target2cam]
        
        try:
            # Solve hand-eye calibration: A*X = X*B
            R_cam2base, t_cam2base = cv2.calibrateHandEye(
                R_gripper2base, t_gripper2base,
                R_target2cam, t_target2cam,
                method=cv2.CALIB_HAND_EYE_TSAI
            )
            
            # Create transformation matrix
            T_cam2base = np.eye(4)
            T_cam2base[:3, :3] = R_cam2base
            T_cam2base[:3, 3] = t_cam2base.flatten()
            
            print("✓ Hand-eye calibration successful!")
            print(f"Camera to base translation: {t_cam2base.flatten()}")
            
            return T_cam2base
            
        except Exception as e:
            print(f"✗ Hand-eye calibration failed: {e}")
            return None

    def validate_calibration(self, T_cam2base):
        """Validate calibration by computing reprojection errors"""
        print("\n=== VALIDATION ===")
        
        errors = []
        for pose_data in self.calibration_data:
            # Transform checkerboard to camera using calibrated transform
            T_base_to_board = pose_data['T_base_to_board']
            T_base2cam = np.linalg.inv(T_cam2base)
            T_board2cam_pred = T_base2cam @ T_base_to_board
            
            # Project to image
            rvec_pred, _ = cv2.Rodrigues(T_board2cam_pred[:3, :3])
            tvec_pred = T_board2cam_pred[:3, 3]
            
            print("Predicted pose:\n", T_board2cam_pred)
            print("tvec_pred:", tvec_pred)
            print("rvec_pred:", rvec_pred)

            projected_points, _ = cv2.projectPoints(
                self.objp, rvec_pred, tvec_pred,
                self.camera_matrix, self.dist_coeffs
            )
            
            # Compare with observed corners
            observed = pose_data['corners']
            error = np.sqrt(np.sum(
                (projected_points.reshape(-1, 2) - observed.reshape(-1, 2))**2, axis=1))
            errors.extend(error)
        
        if errors:
            mean_error = np.mean(errors)
            max_error = np.max(errors)
            
            print(f"Mean reprojection error: {mean_error:.2f} pixels")
            print(f"Max reprojection error:  {max_error:.2f} pixels")
            
            if mean_error < 2.0:
                print("Quality: EXCELLENT ✓✓")
            elif mean_error < 3.0:
                print("Quality: GOOD ✓")
            elif mean_error < 5.0:
                print("Quality: ACCEPTABLE ⚠")
            else:
                print("Quality: POOR ✗")

    def save_calibration_backup(self):
        """Save backup of calibration data"""
        try:
            # pkg_path = rospkg.RosPack().get_path('thermal_camera')
            pkg_path = "/home/cam/ur_bimanual/src/thermal_camera"
            backup_file = os.path.join(pkg_path, 'ur10_calibration_backup.pkl')
        except:
            backup_file = 'ur10_calibration_backup.pkl'
        
        with open(backup_file, 'wb') as f:
            pickle.dump(self.calibration_data, f)

    def save_extrinsic_results(self, T_cam2base):
        """Save final calibration results"""
        package = {
            'camera_to_base_transform': T_cam2base,
            'eef_to_checkerboard_offset': self.T_eef_to_board,
            'intrinsic_calibration': {
                'camera_matrix': self.camera_matrix,
                'dist_coeffs': self.dist_coeffs,
                'rms_error': self.intrinsic_rms_error
            },
            'metadata': {
                'method': 'opencv_hand_eye_tsai',
                'num_poses': len(self.calibration_data),
                'date': datetime.now().isoformat(),
                'checkerboard_size': self.CHECKERBOARD_SIZE,
                'square_size': self.SQUARE_SIZE
            }
        }
        
        try:
            # pkg_path = rospkg.RosPack().get_path('thermal_camera')
            pkg_path = "/home/cam/ur_bimanual/src/thermal_camera"
            result_file = os.path.join(pkg_path, 'ur10_camera_extrinsics_test2.pkl')
        except:
            result_file = 'ur10_camera_extrinsics.pkl'
        
        with open(result_file, 'wb') as f:
            pickle.dump(package, f)
        

    def print_instructions(self):
        """Print calibration instructions"""
        print("\n" + "="*60)
        print("UR10 THERMAL CAMERA EXTRINSIC CALIBRATION")
        print("="*60)
        print("SETUP:")
        print("  1. Put UR10 in teach mode")
        print("  2. Attach heated checkerboard to end-effector")
        print("  3. Use same checkerboard as intrinsic calibration")
        print("  4. Move robot manually to different poses")
        print()
        print("COLLECTION STRATEGY:")
        print("  • Vary position and orientation")
        print("  • Collect 10-15 good poses")
        print("  • Ensure checkerboard clearly visible")
        print()
        print("COMMANDS:")
        print("  'c' - Capture current pose")
        print("  's' - Compute calibration")
        print("  'r' - Reset data")
        print("  'v' - Visual check (show detection)")
        print("  'q' - Quit")
        print("="*60)

    def load_calibration_poses(self, file_path=None):
        """
        Load saved calibration poses from a pickle file.

        :param file_path: Optional path to the .pkl file.
                        If None, default path in the thermal_camera package is used.
        :return: List of calibration pose dictionaries or None if loading fails.
        """
        try:
            if file_path is None:
                # Default path from your project structure
                file_path = "/home/cam/ur_bimanual/src/thermal_camera/ur10_calibration_backup_test2.pkl"
            
            if not os.path.exists(file_path):
                print(f"✗ Calibration file not found: {file_path}")
                return None

            with open(file_path, 'rb') as f:
                calibration_data = pickle.load(f)

            print(f"✓ Loaded {len(calibration_data)} calibration poses from {file_path}")
            self.calibration_data = calibration_data
            return calibration_data

        except Exception as e:
            print(f"✗ Error loading calibration poses: {e}")
            return None

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

        res = libuvc.uvc_open(dev, byref(devh))
        if res < 0:
            print("uvc_open error")
            exit(1)

        print("Thermal camera opened!")
        frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
        if len(frame_formats) == 0:
            print("Device does not support Y16")
            exit(1)

        libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
            frame_formats[0].wWidth, frame_formats[0].wHeight, 
            int(1e7 / frame_formats[0].dwDefaultFrameInterval))

        res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
        if res < 0:
            print("uvc_start_streaming failed")
            exit(1)

        print("Streaming started! Move robot and press 'c' to capture poses...")
        
        visual_mode = False
        
        try:
            while rclpy.ok():
                rclpy.spin_once(calibrator, timeout_sec=0.01)
                data = q.get(True, 500)
                if data is None:
                    break
                
                if not visual_mode:
                    display_data = cv2.resize(data[:,:], (640, 480))
                    minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)
                    display_img = raw_to_8bit(display_data)
                    display_temperature(display_img, minVal, minLoc, (255, 0, 0))
                    display_temperature(display_img, maxVal, maxLoc, (0, 0, 255))
                    cv2.imshow('Thermal Camera', display_img)

                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('c'):
                    success = calibrator.collect_calibration_pose(data)
                    if success:
                        print(f"Poses collected: {len(calibrator.calibration_data)}")
                        print("")
                
                elif key == ord('s'):
                    if len(calibrator.calibration_data) >= 3:
                        T_cam2base = calibrator.compute_hand_eye_calibration()
                        if T_cam2base is not None:
                            calibrator.validate_calibration(T_cam2base)
                            calibrator.save_extrinsic_results(T_cam2base)
                            print("Completed")
                            break
                    else:
                        print(f"Need ≥3 poses, have {len(calibrator.calibration_data)}")
                
                elif key == ord('r'): 
                    calibrator.calibration_data = []
                    print("Calibration data reset")
                
                elif key == ord('v'): 
                    visual_mode = not visual_mode
                    if visual_mode:
                        print("Visual detection mode ON")
                    else:
                        print("Visual detection mode OFF")
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
        
        print("Calibration completed!")
        
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
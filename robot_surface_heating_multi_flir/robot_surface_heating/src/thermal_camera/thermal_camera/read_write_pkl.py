import pickle
import os
import numpy as np
import sys

"""Simple script for reading and writing from Pickle files for quick debugging. Manually change file paths and data to write as necessary."""

def read_pkl():
    # file_path = '/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/calibration_data/extrinsic/extrinsic_calib_0/ur10_camera_extrinsics_test2.pkl'
    file_path = '/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera/calibration_data/extrinsic/lepton0_test/thermal_extrinsics.pkl'
    try:
        if file_path is None:
            # Default path from your project structure
            file_path = "/home/cam/ur_bimanual/src/thermal_camera/ur10_calibration_backup_test2.pkl"
        
        if not os.path.exists(file_path):
            print(f"✗ Calibration file not found: {file_path}")

        with open(file_path, 'rb') as f:
            calibration_data = pickle.load(f)

        print(f"✓ Loaded {len(calibration_data)} calibration poses from {file_path}")

        print(calibration_data)

    except Exception as e:
        print(f"✗ Error loading calibration poses: {e}")

def write_pkl():
    T_cam_to_base = np.array([
        # base_link to camera_link
        # 0.000  1.000 -0.021  0.799
        # -0.034 -0.021 -0.999 -0.264
        # -0.999  0.001  0.034  1.290
        # 0.000  0.000  0.000  1.000

        # base_link to camera_color_optical_frame
        [-1.000,  0.020,  0.005,  0.814],
        [0.020,  0.999, -0.027, -0.264],
        [-0.005, -0.027, -1.000,  1.290],
        [0.000,  0.000,  0.000,  1.000]
    ])
    calibration_data = {
        "camera_to_base_transform": T_cam_to_base,
        "eef_to_checkboard_offset": np.eye(4),
        "intrinsic_calibration": None,
    }
    try:
        # pkg_path = rospkg.RosPack().get_path('thermal_camera')
        pkg_path = "/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/thermal_camera"
        backup_file = os.path.join(pkg_path, 'calibration_data/extrinsic_calib_realsense/camera_params.pkl')
    except:
        backup_file = 'calibration_data/extrinsic_calib_realsense/camera_params.pkl'
    
    with open(backup_file, 'wb') as f:
        pickle.dump(calibration_data, f)
    print("calibration data written to", backup_file)

if __name__ == "__main__":
    if sys.argv[1] == "r":
        read_pkl()
    elif sys.argv[1] == "w":
        write_pkl()
    else:
        print("Invalid syntax. Usage: python3 read_write_pkl.py [w/r]")
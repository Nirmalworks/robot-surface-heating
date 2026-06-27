import numpy as np
import cv2
from scipy.optimize import least_squares
import pickle
import rclpy
from rclpy.node import Node
import os
from ament_index_python.packages import get_package_share_directory
from scipy.spatial.transform import Rotation as R
from datetime import datetime

class DataNode(Node):
    """Node for extracting parameters in ROS config system."""

    def __init__(self):
        super().__init__("ba_data_node")

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

        # store package path
        self.pkg_path = '/'.join(get_package_share_directory('thermal_camera').split('/')[:-4]+['src','thermal_camera'])
        self.get_logger().info(f"pkg path: {self.pkg_path}")

def pose_matrix_to_rvec_tvec(T):
    """Convert 4x4 homogeneous transformation matrix to OpenCV-style rvec, tvec."""
    # If T is T_world_cam, we need T_cam_world = inv(T)
    T_inv: np.ndarray = np.linalg.inv(T)

    R = T_inv[:3, :3]
    t = T_inv[:3, 3].reshape(3, 1)

    rvec, _ = cv2.Rodrigues(R)
    return rvec, t

def load_intrinsic_parameters(
    pkg_path: str,
    num_cam: int, 
    intrinsics_dir_pfx: str
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Load intrinsic calibration results: Camera Intrinsic Matrix
    and Distortion Coefficients."""
    try:
        intrinsic_files = [
            os.path.join(pkg_path, f'calibration_data/intrinsic/{intrinsics_dir_pfx}{i}', 'thermal_intrinsics.npz')
            for i in range(num_cam)
        ]
        
        loaded = False
        camera_matrix_list: list[np.ndarray] = []
        dist_coeffs_list: list[np.ndarray] = []
        rms_list: list[float] = []
        for filepath in intrinsic_files:
            if os.path.exists(filepath):
                data = np.load(filepath)
                camera_matrix_list.append(data['mtx'])
                dist_coeffs_list.append(data['dist'])

                rms_list.append(data['rms'])
                
                # print(f"✓ Loaded intrinsics from: {filepath}")
                # print(f"  RMS error: {rms_list[-1]:.3f} pixels")
                # print(f"  Camera matrix:\n{camera_matrix_list[-1]}")
                # print(f"  Distortion coeffs: {dist_coeffs_list[-1].flatten()}")
                # loaded = True
                # break
        loaded = True
        
        if not loaded:
            raise FileNotFoundError("No intrinsic calibration file found")
        
        print(f"✓ Loaded intrinsics from: {os.path.join(pkg_path, f'calibration_data/intrinsic/{intrinsics_dir_pfx}*')}")

        return camera_matrix_list, dist_coeffs_list, rms_list
            
    except Exception as e:
        print(f"✗ ERROR loading intrinsics: {e}")
        print("Run intrinsic calibration first!")
        raise

def load_extrinsic_parameters(
    pkg_path: str, 
    num_cam: int,
    extrinsics_path: str,
    extrinsics_prefix: str,
    extrinsics_file: str
) -> tuple[list[np.ndarray], dict[str, list]]:
    """Load extrinsic calibration parameters: camera world
    positions and orientations (as 4x4 Transform matrices)."""

    camera_transforms: list[np.ndarray] = []
    eef_tool_offs: list[np.ndarray] = []
    num_samples: list[int] = []
    for i in range(num_cam):
        file_path = os.path.join(pkg_path, extrinsics_path, f'{extrinsics_prefix}{i}', f'{extrinsics_file}.pkl')
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"✗ Extrinsics file not found: {file_path}")

            # get camera transform and convert rotation to quaternion
            with open(file_path, 'rb') as f:
                calibration_data = pickle.load(f)
            t = calibration_data['camera_to_base_transform']
            # quat = R.from_matrix(t[:3,:3]).as_quat()
            # camera_transforms.append([t[:3,3],quat])
            camera_transforms.append(t)

            eef_tool_offs.append(calibration_data['eef_tool_translation'])
            num_samples.append(calibration_data['metadata']['num_poses'])

        except Exception as e:
            raise RuntimeError(f"✗ Error loading camera transforms: {e}")
    
    print("Loaded all camera transforms.")

    return [
        pose_matrix_to_rvec_tvec(T)
        for T in camera_transforms  # list of 4x4 matrices
    ], {"eef_tool_translation": eef_tool_offs, "num_poses": num_samples}

def load_data_samples(
    pkg_path: str,
    num_cam: int,
    extrinsics_prefix: str,
    calibration_backup_file: str,
    extrinsic_params: list[np.ndarray]
) -> dict[str, np.ndarray]:
    """Loads calibration data from a backup file,
    process it and store in a dictionary along
    with the cameras' extrinsic parameters."""

    try:
        # Default path from project structure
        file_path = os.path.join(pkg_path, f'calibration_data/extrinsic/{extrinsics_prefix}{num_cam-1}/{calibration_backup_file}.pkl')

        if not os.path.exists(file_path):
            print(f"✗ Calibration data file not found: {file_path}")
            return None

        with open(file_path, 'rb') as f:
            calibration_data = pickle.load(f)

        print(f"✓ Loaded calibration data from {file_path}")
        # return calibration_data

    except Exception as e:
        print(f"✗ Error loading calibration data: {e}")
        return None

    # Step 1: Gather all observations
    camera_indices = []
    point_indices = []
    points_2d = []
    points_3d = []

    point_id_map = {}   # Map from 3D point tuple to a unique index
    next_point_idx = 0

    for cam_idx, cam_data in enumerate(calibration_data):
        for obs in cam_data:
            pt3d = tuple(obs['3d_pt'].tolist())  # Convert to hashable
            pt2d = obs['2d_pt']

            if pt3d not in point_id_map:
                point_id_map[pt3d] = next_point_idx
                points_3d.append(np.array(pt3d))
                next_point_idx += 1

            point_idx = point_id_map[pt3d]
            camera_indices.append(cam_idx)
            point_indices.append(point_idx)
            points_2d.append(pt2d)


    camera_params = np.array([
        np.hstack([rvec.flatten(), tvec.flatten()])
        for rvec, tvec in extrinsic_params
    ])
    points_3d = np.array(points_3d)
    points_2d = np.array(points_2d)
    camera_indices = np.array(camera_indices)
    point_indices = np.array(point_indices)
    x0 = np.hstack((camera_params.ravel(), points_3d.ravel()))

    return {
        "points_3d": points_3d,
        "points_2d": points_2d,
        "camera_idxs": camera_indices,
        "point_idxs": point_indices,
        "x0": x0 
    }

def project(points_3d, rvec, tvec, K, dist_coeffs):
    projected, _ = cv2.projectPoints(points_3d, rvec, tvec, K, dist_coeffs)
    return projected.reshape(-1, 2)

def bundle_adjustment_residual(
    params, n_cams, n_points, camera_indices, point_indices, points_2d, K, dc
):
    camera_params = params[:n_cams * 6].reshape((n_cams, 6))
    points_3d = params[n_cams * 6:].reshape((n_points, 3))
    residuals = []

    for i in range(len(points_2d)):
        cam_idx = camera_indices[i]
        pt_idx = point_indices[i]

        rvec = camera_params[cam_idx, :3]
        tvec = camera_params[cam_idx, 3:]

        pt3d = points_3d[pt_idx].reshape(1, 3)
        projected = project(pt3d, rvec, tvec, K[cam_idx], dc[cam_idx])[0]
        error = projected - points_2d[i]
        residuals.append(error)

    return np.concatenate(residuals)

def bundle_adjust(
    num_cams: int,
    K: np.ndarray,
    dc: np.ndarray,
    data: dict[str, np.ndarray],   
):
    """Performs bundle adjustment optimization using
    the initial camera estimate and calibration data samples.
    
    Arguments:
        num_cams (int): the number of cameras in the system.
        K (ndarray): the camera intrinsics matrix
        dc (ndarray): the camera distortion coefficients (kept fixed during optimization)
        data (dict): a dictionary containing the initial state `x0`, the collected
            points in 3d space `points_3d`, the projected image points `points_2d`, and
            indices for cameras and points in `camera_idx` and `point_idxs`
    
    Returns:
        The optimized camera poses in `(rvec, tvec)` form.
    """
    n_points = len(data["points_3d"])

    res = least_squares(
        bundle_adjustment_residual,
        data["x0"],
        verbose=2,
        method='trf',
        ftol=1e-4,
        x_scale='jac',
        args=(num_cams, n_points, data["camera_idxs"], data["point_idxs"], data["points_2d"], K, dc)
    )

    # Extract results
    optimized_camera_params = res.x[:num_cams * 6].reshape((num_cams, 6))
    optimized_points_3d = res.x[num_cams * 6:].reshape((n_points, 3))

    optimized_poses = [
        (cam[:3].reshape(3, 1), cam[3:].reshape(3, 1))
        for cam in optimized_camera_params
    ]

    return optimized_poses

def store_updated_extrinsics(
    pkg_path: str,
    opt_poses: list[tuple[np.ndarray, np.ndarray]],
    extrinsics_dir_pfx: str,
    extrinsics_file: str,
    Ks: list[np.ndarray],
    dcs: list[np.ndarray],
    rms_error: list[float],
    metadata: dict[str, list]
):
    """Stores the optimized camera poses
    in a PKL file."""

    for idx in range(len(opt_poses)):
        T = np.eye(4)
        T[:3,:3], _ = cv2.Rodrigues(opt_poses[idx][0])
        T[:3,3] = opt_poses[idx][1].flatten()

        package = {
            'camera_to_base_transform': T,
            'eef_tool_translation': metadata['eef_tool_translation'][idx],
            'intrinsic_calibration': {
                'camera_matrix': Ks[idx],
                'dist_coeffs': dcs[idx],
                'rms_error': rms_error[idx]
            },
            'metadata': {
                'method': 'opencv_hand_eye_tsai_with_bundle_adjustment',
                'num_poses': metadata['num_poses'][idx],
                'date': datetime.now().isoformat(),
            }
        }
        
        try:
            result_file = os.path.join(pkg_path, f'calibration_data/extrinsic/{extrinsics_dir_pfx}{idx}/{extrinsics_file}_BA.pkl')
            print(f"Saving extrinsic results for Camera {idx} to {result_file}")
        except:
            print("Result file not found.")
        
        try:
            with open(result_file, 'wb') as f:
                pickle.dump(package, f)
            print("Save success")
        except Exception as e:
            print(f"Saving failed: {e}")

def main():
    rclpy.init()
    dn = DataNode()
    K, dc, rms = load_intrinsic_parameters(dn.pkg_path, dn.camera_count, dn.intrinsics_dir_pfx)
    init_cam_poses, metadata = load_extrinsic_parameters(
        dn.pkg_path,
        dn.camera_count,
        "calibration_data/extrinsic/",
        dn.extrinsics_dir_pfx,
        dn.extrinsics_file
    )
    data = load_data_samples(
        dn.pkg_path,
        dn.camera_count,
        dn.extrinsics_dir_pfx,
        dn.calibration_backup_file,
        init_cam_poses
    )
    opt_poses = bundle_adjust(dn.camera_count, K, dc, data)
    print(opt_poses)
    store_updated_extrinsics(
        dn.pkg_path,
        opt_poses,
        dn.extrinsics_dir_pfx,
        dn.extrinsics_file,
        K,
        dc,
        rms,
        metadata
    )
    rclpy.shutdown()

if __name__ == "__main__":
    main()
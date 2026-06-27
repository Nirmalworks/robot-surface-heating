#!/usr/bin/env python3
from ctypes import cdll
# initialize X11 threading support
cdll.LoadLibrary('libX11.so').XInitThreads()

import os
import subprocess
import pickle
import numpy as np
import multiprocessing
import threading
import time

import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from ament_index_python import get_package_share_directory

from urdfpy import URDF
import trimesh
import pyrender

# OpensGL flip for going from OpenCV frame (X right, Y down, Z forward)
# to OpenGL camera coords (X right, Y up, Z backward).
CV2GL = np.diag([1, -1, -1, 1])


class RobotViewer(Node):
    def __init__(self, camera_number, urdf_path, mesh_root, K, camera_to_base):
        super().__init__(f'urdf_robot_viewer_{camera_number}')
        self.camera_number = camera_number

        # — load robot & subscribe to joint_states
        self.robot = URDF.load(urdf_path)
        self.mesh_root = mesh_root
        self.joint_positions = {}
        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # — build pyrender scene
        self.scene = pyrender.Scene()
        self.link_nodes = {}
        init_tf = self.robot.link_fk()
        for link in self.robot.links:
            for vis in link.visuals or []:
                if vis.geometry.mesh is None:
                    continue
                rel = vis.geometry.mesh.filename
                mesh_path = rel if os.path.isabs(rel) else os.path.join(self.mesh_root, rel)
                try:
                    mesh = trimesh.load(mesh_path, force='mesh')
                    if vis.geometry.mesh.scale is not None:
                        mesh.apply_scale(vis.geometry.mesh.scale)
                    pose = init_tf.get(link, np.eye(4)) @ vis.origin
                    node = self.scene.add(
                        pyrender.Mesh.from_trimesh(mesh, smooth=False),
                        pose=pose
                    )
                    self.link_nodes[link.name] = (node, vis.origin)
                except Exception as e:
                    print(f"[WARN] unable to load mesh {mesh_path}: {e}")

        # — camera intrinsics
        fx, fy = K[0,0], K[1,1]
        cx, cy = K[0,2], K[1,2]
        camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=0.01, zfar=100.0)

        # — attach to scene
        cam_pose = camera_to_base @ CV2GL
        print(f"Using camera pose for camera {camera_number}:\n{cam_pose}")
        self.camera_node = self.scene.add(camera, pose=cam_pose)

        # — offscreen renderer + bridge + mask publisher + timer
        self.offscreen     = pyrender.OffscreenRenderer(640, 480)
        self.bridge        = CvBridge()
        self.mask_pub      = self.create_publisher(Image, 'simulated_robot_mask', 10)
        self.create_timer(1/30.0, self.publish_mask)  # 30 Hz

        # — kick off FK‐update loop
        threading.Thread(target=self.update_loop, daemon=True).start()

    def joint_state_callback(self, msg: JointState):
        self.joint_positions = dict(zip(msg.name, msg.position))

    def update_loop(self):
        while rclpy.ok():
            if self.joint_positions:
                tf_dict = self.robot.link_fk(cfg=self.joint_positions)
                for name, (node, origin) in self.link_nodes.items():
                    link = self.robot.link_map[name]
                    new_pose = tf_dict.get(link, np.eye(4)) @ origin
                    self.scene.set_pose(node, pose=new_pose)
            time.sleep(0.02)  # 50 Hz

    def publish_mask(self):
        # 1) render offscreen
        color, depth = self.offscreen.render(self.scene)

        # 2) create raw inverted mask so robot=255, background=0
        gray = cv2.cvtColor(color, cv2.COLOR_RGBA2GRAY)
        _, raw_inv = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)

        # 3) dilate the robot region by buffer_px pixels
        buffer_px = 30
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (2*buffer_px + 1, 2*buffer_px + 1)
        )
        dilated = cv2.dilate(raw_inv, kernel, iterations=1)

        # 4) invert back so robot=0, background=255
        mask = cv2.bitwise_not(dilated)

        # 5) debug: visualize before/after
        # cv2.imshow(f"Raw Mask {self.camera_number}", raw_inv) 
        # cv2.imshow(f"Padded Mask {self.camera_number}", mask)
        # cv2.waitKey(1)

        # 6) publish the buffered mask
        img_msg = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
        img_msg.header.stamp = self.get_clock().now().to_msg()
        self.mask_pub.publish(img_msg)

        # 7) show color frame for debugging
        bgr = cv2.cvtColor(color, cv2.COLOR_RGBA2BGR)
        cv2.imshow(f'Simulated Lepton {self.camera_number}', bgr)
        cv2.waitKey(1)


def run_viewer(cam_num, urdf_out, mesh_root):
    # load & scale intrinsics
    npz_path = (
        "/home/cam/robot_surface_heating_dev/"
        "robot_surface_heating_multi_flir/"
        "robot_surface_heating/"
        f"src/thermal_camera/"
        f"calibration_data/intrinsic/lepton{cam_num}/"
        "thermal_intrinsics.npz"
    )
    data = np.load(npz_path)
    K_raw = data["mtx"]
    scale_x, scale_y = 640.0/160.0, 480.0/120.0
    K = K_raw.copy()
    K[0,:] *= scale_x
    K[1,:] *= scale_y

    # load extrinsics
    pkl_path = (
        "/home/cam/robot_surface_heating_dev/"
        "robot_surface_heating_multi_flir/"
        "robot_surface_heating/"
        "src/thermal_camera/"
        f"calibration_data/extrinsic/coldest_node_{cam_num}/"
        "extrinsics.pkl"
    )
    raw = pickle.load(open(pkl_path, "rb"))
    if isinstance(raw, dict) and "camera_to_base_transform" in raw:
        camera_to_base = raw["camera_to_base_transform"]
    else:
        raise RuntimeError("Missing camera_to_base_transform")

    # launch node
    rclpy.init()
    node = RobotViewer(cam_num, urdf_out, mesh_root, K, camera_to_base)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.offscreen.delete()  # clean up GL context
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


def main():
    # 1) Expand Xacro → URDF
    pkg = "my_robot_cell_description"
    pkg_path = get_package_share_directory(pkg)
    xacro_file = os.path.join(pkg_path, "urdf", "my_robot_cell.urdf.xacro")
    urdf_out = "/tmp/my_robot_cell.urdf"
    subprocess.run(
        ["xacro", xacro_file, "ur_type:=ur10e"],
        stdout=open(urdf_out, "w"), check=True
    )

    # 2) Fix mesh paths
    mesh_root = (
        "/home/cam/robot_surface_heating_dev/"
        "robot_surface_heating_multi_flir/"
        "robot_surface_heating/"
        "src/Universal_Robots_ROS2_Description"
    )
    txt = open(urdf_out, "r").read()
    txt = txt.replace(
        "package://Universal_Robots_ROS2_Description/",
        mesh_root + "/"
    ).replace(
        "package://ur_description/",
        mesh_root + "/"
    )
    cell_pkg_root = get_package_share_directory("my_robot_cell_description")
    txt = txt.replace(
        "package://my_robot_cell_description",
        cell_pkg_root + "/"
    )
    open(urdf_out, "w").write(txt)

    # 3) Spawn one process per camera
    processes = []
    for cam_num in range(4):
        p = multiprocessing.Process(
            target=run_viewer,
            args=(cam_num, urdf_out, mesh_root),
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
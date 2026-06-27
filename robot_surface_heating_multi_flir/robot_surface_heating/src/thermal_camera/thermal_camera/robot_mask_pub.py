#!/usr/bin/env python3
from ctypes import cdll
cdll.LoadLibrary('libX11.so').XInitThreads()

import os
import subprocess
import pickle
import threading
import time

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image, JointState
from thermal_camera_interfaces.msg import RobotMask
from ament_index_python import get_package_share_directory

from urdfpy import URDF
import trimesh
import pyrender

# Flip from OpenCV to OpenGL coords
CV2GL = np.diag([1, -1, -1, 1])


class MaskPublisher(Node):
    def __init__(self, camera_infos, urdf_path, mesh_root):
        super().__init__('mask_publisher')

        self.get_logger().info('Waiting for joint states...')

        self.cycle_time = 1 / 100

        self.bridge = CvBridge()
        self.mask_topic = 'robot_mask'
        self.mask_pub = self.create_publisher(RobotMask, self.mask_topic, 10)
        self.camera_infos = camera_infos

        # Load URDF & build robot scene
        self.robot = URDF.load(urdf_path)
        self.scene = pyrender.Scene()
        self.link_nodes = {}
        init_tf = self.robot.link_fk()
        for link in self.robot.links:

            # self.get_logger().info(link.name)
            if link.name in ["base", "base_link", "base_link_inertia", "shoulder_link"]:
                # self.get_logger().warn(f"Skipping link: '{link.name}'")
                continue

            for vis in link.visuals or []:
                if vis.geometry.mesh is None:
                    continue
                rel = vis.geometry.mesh.filename
                mesh_path = rel if os.path.isabs(rel) else os.path.join(mesh_root, rel)
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
                    self.get_logger().warn(f"Unable to load mesh {mesh_path}: {e}")

        # Subscribe to joint states to update render
        self.joint_positions = {}
        self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )

        # One OffscreenRenderer for all cameras
        self.offscreen = pyrender.OffscreenRenderer(640, 480)

        # Start FK update thread
        threading.Thread(target=self.update_loop, daemon=True).start()

        # Timer to build & publish all masks
        self.create_timer(self.cycle_time, self.publish_all_masks)


    def joint_state_callback(self, msg: JointState):
        if self.joint_positions == {}:
            self.get_logger().info(f"Reading joint states. Publishing masks to /{self.mask_topic}")

        self.joint_positions = dict(zip(msg.name, msg.position))


    def update_loop(self):
        while rclpy.ok():
            if self.joint_positions:
                tf_dict = self.robot.link_fk(cfg=self.joint_positions)
                for name, (node, origin) in self.link_nodes.items():
                    link     = self.robot.link_map[name]
                    new_pose = tf_dict.get(link, np.eye(4)) @ origin
                    self.scene.set_pose(node, pose=new_pose)
            time.sleep(0.02)


    def publish_all_masks(self):
        msg   = RobotMask()
        masks = []

        for idx, (K, cam_to_base) in enumerate(self.camera_infos):
            # 1) temporarily add this camera to the scene
            fx, fy = K[0,0], K[1,1]
            cx, cy = K[0,2], K[1,2]
            cam_node = self.scene.add(
                pyrender.IntrinsicsCamera(fx, fy, cx, cy, znear=0.01, zfar=100.0),
                pose=cam_to_base @ CV2GL
            )

            # 2) render offscreen
            color, _ = self.offscreen.render(self.scene)

            # 3) remove that camera so next iteration stays clean
            self.scene.remove_node(cam_node)

            # DEBUG: show the virtual camera image
            # bgr = cv2.cvtColor(color, cv2.COLOR_RGBA2BGR)
            # cv2.imshow(f"Simulated Lepton {idx}", bgr)
            # cv2.waitKey(1)

            # 4) build your mask with 30px buffer
            buffer = 40 # pixel buffer around robot
            kernel_buffer = 2 * buffer + 1
            gray    = cv2.cvtColor(color, cv2.COLOR_RGBA2GRAY)
            _, inv  = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
            kernel  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_buffer, kernel_buffer))
            dilated = cv2.dilate(inv, kernel, iterations=1)
            mask    = cv2.bitwise_not(dilated)

            # # DEBUG: show mask
            # cv2.imshow(f"Debug Mask {idx}", mask)
            # cv2.waitKey(1)

            # 5) convert to ROS Image and stamp it
            img_msg         = self.bridge.cv2_to_imgmsg(mask, encoding='mono8')
            img_msg.header.stamp = self.get_clock().now().to_msg()
            masks.append(img_msg)

        # publish the bundle
        msg.masks = masks
        msg.header.stamp = self.get_clock().now().to_msg()
        self.mask_pub.publish(msg)


def main():
    rclpy.init()

    # 1) Expand Xacro → URDF
    pkg      = "my_robot_cell_description"
    pkg_path = get_package_share_directory(pkg)
    xacro    = os.path.join(pkg_path, "urdf", "my_robot_cell.urdf.xacro")
    urdf_out = "/tmp/my_robot_cell.urdf"
    subprocess.run(
        ["xacro", xacro, "ur_type:=ur10e"],
        stdout=open(urdf_out, "w"), check=True
    )

    # 2) Fix mesh paths in that URDF
    mesh_root = (
        "/home/cam/robot_surface_heating_dev/"
        "robot_surface_heating_multi_flir/"
        "robot_surface_heating/"
        "src/Universal_Robots_ROS2_Description"
    )
    txt = open(urdf_out).read()
    txt = txt.replace(
        "package://Universal_Robots_ROS2_Description/",
        mesh_root + "/"
    ).replace(
        "package://ur_description/",
        mesh_root + "/"
    )
    cell_pkg = get_package_share_directory("my_robot_cell_description")
    txt = txt.replace(
        "package://my_robot_cell_description",
        cell_pkg + "/"
    )
    open(urdf_out, "w").write(txt)

    # 3) Load intrinsics & extrinsics for each camera
    camera_infos = []
    for cam_num in range(4):
        # intrinsics
        intr = (
            "/home/cam/robot_surface_heating_dev/"
            "robot_surface_heating_multi_flir/"
            "robot_surface_heating/"
            "src/thermal_camera/"
            f"calibration_data/intrinsic/lepton{cam_num}/"
            "thermal_intrinsics.npz"
        )
        data = np.load(intr)
        K_raw = data["mtx"]
        sx, sy = 640/160, 480/120
        K = K_raw.copy()
        K[0,:] *= sx; K[1,:] *= sy

        # extrinsics
        ext = (
            "/home/cam/robot_surface_heating_dev/"
            "robot_surface_heating_multi_flir/"
            "robot_surface_heating/"
            "src/thermal_camera/"
            # f"calibration_data/extrinsic/dummy_{cam_num}/"
            f"calibration_data/extrinsic/03_03_26_calibration_{cam_num}/"
            "extrinsics.pkl"
        )
        raw = pickle.load(open(ext, "rb"))
        if isinstance(raw, dict) and "camera_to_base_transform" in raw:
            T = raw["camera_to_base_transform"]
        else:
            raise RuntimeError("Missing camera_to_base_transform")

        camera_infos.append((K, T))

    # 4) Start the node
    node = MaskPublisher(camera_infos, urdf_out, mesh_root)
    rclpy.spin(node)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

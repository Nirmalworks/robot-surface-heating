import os
import subprocess
import numpy as np
from ament_index_python.packages import get_package_share_directory
from urdfpy import URDF
import trimesh
import pyrender
import matplotlib.pyplot as plt

# === Step 1: Locate URDF package ===
urdf_package = "my_robot_cell_description"
urdf_package_path = get_package_share_directory(urdf_package)

xacro_file_path = os.path.join(urdf_package_path, "urdf", "my_robot_cell.urdf.xacro")
urdf_output_path = "/tmp/my_robot_cell.urdf"

# === Step 2: Convert Xacro to URDF ===
xacro_args = ['ur_type:=ur10e']
xacro_command = ["xacro", xacro_file_path] + xacro_args
with open(urdf_output_path, "w") as urdf_file:
    subprocess.run(xacro_command, stdout=urdf_file, check=True)

# === Step 3: Rewrite 'package://' paths ===
with open(urdf_output_path, "r") as f:
    urdf_text = f.read()

# Replace with your actual path
fallback_root = "/home/cam/robot_surface_heating_dev/robot_surface_heating_multi_flir/robot_surface_heating/src/Universal_Robots_ROS2_Description/"
urdf_text = urdf_text.replace("package://Universal_Robots_ROS2_Description/", fallback_root)
urdf_text = urdf_text.replace("package://ur_description/", fallback_root)

with open(urdf_output_path, "w") as f:
    f.write(urdf_text)

# === Step 4: Load robot ===
robot = URDF.load(urdf_output_path)

joint_positions = {
    "shoulder_pan_joint": 0.0,
    "shoulder_lift_joint": -1.8,
    "elbow_joint": -1.8,
    "wrist_1_joint": -1.12,
    "wrist_2_joint": 1.596,
    "wrist_3_joint": -1.732,
}

link_transforms = robot.link_fk(cfg=joint_positions)

# === Step 5: Build scene ===
scene = pyrender.Scene()

for link in robot.links:
    if link.visuals:
        for visual in link.visuals:
            geometry = visual.geometry
            if geometry.mesh:
                mesh_path = geometry.mesh.filename
                if not os.path.isabs(mesh_path):
                    mesh_path = os.path.join(fallback_root, mesh_path)

                try:
                    mesh = trimesh.load(mesh_path, force='mesh')

                    link_tf = link_transforms.get(link, np.eye(4))
                    mesh.apply_transform(link_tf @ visual.origin)

                    scene.add(pyrender.Mesh.from_trimesh(mesh))
                except Exception as e:
                    print(f"[WARN] Could not load mesh: {mesh_path}")
                    print(f"       Reason: {e}")

# Optional: show base frame
axis = trimesh.creation.axis(origin_size=0.02)
scene.add(pyrender.Mesh.from_trimesh(axis, smooth=False))

# === Step 6: Add LIGHTING (comment this block out to render without colors) ===
light = pyrender.DirectionalLight(color=np.ones(3), intensity=3.0)
light_pose = np.eye(4)
light_pose[:3, 3] = [1.0, 1.0, 1.0]  # light position
scene.add(light, pose=light_pose)

# === Step 7: Add VIRTUAL CAMERA from OVERHEAD ANGLE ===
camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)

def look_at(cam_pos, target, up):
    forward = (target - cam_pos)
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    true_up = np.cross(right, forward)

    view = np.eye(4)
    view[:3, 0] = right
    view[:3, 1] = true_up
    view[:3, 2] = -forward
    view[:3, 3] = cam_pos
    return view

robot_center = np.array([0.0, 0.0, 0.5])
camera_pos = robot_center + np.array([1, -1, 0.2])  # above, front, right
camera_pose = look_at(camera_pos, robot_center, np.array([0.0, 0.0, 1.0]))
scene.add(camera, pose=camera_pose)

# === Step 8: Render camera view to image ===
r = pyrender.OffscreenRenderer(viewport_width=800, viewport_height=600)
color, _ = r.render(scene)

# === Step 9: Show using matplotlib ===
plt.imshow(color)
plt.title("Virtual Camera View")
plt.axis("off")
plt.show()

#!/usr/bin/env python3

import numpy as np
import rclpy

import time
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point


CAD_POINTCLOUD_TOPIC = "/cad_pointcloud"
MARKER_TOPIC = "/debug_surface_normals"

STRIDE = 1
NORMAL_SCALE_M = 0.03
LINE_WIDTH_M = 0.002

COLOR_MODE = "z_sign"   # "z_sign" or "single_color"

# single_color mode
COLOR_SINGLE = (0.0, 1.0, 0.0, 1.0)

# z_sign mode
COLOR_POS = (0.0, 1.0, 0.0, 1.0)   # green for nz >= 0
COLOR_NEG = (1.0, 0.0, 0.0, 1.0)   # red for nz < 0


def normalize_rows(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    out = arr.copy()
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    good = norms[:, 0] > eps
    out[good] /= norms[good]
    return out


class DebugSurfaceNormalsNode(Node):
    def __init__(self):
        super().__init__("debug_surface_normals")

        self.sub = self.create_subscription(
            PointCloud2,
            CAD_POINTCLOUD_TOPIC,
            self.pointcloud_callback,
            10,
        )

        self.pub = self.create_publisher(MarkerArray, MARKER_TOPIC, 10)

        self.get_logger().info(f"Subscribed to: {CAD_POINTCLOUD_TOPIC}")
        self.get_logger().info(f"Publishing markers to: {MARKER_TOPIC}")

    def pointcloud_callback(self, msg: PointCloud2):
        pts = []
        nrms = []

        for p in point_cloud2.read_points(
            msg,
            field_names=("x", "y", "z", "normal_x", "normal_y", "normal_z"),
            skip_nans=True,
        ):
            x, y, z, nx, ny, nz = p
            pts.append([x, y, z])
            nrms.append([nx, ny, nz])

        if len(pts) == 0:
            self.get_logger().warn("Received empty cad_pointcloud.")
            return

        pts = np.asarray(pts, dtype=np.float64)
        nrms = np.asarray(nrms, dtype=np.float64)
        nrms = normalize_rows(nrms)

        good = np.linalg.norm(nrms, axis=1) > 1e-12
        pts = pts[good]
        nrms = nrms[good]

        if len(pts) == 0:
            self.get_logger().warn("No valid normals found in cad_pointcloud.")
            return

        # --------------------------------------------------------
        # Positive-slope enforcement:
        # make every normal point with positive Z in the cloud frame
        # --------------------------------------------------------
        flip_mask = nrms[:, 1] < 0.0
        nrms[flip_mask] *= -1.0

        idx = np.arange(0, len(pts), STRIDE, dtype=np.int32)
        pts_s = pts[idx]
        nrms_s = nrms[idx]

        marker_array = MarkerArray()

        delete_marker = Marker()
        delete_marker.header = msg.header
        delete_marker.ns = "surface_normals"
        delete_marker.id = 0
        delete_marker.action = Marker.DELETEALL
        marker_array.markers.append(delete_marker)

        line_marker = Marker()
        line_marker.header = msg.header
        line_marker.ns = "surface_normals"
        line_marker.id = 1
        line_marker.type = Marker.LINE_LIST
        line_marker.action = Marker.ADD
        line_marker.scale.x = LINE_WIDTH_M
        line_marker.pose.orientation.w = 1.0

        for p, n in zip(pts_s, nrms_s):
            p0 = Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
            p1 = Point(
                x=float(p[0] + NORMAL_SCALE_M * n[0]),
                y=float(p[1] + NORMAL_SCALE_M * n[1]),
                z=float(p[2] + NORMAL_SCALE_M * n[2]),
            )

            line_marker.points.append(p0)
            line_marker.points.append(p1)

            # After enforcement, everything should be green unless nz == 0 edge cases
            if n[2] >= 0.0:
                color = COLOR_POS
            else:
                color = COLOR_NEG

            for _ in range(2):
                c = Marker().color
                c.r = float(color[0])
                c.g = float(color[1])
                c.b = float(color[2])
                c.a = float(color[3])
                line_marker.colors.append(c)

        marker_array.markers.append(line_marker)
        self.pub.publish(marker_array)

        n_z = nrms[:, 2]
        n_up = int(np.sum(n_z >= 0.0))
        n_down = int(np.sum(n_z < 0.0))
        n_flipped = int(np.sum(flip_mask))

        self.get_logger().info(
            f"Published {len(pts_s)} normal markers "
            f"(sampled from {len(pts)} points, stride={STRIDE}, "
            f"flipped={n_flipped}, up={n_up}, down={n_down}, frame={msg.header.frame_id})"
        )

        time.sleep(1)


def main(args=None):
    rclpy.init(args=args)
    node = DebugSurfaceNormalsNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from visualization_msgs.msg import Marker
import matplotlib.pyplot as plt
import time

MAX_DURATION_SEC = 30.0  # Size of the moving time window
TRACKED_INDEX = 100       # <--- Set your desired point index here

class SelectedPointTempPlotter(Node):
    def __init__(self):
        super().__init__('selected_temp_plotter')
        self.subscription = self.create_subscription(
            PointCloud2,
            '/selected_surface_points',
            self.callback,
            10
        )

        self.marker_pub = self.create_publisher(Marker, '/heatmap_markers', 10)

        self.timestamps = []
        self.temperatures = []
        self.paused = False

        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [], label=f"Point {TRACKED_INDEX} Temp (°C)")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Temperature (°C)")
        self.ax.set_ylim(20, 80)  # Fixed vertical axis range
        self.ax.grid(True)
        self.ax.legend()

        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)
        plt.ion()
        plt.show()

    def on_key_press(self, event):
        if event.key == "p":
            self.paused = not self.paused
            print("Paused" if self.paused else "Resumed")

    def callback(self, msg: PointCloud2):
        if self.paused:
            return

        try:
            points = list(point_cloud2.read_points(
                msg, field_names=("x", "y", "z", "thermal"), skip_nans=True
            ))
        except Exception as e:
            self.get_logger().warn(f"PointCloud2 read failed: {e}")
            return

        if len(points) <= TRACKED_INDEX:
            self.get_logger().warn(f"Index {TRACKED_INDEX} out of range ({len(points)} points).")
            return

        x, y, z, temp_k = points[TRACKED_INDEX]
        temp_c = (temp_k - 27315) / 100.0  # Convert raw K*100 to °C

        now = time.time()
        self.timestamps.append(now)
        self.temperatures.append(temp_c)

        # Keep only last MAX_DURATION_SEC seconds of data
        while self.timestamps and (now - self.timestamps[0] > MAX_DURATION_SEC):
            self.timestamps.pop(0)
            self.temperatures.pop(0)

        relative_times = [t - self.timestamps[0] for t in self.timestamps]

        self.line.set_data(relative_times, self.temperatures)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.draw()
        plt.pause(0.001)

        self.publish_highlight_marker(x, y, z, msg.header.frame_id)

    def publish_highlight_marker(self, x, y, z, frame_id):
        # Sphere marker
        sphere = Marker()
        sphere.header.frame_id = frame_id
        sphere.header.stamp = self.get_clock().now().to_msg()
        sphere.ns = "tracked_point"
        sphere.id = 0
        sphere.type = Marker.SPHERE
        sphere.action = Marker.ADD
        sphere.pose.position.x = float(x)
        sphere.pose.position.y = float(y)
        sphere.pose.position.z = float(z)
        sphere.pose.orientation.w = 1.0
        sphere.scale.x = 0.02
        sphere.scale.y = 0.02
        sphere.scale.z = 0.02
        sphere.color.r = 1.0
        sphere.color.g = 1.0
        sphere.color.b = 0.0
        sphere.color.a = 1.0
        sphere.lifetime.sec = 1

        # Text marker
        text = Marker()
        text.header.frame_id = frame_id
        text.header.stamp = self.get_clock().now().to_msg()
        text.ns = "tracked_point"
        text.id = 1
        text.type = Marker.TEXT_VIEW_FACING
        text.action = Marker.ADD
        text.pose.position.x = float(x)
        text.pose.position.y = float(y)
        text.pose.position.z = float(z) + 0.03  # Slightly above
        text.pose.orientation.w = 1.0
        text.scale.z = 0.05  # Text height
        text.color.r = 1.0
        text.color.g = 1.0
        text.color.b = 1.0
        text.color.a = 1.0
        temp_c = self.temperatures[-1] if self.temperatures else 0.0
        text.text = f"{temp_c:.1f}C"
        text.lifetime.sec = 1

        self.marker_pub.publish(sphere)
        self.marker_pub.publish(text)

def main(args=None):
    rclpy.init(args=args)
    node = SelectedPointTempPlotter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("Shutting down.")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        plt.ioff()
        plt.show()

if __name__ == '__main__':
    main()
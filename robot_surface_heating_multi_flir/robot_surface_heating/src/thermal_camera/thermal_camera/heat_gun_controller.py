#!/usr/bin/env python3

import time
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray
import serial


class HeatgunFeedbackSerialNode(Node):
    def __init__(self) -> None:
        super().__init__("heatgun_feedback_serial_node")

        # Runtime-tunable ROS parameters
        self.declare_parameter(
            "serial_port",
            "/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_4423831323935120C011-if00",
        )
        self.declare_parameter("baud_rate", 9600)

        self.declare_parameter("target_low_C", 40.0)
        self.declare_parameter("target_high_C", 50.0)
        self.declare_parameter("target_percent", 90.0)
        # self.declare_parameter("enable_mode_switching", False)

        # ROS interfaces
        self.serial_port = self.get_parameter("serial_port").value
        self.baud_rate = int(self.get_parameter("baud_rate").value)

        self.metrics_topic = "/thermal_roi_metrics"
        self.percent_topic = "/selected_surface_percent_in_band"

        # Experiment settings
        self.target_low_C = float(self.get_parameter("target_low_C").value)
        self.target_high_C = float(self.get_parameter("target_high_C").value)
        self.target_percent = float(self.get_parameter("target_percent").value)
        # self.enable_mode_switching = bool(
        #     self.get_parameter("enable_mode_switching").value
        # )

        # True for large and contour heating tests
        # False for local heating tests
        self.enable_mode_switching = True

        # Fixed heater behavior
        self.ramp_heater_value = 60.0
        self.heater_min = 40.0
        self.heater_max = 60.0

        # Fixed safety behavior
        self.overheat_margin_C = 3.0
        self.overheat_reset_margin_C = 1.0
        self.hard_shutdown_C = 60.0

        # Fixed smoothing / serial update behavior
        self.filter_alpha = 0.25
        self.publish_period_s = 0.5
        self.min_command_change = 1.0

        # Serial state
        self.ser: Optional[serial.Serial] = None

        # Sensor/control state
        self.filtered_max_C: Optional[float] = None
        self.filtered_median_C: Optional[float] = None
        self.latest_percent_in_band: Optional[float] = None

        self.overheat_latched = False
        self.last_sent_value: Optional[float] = None
        self.last_send_time = 0.0

        self.mode = "RAMP_UP" if self.enable_mode_switching else "MAINTENANCE"

        self.connect_serial()

        self.metrics_sub = self.create_subscription(
            Float64MultiArray,
            self.metrics_topic,
            self.metrics_callback,
            10,
        )

        self.percent_sub = self.create_subscription(
            Float64,
            self.percent_topic,
            self.percent_callback,
            10,
        )

        self.get_logger().info(
            f"Target band: {self.target_low_C:.1f}-{self.target_high_C:.1f} C | "
            f"target percent: {self.target_percent:.1f}% | "
            f"mode: {self.mode}"
        )

    def connect_serial(self) -> None:
        self.get_logger().info(
            f"Opening serial port {self.serial_port} at {self.baud_rate} baud"
        )

        self.ser = serial.Serial(
            self.serial_port,
            self.baud_rate,
            timeout=0.2,
        )

        time.sleep(2.0)

        self.get_logger().info("Connected to Arduino")
        self.drain_initial_output()

    def drain_initial_output(self) -> None:
        if self.ser is None:
            return

        end_time = time.time() + 2.0
        while time.time() < end_time:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode(errors="ignore").strip()
                if line:
                    self.get_logger().info(f"Arduino: {line}")

    def percent_callback(self, msg: Float64) -> None:
        self.latest_percent_in_band = float(msg.data)

        if (
            self.enable_mode_switching
            and self.mode == "RAMP_UP"
            and self.latest_percent_in_band >= self.target_percent
        ):
            self.mode = "MAINTENANCE"
            self.get_logger().info(
                f"Reached {self.latest_percent_in_band:.1f}% in band. "
                f"Switching to MAINTENANCE mode."
            )

    def metrics_callback(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < 2:
            self.get_logger().warn(
                "Expected /thermal_roi_metrics data = [max_temp_C, median_temp_C]"
            )
            return

        raw_max_C = float(msg.data[0])
        raw_median_C = float(msg.data[1])

        self.update_filtered_temperatures(raw_max_C, raw_median_C)

        if self.filtered_max_C is None or self.filtered_median_C is None:
            return

        command = self.compute_heater_command(
            raw_max_C=raw_max_C,
            filtered_max_C=self.filtered_max_C,
            filtered_median_C=self.filtered_median_C,
        )

        self.send_value_if_needed(command)

    def update_filtered_temperatures(self, raw_max_C: float, raw_median_C: float) -> None:
        alpha = max(0.0, min(1.0, self.filter_alpha))

        if self.filtered_max_C is None:
            self.filtered_max_C = raw_max_C
        else:
            self.filtered_max_C = alpha * raw_max_C + (1.0 - alpha) * self.filtered_max_C

        if self.filtered_median_C is None:
            self.filtered_median_C = raw_median_C
        else:
            self.filtered_median_C = alpha * raw_median_C + (1.0 - alpha) * self.filtered_median_C

    def compute_heater_command(
        self,
        *,
        raw_max_C: float,
        filtered_max_C: float,
        filtered_median_C: float,
    ) -> float:
        shutoff_C = self.target_high_C + self.overheat_margin_C
        reset_C = self.target_high_C + self.overheat_reset_margin_C

        if raw_max_C >= self.hard_shutdown_C:
            self.overheat_latched = True
            return 0.0

        if self.overheat_latched:
            if filtered_max_C <= reset_C:
                self.overheat_latched = False
            else:
                return 0.0

        if filtered_max_C >= shutoff_C:
            self.overheat_latched = True
            return 0.0

        if self.enable_mode_switching and self.mode == "RAMP_UP":
            return float(self.ramp_heater_value)

        return self.compute_dynamic_command(filtered_median_C)

    def compute_dynamic_command(self, filtered_median_C: float) -> float:
        band_width = max(self.target_high_C - self.target_low_C, 1e-6)
        ratio = (filtered_median_C - self.target_low_C) / band_width
        ratio = max(0.0, min(1.0, ratio))

        command = self.heater_max - ratio * (self.heater_max - self.heater_min)
        command = max(self.heater_min, min(self.heater_max, command))

        return float(command)

    def send_value_if_needed(self, value: float) -> None:
        now = time.time()

        should_send = False

        if self.last_sent_value is None:
            should_send = True
        elif abs(value - self.last_sent_value) >= self.min_command_change:
            should_send = True
        elif now - self.last_send_time >= self.publish_period_s:
            should_send = True

        if not should_send:
            return

        self.send_value(value)
        self.last_sent_value = value
        self.last_send_time = now

        percent_text = (
            f"{self.latest_percent_in_band:.1f}%"
            if self.latest_percent_in_band is not None
            else "none"
        )

        self.get_logger().info(
            f"percent_in_band={percent_text} | "
            f"mode={self.mode} | "
            f"heater={value:.1f} | "
            f"max={self.filtered_max_C:.1f} C | "
            f"median={self.filtered_median_C:.1f} C | "
            f"overheat_latched={self.overheat_latched}"
        )

    def send_value(self, value: float) -> None:
        if self.ser is None or not self.ser.is_open:
            self.get_logger().error("Serial port is not open")
            return

        if value != 0.0 and not (30.0 <= value <= 60.0):
            self.get_logger().warn(f"Invalid heater value blocked: {value:.1f}")
            return

        command = f"{value:.1f}\n"
        self.ser.write(command.encode("utf-8"))
        self.ser.flush()

        self.read_available_responses()

    def read_available_responses(self) -> None:
        if self.ser is None:
            return

        start_time = time.time()
        while time.time() - start_time < 0.05:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode(errors="ignore").strip()
                if line:
                    self.get_logger().info(f"Arduino: {line}")
            else:
                break

    def destroy_node(self) -> None:
        try:
            self.send_value(0.0)
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass

        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HeatgunFeedbackSerialNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
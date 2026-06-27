#!/usr/bin/env python3

import time

import rclpy
from rclpy.node import Node
import serial


class HeatgunTerminalSerialNode(Node):
    def __init__(self) -> None:
        super().__init__("heatgun_terminal_serial_node")

        self.declare_parameter(
            "serial_port",
            "/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_4423831323935120C011-if00",
        )
        self.declare_parameter("baud_rate", 9600)

        self.serial_port = self.get_parameter("serial_port").value
        self.baud_rate = int(self.get_parameter("baud_rate").value)

        self.ser = None
        self.connect_serial()

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
        end_time = time.time() + 2.0
        while time.time() < end_time:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode(errors="ignore").strip()
                if line:
                    print(f"Arduino: {line}")

    def send_value(self, value: float) -> None:
        command = f"{value:.1f}\n"
        self.ser.write(command.encode("utf-8"))
        self.ser.flush()
        self.get_logger().info(f"Sent: {command.strip()}")

    def read_responses(self, duration_sec: float = 1.0) -> None:
        end_time = time.time() + duration_sec
        while time.time() < end_time and rclpy.ok():
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode(errors="ignore").strip()
                if line:
                    print(f"Arduino: {line}")

    def run_terminal_loop(self) -> None:
        print("Type heater value: 0 or 30-60. Type q to quit.")

        while rclpy.ok():
            try:
                user_input = input("Enter value: ").strip()

                if user_input.lower() in ["q", "quit", "exit"]:
                    break

                try:
                    value = float(user_input)
                except ValueError:
                    self.get_logger().warn("Invalid number")
                    continue

                if value != 0.0 and not (30.0 <= value <= 60.0):
                    self.get_logger().warn("Value must be 0 or between 30 and 60")
                    continue

                self.send_value(value)
                self.read_responses(1.0)

            except KeyboardInterrupt:
                break

    def destroy_node(self):
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()
        except Exception:
            pass
        super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HeatgunTerminalSerialNode()

    try:
        node.run_terminal_loop()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
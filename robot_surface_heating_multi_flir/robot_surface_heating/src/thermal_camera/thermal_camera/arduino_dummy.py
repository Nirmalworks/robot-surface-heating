import serial
import time

PORT = "/dev/serial/by-id/usb-Arduino__www.arduino.cc__0043_4423831323935120C011-if00"
BAUD = 9600

def main():
    print("Opening serial connection...")
    ser = serial.Serial(PORT, BAUD, timeout=1)

    # Arduino resets when serial opens
    time.sleep(2)

    print("Connected. Type values (0 or 30–60). Type 'q' to quit.\n")

    try:
        while True:
            user_input = input("Enter value: ").strip()

            if user_input.lower() in ["q", "quit", "exit"]:
                break

            try:
                value = float(user_input)
            except ValueError:
                print("Invalid number")
                continue

            # send to Arduino
            cmd = f"{value}\n"
            ser.write(cmd.encode("utf-8"))
            ser.flush()

            print(f"Sent: {cmd.strip()}")

            # read response for ~1 second
            t_end = time.time() + 1.0
            while time.time() < t_end:
                if ser.in_waiting > 0:
                    line = ser.readline().decode(errors="ignore").strip()
                    if line:
                        print(f"Arduino: {line}")

    finally:
        ser.close()
        print("Closed serial connection.")


if __name__ == "__main__":
    main()
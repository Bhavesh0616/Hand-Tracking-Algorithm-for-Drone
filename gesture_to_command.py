import time
from pymavlink import mavutil

# Initialize connection to the flight controller
drone = mavutil.mavlink_connection('COM3')  # Adjust for your setup
drone.wait_heartbeat()

# Function to set flight mode
def set_flight_mode(mode):
    print(f"Setting flight mode to {mode}")
    mode_id = drone.mode_mapping()[mode]  # Get the mode ID for the desired mode
    drone.mav.set_mode_send(
        drone.target_system,  # Target system
        0b00000000,  # 0 = 0, 1 = 1, 2 = 2, ..., 255 = 255
        mode_id  # Set to the desired mode ID
    )

# Set the flight mode to STABILIZE by default
set_flight_mode("STABILIZE")

# Function to send commands to the drone
def send_command_to_drone(command):
    if command == "arm":
        print("Sending ARM command to the drone")
        drone.mav.command_long_send(
            drone.target_system, drone.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 1, 0, 0, 0, 0, 0, 0
        )
    elif command == "disarm":
        print("Sending DISARM command to the drone")
        drone.mav.command_long_send(
            drone.target_system, drone.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
            0, 0, 0, 0, 0, 0, 0, 0
        )
    # Add more commands like takeoff, land, etc.

# Function to read the recognized gesture file in real-time (like tail -f)
def tail(f):
    f.seek(0, 2)  # Move to the end of the file
    while True:
        line = f.readline()
        if not line:
            time.sleep(0.1)  # Sleep briefly before retrying
            continue
        yield line.strip()

# Function to log gesture-command pairs to a file
def log_gesture_command(gesture, command):
    with open("gesture_command_log.txt", "a") as log_file:
        log_file.write(f"{gesture} - {command}\n")

# Main loop to read gestures and send corresponding drone commands
with open("recognized_gestures.txt", "r") as file:
    for gesture in tail(file):  # Real-time reading of the gesture file
        if gesture == "Close":
            send_command_to_drone("arm")
            log_gesture_command(gesture, "Arm")
        elif gesture == "Pointer":
            send_command_to_drone("disarm")
            log_gesture_command(gesture, "Disarm")
        # Add more gesture mappings here as needed

        time.sleep(1)  # Adjust the polling interval as needed

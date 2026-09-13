# Project: LeKiwi Educational Robotics Platform

## 1. Project Overview & Objectives
The goal of this project is to adapt the open-source LeKiwi robot (a 3-wheel omnidirectional mobile base paired with a lightweight manipulator arm) into an affordable, rugged, and reliable educational toy for middle school students. 

The system must be "plug-and-play" with a headless setup. Students should be able to connect to the robot, drive it, record teleoperation data, and replay autonomous actions to learn the basics of robotics and imitation learning.


## 3. Software & Control Architecture
The backend should be a lightweight web server (FastAPI/Flask) that auto-starts on boot.

### 3.1 Input Multiplexer (3 Teleop Modes)
The system must safely arbitrate between three control modes:
1. **Bluetooth Gamepad (Default):** 8BitDo or Xbox controller connected directly to the Pi's Bluetooth. Read via `evdev` or `pygame`. Left stick maps to base translation ($V_x, V_y$), right stick to base rotation ($\omega_z$) and arm pitch.
2. **Web UI Joystick:** Virtual joysticks sent via WebSockets from a browser (tablet, phone, or Chromebook). Overrides or blends with gamepad input.
3. **Leader Arm Teleoperation:** Physical leader arm (using potentiometers/encoders via serial) controls the follower arm joints directly. Base movement remains controlled by Web UI or Gamepad simultaneously.

### 3.2 Dataset Recording & Replay
* Must record state-action pairs at ~30Hz (timestamps, wheel velocities, joint positions) synchronized with camera frames (MJPEG/WebRTC stream).
* Output format should be compatible with **Hugging Face LeRobot** (`LeRobotDataset` format: Parquet + MP4).
* UI/Gamepad must have a physical/virtual toggle to Start/Stop recording and a "Replay Episode" function to immediately play back recorded actions autonomously.

### 3.3 Extensible Web Interface
The web UI should be modular (tabbed layout):
* **Tab 1: Drive & Video:** Real-time camera feed and virtual joysticks/mode toggles.
* **Tab 2: Recording / Datasets:** Episode management (record, save, replay).
* **Tab 3: Calibration & Diagnostics:** Joint offsets, servo temp monitoring, battery voltage.

## 4. Networking & Development Setup
To ensure a frictionless experience for students while maintaining internet access for AI agent development (Claude Code, GitHub), the Pi utilizes the following networking strategies via `NetworkManager`.

### 4.1 Production Field Mode (Student Access)
* The Pi's internal Wi-Fi (`wlan0`) hosts an isolated hotspot named `LeKiwi-RoboNet` with DHCP enabled.
* Students connect their devices to this hotspot and access the web UI via the Pi's gateway IP (e.g., `10.42.0.1`).

### 4.2 AI Agent Development Environments
For running Claude Code and pulling dependencies, the Pi requires external internet without disabling the hotspot. Three supported topologies:
1. **Dual Network (Recommended):** Pi connects to a router via Ethernet (`eth0`) OR a secondary USB Wi-Fi dongle (`wlan1`) for internet access. The internal Wi-Fi (`wlan0`) continues to broadcast the `LeKiwi-RoboNet` hotspot.
2. **Internet Connection Sharing:** Pi connects to the host PC via Ethernet. Host PC shares its Wi-Fi internet connection over the Ethernet interface.
3. **Software Toggle (`nmcli`):** If strictly limited to `wlan0`, use bash aliases to toggle the interface between Station mode (Home Wi-Fi) and AP mode (Hotspot) for testing.

## 5. Agent Directives (Claude Code Instructions)
* **Safety First:** Assume the robot is on a desk. Warn the user to prop the wheels off the ground before executing physical motor test scripts.
* **Incremental Implementation:** Start by mocking the servo outputs. Build the FastAPI + WebSocket multiplexer first, ensure the inputs register correctly from a web browser, and *then* connect the serial bus hardware driver.
* **Stateless UI:** Keep the frontend as vanilla as possible (HTML5, JS, WebSockets) served directly from the Python backend so users don't need to install Node/NPM to run the robot.

## 6. Use this repo
Do the development on host pc, and run testing on raspberry pi. 
# Target Tracking Demonstrator

**Project 23 · Joint ECE / ME / CS Senior Design**

A multidisciplinary embedded vision and control system for real-time target tracking. The planned system uses camera and LiDAR data to detect and track a target, coordinate tracking commands, and display live video and status to an operator.

## Where the software runs

| Folder | Platform | Purpose |
| --- | --- | --- |
| [`ros2_ws/src/`](ros2_ws/src/) | Jetson | Main ROS 2 software: operator coordination, perception, hardware interfaces, logging, and system startup. |
| [`ui_stack/`](ui_stack/) | Operator PC / laptop | The operator application for viewing video and tracking status, selecting targets, and sending commands. |
| [`firmware/`](firmware/) | External microcontroller / real-time controller | Motor-control firmware and position feedback for the ECE subsystem. |

## Hardware Used
HuskyLens2 - https://www.dfrobot.com/huskylens/huskylens2

**Current status:** this repository contains the folder scaffold. Each folder has a `placeholder.txt` file to preserve it in Git. The descriptions below explain the intended responsibilities; ROS 2 packages, the UI, and firmware have not been implemented yet.

## Setting up a Python virtual environment

A virtual environment ("venv") keeps this project's Python packages separate from the rest of your computer, so different projects don't conflict. Each developer should create one locally after cloning the repo.

You need **Python 3.10 or newer**. Check your version with `python3 --version` (macOS/Linux) or `py --version` (Windows). If Python is not installed, download it from [python.org](https://www.python.org/downloads/).

All commands below are run from the root of the repository (the `target-tracking-demonstrator/` folder).

### macOS

```bash
# Create the environment (only needed once)
python3 -m venv .venv

# Activate it (needed every time you open a new terminal)
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

### Linux (Ubuntu / Jetson)

```bash
# On Ubuntu/Debian, install the venv module first if it is missing
sudo apt update && sudo apt install python3-venv

# Create the environment (only needed once)
python3 -m venv .venv

# Activate it (needed every time you open a new terminal)
source .venv/bin/activate

# Upgrade pip
python -m pip install --upgrade pip
```

> **ROS 2 note:** On the Jetson, ROS 2 Python packages (such as `rclpy`) are installed system-wide by ROS, not through pip. If you use a venv alongside ROS 2, create it with `python3 -m venv --system-site-packages .venv` so those packages remain visible, and source ROS (`source /opt/ros/<distro>/setup.bash`) before activating the venv.

### Windows

In **PowerShell**:

```powershell
# Create the environment (only needed once)
py -m venv .venv

# Activate it (needed every time you open a new terminal)
.venv\Scripts\Activate.ps1

# Upgrade pip
python -m pip install --upgrade pip
```

If PowerShell says running scripts is disabled, run this once and then try activating again:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

In **Command Prompt (cmd.exe)**, activate with `.venv\Scripts\activate.bat` instead.

### Using the environment

- When the venv is active, your terminal prompt starts with `(.venv)`. Any `pip install` now installs into this project only.
- Once a `requirements.txt` is added to the project, install dependencies with `pip install -r requirements.txt`.
- To leave the environment, run `deactivate`.
- Don't commit the `.venv/` folder — it is specific to your machine. Add `.venv/` to `.gitignore`.
- In VS Code, open the Command Palette and choose **Python: Select Interpreter**, then pick the interpreter inside `.venv`.

## Folder layout

```text
target-tracking-demonstrator/
├── ros2_ws/
│   └── src/
│       ├── operator_stack/
│       │   ├── operator_gateway/
│       │   └── session_target_manager/
│       ├── perception_stack/
│       │   ├── detection_tracking/
│       │   └── fusion_prediction/
│       ├── integration_stack/
│       │   ├── sensor_interface/
│       │   └── control_interface/
│       ├── data_stack/
│       │   └── logging_diagnostics/
│       ├── interfaces/
│       │   ├── msg/
│       │   ├── srv/
│       │   └── action/
│       └── bringup/
│           ├── launch/
│           └── config/
├── ui_stack/
│   └── src/
│       ├── display_gui/
│       ├── input_commands/
│       └── gateway_client/
└── firmware/
    ├── arduino/
    ├── src/
    └── include/
```

Placeholder files are omitted from the tree for readability.

## Jetson software — `ros2_ws/src/`

This is where the main software stacks running on the Jetson belong. ROS 2 provides communication between them; each stack groups related components by responsibility.

| Folder | What belongs here |
| --- | --- |
| `operator_stack/` | Coordination between the operator and the tracking system. `operator_gateway/` receives UI commands and sends video and status back to the workstation. `session_target_manager/` manages start/stop requests, target selection, and tracking state. |
| `perception_stack/` | Target detection and estimation. `detection_tracking/` detects, classifies, and visually tracks targets in camera frames. `fusion_prediction/` combines visual observations with LiDAR range measurements to estimate target motion. |
| `integration_stack/` | Connections to physical hardware. `sensor_interface/` acquires and publishes camera and LiDAR data. `control_interface/` sends tracking commands to the external controller and receives position and controller feedback. |
| `data_stack/` | Recording and system health. `logging_diagnostics/` records events and telemetry and reports diagnostic information. This folder holds the recording and diagnostic code; generated recordings are runtime output. |
| `interfaces/` | Shared ROS 2 communication definitions used across stacks: `msg/` for messages, `srv/` for request/response services, and `action/` for longer-running operations that need progress or cancellation, if required. |
| `bringup/` | System startup and configuration. `launch/` holds launch files that start the required components together. `config/` holds their runtime parameters and configuration. |

The Jetson coordinates tracking and exchanges commands with the controller. The controller firmware owns the real-time motor-control loop.

## Operator application — `ui_stack/`

This contains the application running on the operator's PC or laptop. It communicates with the Jetson through the operator gateway.

| Folder | What belongs here |
| --- | --- |
| `src/display_gui/` | Screens and visual components for live video, target overlays, and tracking status. |
| `src/input_commands/` | Handling operator actions such as starting or stopping tracking and selecting a target. |
| `src/gateway_client/` | The workstation side of the connection to the Jetson: sending commands and receiving video and status. |

## Controller firmware — `firmware/`

This contains code that runs on the external microcontroller or real-time controller. Its role is to receive commands from the Jetson, control the motors, and return position feedback.

| Folder | What belongs here |
| --- | --- |
| `arduino/` | Arduino sketches (`.ino`), if an Arduino-compatible platform is selected. |
| `src/` | Controller source files for motor control, command handling, and feedback. |
| `include/` | Header files and declarations used by the controller source code. |

The board and firmware framework are still to be selected. These folders reserve space for that work; they do not require separate Arduino and native firmware implementations.

## Planned Operator Workflow

The operator application is intended to provide a simple interface for monitoring and controlling the target tracking system. The planned workflow is:

1. Launch the operator application and establish a connection to the Jetson.
2. View the live camera feed and available target detections.
3. Select a target to begin tracking.
4. Monitor tracking status and system feedback through the interface.
5. Stop tracking or select a different target when needed.

This workflow represents the intended software behavior and may be updated as development progresses.

## UC-10 / PROJECT23-14: preliminary camera-swap testing

`ros2_ws/src/integration_stack/sensor_interface/camera_source.py` provides a shared
camera adapter for the existing operator UI. It converts decoded visible-color
or monochrome frames to the same `uint8` BGR format consumed by the existing
RF-DETR detector and HUD renderer. The UI's **Camera** source replaces the former
**Webcam** entry; Demo and Video file sources keep their existing behavior.

**Preliminary mock test only: real camera hardware and physical swapping are
untested.** The mock backend generates synthetic visible and SWIR-style frames;
it does not simulate SWIR optics or establish model accuracy. Tests reuse the
existing `DemoDetector`, target-selection logic, tracking states, and HUD renderer.
They do not run a trained RF-DETR model or a ROS ingestion node. The adapter can
later be used by a ROS node once that integration exists.

Install the existing UI dependencies and the test runner inside your venv:

```bash
python -m pip install -r ui_stack/requirements.txt pytest
```

Run the preliminary checks from the repository root:

```bash
python -m pytest ros2_ws/src/integration_stack/sensor_interface/test_camera_source.py ui_stack/tests/test_camera_swap.py -v -p no:cacheprovider
```

The tests inject camera frames and disconnection at the capture boundary. They
check visible and monochrome ingestion, 12-bit samples stored in `uint16`, device
release, missing cameras, invalid profiles, stale-frame removal, and successful
tracking after restarting with the replacement sensor. Qt runs offscreen; no
camera or GPU is opened.

**Recorded preliminary result (2026-11-06):** 28 tests passed, with no failures,
errors, or skips, on Windows with Python 3.11.0. The observed mock UI sequence was
visible feed **TRACKING**, injected disconnect **IDLE** with no stale detections,
then SWIR-style feed **TRACKING** after restart and target reselection. Real
hardware acceptance and trained-model accuracy remain untested.

To see the same path in the operator application, start either example:

```bash
python ui_stack/run_ui.py --demo --camera-config ros2_ws/src/bringup/config/camera_visible_mock.json
python ui_stack/run_ui.py --demo --camera-config ros2_ws/src/bringup/config/camera_swir_mock.json
```

For a restart demonstration in one window, use a temporary copy of either JSON
profile, pass that file with `--camera-config`, replace its contents with the other
profile, and press **Restart camera**. The UI reloads the file, releases the old
capture, validates a frame from the new sensor, and clears old target IDs and
selection. Select a target again and press **Start tracking**. Mock sources are
labeled **PRELIMINARY MOCK** on screen. A failed camera shows **Camera unavailable**
instead of silently substituting demo video.

For future hardware testing, change `backend` to `opencv` and set `device` to the
camera index or device path. `pixel_format` describes the decoded output:
`bgr8`, `mono8`, or `mono16`. For `mono16`, set `white_level` to the sensor's maximum
sample value, such as 4095 for unpacked, right-aligned 12-bit data. Scaling uses
that fixed range rather than changing with each frame. OpenCV backend support is
device-dependent; packed pixels, Bayer data, and vendor SDK transports need an
appropriate driver adapter. No SWIR camera driver is claimed to be validated.

Hardware acceptance remains: disconnect the visible camera, attach the SWIR
camera, update only the sensor configuration if needed, restart ingestion, and
verify live video and detector initialization with the actual devices and model.

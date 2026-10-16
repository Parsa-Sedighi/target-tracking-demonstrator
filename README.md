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

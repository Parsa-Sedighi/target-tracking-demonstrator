# UC-07 calibration tests — PROJECT23-11

**Preliminary mock tests only. Physical hardware, ROS 2 integration, and camera/LiDAR
optical alignment have not been validated.** These tests extend the existing AimBot
engine, operator GUI, gimbal simulator, and TF02-Pro byte-frame parser. They do not
use a trained detector or open a camera, serial port, or motor controller.

## Run

From `aimbot_prototype`, use its separate virtual environment (PyQt6, not the older
`ui_stack` PySide6 environment). The mock needs only these existing dependencies:

```powershell
python -m pip install numpy opencv-python PyQt6 PyYAML pytest
$env:QT_QPA_PLATFORM = 'offscreen'
python -m pytest tests/test_calibration.py -q -o junit_family=legacy --junitxml=uc07.xml
```

The JUnit report includes measured range, encoder readback, crosshair error, and
each calibration step as properties of `test_gui_calibration_at_ten_feet`. Keep
generated reports outside the checkout, or remove them before committing.

The recorded Windows run on November 13, 2026 passed **28/28 UC-07 checks**. Its
independent range measurement was **3.0493 m** at the 3.048 m reference (**0.0512 in**
error), and crosshair error was **0.6272 px**. See [the recorded mock result](uc07_preliminary_result.json).
The full prototype suite had 116 passes, 8 skips, and one occlusion/reacquisition
failure. That same test also failed on unchanged main (`6ee96aa`); it is not a UC-07
calibration failure. Skipped cases and hardware limitations are listed in the result.

For an interactive demonstration, unset `QT_QPA_PLATFORM` and run:

```powershell
Remove-Item Env:QT_QPA_PLATFORM -ErrorAction SilentlyContinue
python -m aimbot -c calibration_mock
```

Leave tracking stopped, use **10 ft**, and click **Calibrate (zero gimbal + LiDAR)**.
After it finishes, click **Set aim point (click)**, then the board's centre marker.
The latter is the prototype's existing manual crosshair/boresight step; the calibration
button alone does not find a target or physically align the sensors. This configuration
stores its settings separately in the ignored `calibration_mock.json`.

## What is checked

- A stationary 12-inch square mock board has its face at **3.048 m (10 ft)**. The
  simulated beam intersects that plane; pointing away returns the background range.
- A GUI mouse click starts the real asynchronous calibration command from a nonzero
  gimbal pose. Fresh feedback must confirm home and software zero positions.
- The existing LiDAR path receives simulated TF02-Pro packets with a **+2.5 cm**
  bias. A separate batch after correction must be within **±2 inches (0.0508 m)**.
- The operator's actual aim-point mouse path moves an intentionally wrong crosshair
  onto the rendered board centre. The test allows **2 pixels** for click rounding;
  this is a mock test tolerance, not a hardware accuracy requirement.
- Settings survive reload. Invalid distances, busy states, failed homing/zero feedback,
  missing or nonfinite range data, failed persistence, Stop, and E-stop cannot produce
  a successful calibration. Failed runs retain prior saved settings.
- Injected verification errors exercise both signs of the inclusive 2-inch boundary:
  1.99 and 2.00 inches pass; 2.01 inches fails.

## Limits before the bench test

The simulator assumes camera and LiDAR share a reference axis and origin. Moving a
crosshair is not a measurement of physical boresight alignment. `zero_here()` changes
software offsets; these tests do not prove a hardware encoder-reset command.

The checked-in hardware configurations describe Blackfly S/Kowa imaging, TF02-Pro
UART, and Gremsy/ESP32 control. Confirm the final installed gimbal, mounting transform,
encoder home/reset semantics, and board dimensions/reference marks before hardware
acceptance. Use an independently surveyed board-face distance, observe the LiDAR spot,
and verify fresh readings after calibration. A fitted offset alone cannot establish
that the beam hit the intended board.

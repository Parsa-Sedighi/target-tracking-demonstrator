# AimBot prototype — YOLO26 tracking pipeline

A standalone prototype of the whole tracking pipeline: camera ingestion, asynchronous YOLO26
detection, a frame-rate tracker, Kalman prediction and gimbal control, with an operator GUI, data
logging and a simulator. It runs on a PC (simulator, webcam or video file) and is written for the
Jetson Orin Nano with the FLIR Blackfly S, Gremsy Mio and TF02-Pro.

It is self-contained and does not change the ROS 2 stacks in `../ros2_ws/` or the operator app in
`../ui_stack/`; parts of it can be moved into those stacks later. Run everything from this folder.

## Setup (Windows, NVIDIA GPU, Python 3.11 – 3.13)
Use a separate venv inside this folder: this prototype needs PyQt6 and the full `opencv-python`,
while `ui_stack` uses PySide6 and `opencv-python-headless`.
```
cd aimbot_prototype
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```
Install PyTorch first, as above, so it gets the CUDA build. Reading index cards (OCR) also needs
Tesseract: the UB Mannheim Windows installer, default folder.

## Run
```
python -m aimbot                                     # simulator: press Start tracking
python -m aimbot -c webcam                           # trained detector on your webcam (no gimbal)
python -m aimbot -c webcam --video clip.mp4 --target drone --auto-start   # ... on a video file
python3 -m aimbot -c jetson_headless --headless      # on the Jetson with the hardware
python -m aimbot --connect <jetson-ip>               # GUI on the PC, driving the Jetson
```
Configs are in `config/` (`default.yaml` has every setting, commented); override any value with
`--set key=value`.

## Model
`models/aimbot_yolo26n.pt` is YOLO26n fine-tuned for 20 epochs on 10,000 images taken evenly from six
drone datasets (Anti-UAV, Drone Hunter, MAV-VID, Roboflow indoor, DUT Anti-UAV, Halmstad). Classes:
drone, bird, airplane, helicopter. Averaged over the datasets' validation images it reaches drone
precision 0.91 and recall 0.87 (PRD target 0.85 / 0.80); it is weakest on tiny, distant drones.
The tennis ball comes from the stock COCO yolo26n, which downloads by itself on the first run.
To retrain it on any YOLO-format dataset: `python tools/train_yolo26.py --data <dataset.yaml>`.

## Using your own model
Any Ultralytics detection model (YOLOv8, YOLO11, YOLO26, ...; `.pt`, `.onnx` or a TensorRT `.engine`) drops in:
```
python -m aimbot -c webcam --video clip.mp4 --target drone --auto-start --set detector.model=path/to/model.pt
```
Its drone class must be called drone, uav or quadcopter, in any capitalisation; for another name add
`--set detector.class_map.<name>=drone`, and `--set detector.imgsz=<size>` if it was trained at another size.
To score it on a labelled YOLO dataset: `yolo val model=path/to/model.pt data=<dataset.yaml>`.

## Tests
`python -m pytest`: unit tests, closed-loop simulator tests, the dataset pipeline and the training tools.
Three detector tests run only after the stock yolo26n has downloaded (start the app once).

## Layout
```
aimbot/      tracker: sensors, perception, control, gimbal drivers, GUI, logging, simulator
             training/: dataset readers, quality filter and builder for retraining
config/      settings for each setup        models/   trained detector
tests/       automated tests                tools/    training, label audit, run analysis, gimbal emulator
```

## Status
* Simulator: capture → gimbal-command latency ~21 ms at the 95th percentile (budget 100 ms); the
  target stays in view in 100 % of tracked frames.
* Not yet run on the physical Jetson, Blackfly, Mio or TF02-Pro; expect first-power-up fixes such as
  axis signs, baud rates and lead time.
* Index card and hand need team-labelled frames: no public dataset has them.
* Licences: Ultralytics YOLO is AGPL-3.0, and the model was trained partly on research-only datasets
  (Anti-UAV, DUT Anti-UAV, MAV-VID, Drone Hunter); check their terms before using it outside the project.

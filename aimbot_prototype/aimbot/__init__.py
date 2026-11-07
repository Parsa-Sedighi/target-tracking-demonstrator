"""AimBot - Target Tracking Demonstrator (SDSU Senior Design Project #23, GA-EMS).

Prototype software engine: camera ingestion -> async YOLO detection -> micro
tracker -> Kalman prediction -> gimbal commands, with an operator GUI, data
logging and hardware abstraction so the same code runs on a Windows PC (sim /
webcam) and on the Jetson Orin Nano Super with the FLIR Blackfly S, Gremsy Mio
gimbal and Benewake TF02-Pro LiDAR.
"""

__version__ = "0.1.0"

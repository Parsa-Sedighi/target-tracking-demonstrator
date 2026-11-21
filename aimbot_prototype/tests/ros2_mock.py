"""Minimal ROS API spies for preliminary publisher tests; no executor, DDS or real ROS traffic."""

from copy import deepcopy
import logging
import sys
from types import ModuleType, SimpleNamespace

from aimbot import clock


class String:
    def __init__(self):
        self.data = ""


class Publisher:
    def __init__(self, topic):
        self.topic = topic
        self.messages = []

    def publish(self, message):
        self.messages.append((clock.now(), deepcopy(message)))


class Node:
    def __init__(self, name, **kwargs):
        self.name = name
        self.publishers = {}
        self.subscriptions = {}
        self.timers = []

    def create_publisher(self, message_type, topic, qos):
        publisher = self.publishers[topic] = Publisher(topic)
        return publisher

    def create_subscription(self, message_type, topic, callback, qos):
        self.subscriptions[topic] = callback
        return callback

    def create_timer(self, period, callback):
        self.timers.append((period, callback))
        return callback

    def get_logger(self):
        return logging.getLogger("uc09.mock_ros_node")


def install_ros2_mock(monkeypatch):
    def module(name, **attrs):
        m = ModuleType(name)
        m.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    rclpy = module("rclpy", ok=lambda: True, create_node=Node)
    rclpy.node = module("rclpy.node", Node=Node)
    rclpy.qos = module("rclpy.qos", QoSProfile=lambda **kwargs: SimpleNamespace(**kwargs),
                       ReliabilityPolicy=SimpleNamespace(BEST_EFFORT=1, RELIABLE=2),
                       HistoryPolicy=SimpleNamespace(KEEP_LAST=1))
    rclpy.time = module("rclpy.time", Time=SimpleNamespace)
    for package, messages in (("std_msgs", {"String": String}),
                              ("geometry_msgs", {"Vector3Stamped": SimpleNamespace}),
                              ("sensor_msgs", {"JointState": SimpleNamespace, "Range": SimpleNamespace})):
        parent = module(package)
        parent.msg = module(package + ".msg", **messages)

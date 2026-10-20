# src/aimbot_core/state_machine_node.py
import rclpy
from rclpy.node import Node
from enum import Enum
from std_msgs.msg import String

class AimBotState(Enum):
    IDLE = "IDLE"
    SEARCHING = "SEARCHING"
    TRACKING = "TRACKING"

class SystemStateMachine(Node):
    def __init__(self):
        super().__init__('system_state_machine')
        self.state = AimBotState.IDLE
        self.target_visible = False
        
        self.state_publisher = self.create_publisher(String, '/aimbot/state', 10)
        self.target_sub = self.create_subscription(
            String, '/vision/target_status', self.target_callback, 10
        )
        self.timer = self.create_timer(0.01, self.publish_state)  # 100 Hz

    def target_callback(self, msg: String):
        self.target_visible = (msg.data == "ACQUIRED")
        self.evaluate_state_transition()

    def evaluate_state_transition(self):
        if self.state == AimBotState.TRACKING and not self.target_visible:
            self.get_logger().warn("Target lost. Transitioning to SEARCHING mode.")
            self.state = AimBotState.SEARCHING
        elif self.state in (AimBotState.SEARCHING, AimBotState.IDLE) and self.target_visible:
            self.get_logger().info("Target acquired. Transitioning to TRACKING mode.")
            self.state = AimBotState.TRACKING

    def publish_state(self):
        msg = String()
        msg.data = self.state.value
        self.state_publisher.publish(msg)
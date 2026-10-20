# tests/test_state_machine.py
import pytest
import rclpy
from std_msgs.msg import String
from aimbot_core.state_machine_node import SystemStateMachine, AimBotState

@pytest.fixture(scope="module")
def ros_context():
    rclpy.init()
    yield
    rclpy.shutdown()

def test_state_transition_on_lost_target(ros_context):
    node = SystemStateMachine()
    node.state = AimBotState.TRACKING
    
    # Simulate target loss
    msg = String()
    msg.data = "LOST"
    node.target_callback(msg)
    
    assert node.state == AimBotState.SEARCHING
    node.destroy_node()

def test_state_transition_on_acquired_target(ros_context):
    node = SystemStateMachine()
    node.state = AimBotState.SEARCHING
    
    # Simulate target acquisition
    msg = String()
    msg.data = "ACQUIRED"
    node.target_callback(msg)
    
    assert node.state == AimBotState.TRACKING
    node.destroy_node()
# ros2_ws/src/integration_stack/sensor_interface/sim_camera_node.py
"""
ROS 2 node that publishes simulated target data, standing in for the real
camera + LiDAR until hardware is available.

Publishes:
  /vision/target_status  (std_msgs/String)       "ACQUIRED" or "LOST"
      -> the topic the system state machine already listens to.
  /sim/target_error      (geometry_msgs/Point)   x, y = pixel offset from frame
      center, z = simulated LiDAR range in meters.
      -> the kind of data gimbal control will eventually consume.

Run (after sourcing ROS 2):
  python3 sim_camera_node.py
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Point

from target_simulator import SimulatedTarget


class SimCameraNode(Node):
    def __init__(self):
        super().__init__('sim_camera')

        # Parameters can be changed at launch without editing code, e.g.
        #   --ros-args -p rate_hz:=60.0 -p dropout_prob:=0.1
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('speed_px_s', 400.0)
        self.declare_parameter('dropout_prob', 0.0)
        self.declare_parameter('seed', 42)

        rate_hz = self.get_parameter('rate_hz').value
        self.dt = 1.0 / rate_hz
        self.sim = SimulatedTarget(
            speed_px_s=self.get_parameter('speed_px_s').value,
            dropout_prob=self.get_parameter('dropout_prob').value,
            seed=self.get_parameter('seed').value,
        )

        self.status_pub = self.create_publisher(String, '/vision/target_status', 10)
        self.error_pub = self.create_publisher(Point, '/sim/target_error', 10)
        self.timer = self.create_timer(self.dt, self.tick)
        self.last_visible = None

    def tick(self):
        obs = self.sim.step(self.dt)

        status = String()
        status.data = "ACQUIRED" if obs.visible else "LOST"
        self.status_pub.publish(status)

        if obs.visible:
            point = Point()
            point.x = obs.error_x
            point.y = obs.error_y
            point.z = obs.range_m
            self.error_pub.publish(point)

        # Log only when visibility changes, so the console isn't flooded.
        if obs.visible != self.last_visible:
            self.get_logger().info(f"Target {status.data} at t={obs.t:.1f}s")
            self.last_visible = obs.visible


def main():
    rclpy.init()
    node = SimCameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

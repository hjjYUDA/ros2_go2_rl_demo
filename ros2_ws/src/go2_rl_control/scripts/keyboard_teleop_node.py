#!/usr/bin/python3
# NOTE: Use system Python — see mujoco_sim_node.py header for conda / rclpy issue.
"""ROS 2 keyboard teleop for the Go2 RL controller.

Mirrors the behaviour of the legacy ``keyboard_teleop`` ROS 1 node so the
ROS-side experience is unchanged: each tap of W/S/A/D/Q/E adds a fixed delta
to the velocity command, Space zeroes it, and 1/2/3 toggle stand/walk/passive
modes that the rl_inference_node listens to.

Usage:
    ros2 run go2_rl_control keyboard_teleop_node.py
"""

from __future__ import annotations

import select
import sys
import termios
import tty
from typing import Optional

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from geometry_msgs.msg import Twist


HELP_MSG = """
Go2 ROS 2 Keyboard Teleop
-------------------------
  W / S : forward / backward     (each press +/- step_v)
  A / D : strafe left / right
  Q / E : yaw   left / right     (each press +/- step_w)
  Space : zero all velocity
  1     : STAND   mode
  2     : WALK    mode
  3     : PASSIVE mode
  Ctrl+C: quit
"""


def _getch_nonblocking(timeout: float = 0.1) -> str:
    """Read a single character if one is available within timeout seconds."""
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if not rlist:
        return ""
    return sys.stdin.read(1)


class KeyboardTeleopNode(Node):
    def __init__(self):
        super().__init__("go2_keyboard_teleop")

        self.declare_parameter("step_v", 0.1)
        self.declare_parameter("step_w", 0.2)
        self.declare_parameter("max_vx", 1.0)
        self.declare_parameter("max_vy", 0.4)
        self.declare_parameter("max_wz", 1.0)

        self.step_v: float = float(self.get_parameter("step_v").value)
        self.step_w: float = float(self.get_parameter("step_w").value)
        self.max_vx: float = float(self.get_parameter("max_vx").value)
        self.max_vy: float = float(self.get_parameter("max_vy").value)
        self.max_wz: float = float(self.get_parameter("max_wz").value)

        self.cmd_pub = self.create_publisher(Twist, "/go2/cmd_vel", 1)
        self.mode_pub = self.create_publisher(String, "/go2/mode", 1)

        self.twist = Twist()
        self.timer = self.create_timer(0.05, self._tick)

        self.get_logger().info(HELP_MSG.strip())
        sys.stdout.write(HELP_MSG)
        sys.stdout.flush()

    @staticmethod
    def _clip(v: float, lim: float) -> float:
        return max(-lim, min(lim, v))

    def _tick(self) -> None:
        c = _getch_nonblocking(timeout=0.0)
        if not c:
            return
        updated = True
        if c in ("w", "W"):
            self.twist.linear.x = self._clip(self.twist.linear.x + self.step_v, self.max_vx)
        elif c in ("s", "S"):
            self.twist.linear.x = self._clip(self.twist.linear.x - self.step_v, self.max_vx)
        elif c in ("a", "A"):
            self.twist.linear.y = self._clip(self.twist.linear.y + self.step_v, self.max_vy)
        elif c in ("d", "D"):
            self.twist.linear.y = self._clip(self.twist.linear.y - self.step_v, self.max_vy)
        elif c in ("q", "Q"):
            self.twist.angular.z = self._clip(self.twist.angular.z + self.step_w, self.max_wz)
        elif c in ("e", "E"):
            self.twist.angular.z = self._clip(self.twist.angular.z - self.step_w, self.max_wz)
        elif c == " ":
            self.twist.linear.x = 0.0
            self.twist.linear.y = 0.0
            self.twist.angular.z = 0.0
        elif c == "1":
            m = String(); m.data = "stand"; self.mode_pub.publish(m)
            sys.stdout.write("\nMode: STAND\n"); sys.stdout.flush()
        elif c == "2":
            m = String(); m.data = "walk"; self.mode_pub.publish(m)
            sys.stdout.write("\nMode: WALK\n"); sys.stdout.flush()
        elif c == "3":
            m = String(); m.data = "passive"; self.mode_pub.publish(m)
            sys.stdout.write("\nMode: PASSIVE\n"); sys.stdout.flush()
        else:
            updated = False
        if updated:
            self.cmd_pub.publish(self.twist)
            sys.stdout.write(
                f"\rvx={self.twist.linear.x:+.2f} vy={self.twist.linear.y:+.2f} "
                f"wz={self.twist.angular.z:+.2f}    "
            )
            sys.stdout.flush()


def main(args: Optional[list] = None):
    rclpy.init(args=args)
    node = KeyboardTeleopNode()

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

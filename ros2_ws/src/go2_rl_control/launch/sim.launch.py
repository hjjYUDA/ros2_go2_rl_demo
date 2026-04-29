"""ROS 2 launch: bring up MuJoCo + RL inference for the Go2.

Usage:
    ros2 launch go2_rl_control sim.launch.py rl_mode:=stand visualize:=true

Then in another terminal, run the keyboard teleop:
    ros2 run go2_rl_control keyboard_teleop_node.py

The teleop publishes /go2/cmd_vel (geometry_msgs/Twist) and /go2/mode
(std_msgs/String). The rl_inference_node consumes both, plus /go2/joint_states
and /go2/imu published by the mujoco_sim_node.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    pkg = get_package_share_directory("go2_rl_control")
    default_onnx = os.path.join(pkg, "policy", "policy.onnx")
    default_yaml = os.path.join(pkg, "policy", "deploy.yaml")
    default_mjcf = os.path.join(pkg, "description", "mujoco", "scene.xml")

    rl_mode = LaunchConfiguration("rl_mode")
    visualize = LaunchConfiguration("visualize")
    onnx_path = LaunchConfiguration("onnx_path")
    deploy_yaml = LaunchConfiguration("deploy_yaml")
    mjcf_path = LaunchConfiguration("mjcf_path")

    return LaunchDescription([
        DeclareLaunchArgument("rl_mode", default_value="stand",
                              description="initial controller mode: stand|walk|passive"),
        DeclareLaunchArgument("visualize", default_value="true",
                              description="show MuJoCo passive viewer window"),
        DeclareLaunchArgument("onnx_path", default_value=default_onnx),
        DeclareLaunchArgument("deploy_yaml", default_value=default_yaml),
        DeclareLaunchArgument("mjcf_path", default_value=default_mjcf),

        Node(
            package="go2_rl_control",
            executable="mujoco_sim_node.py",
            name="mujoco_sim_node",
            output="screen",
            parameters=[{
                "mjcf_path": mjcf_path,
                "visualize": visualize,
            }],
        ),
        Node(
            package="go2_rl_control",
            executable="rl_inference_node",
            name="go2_rl_inference_node",
            output="screen",
            parameters=[{
                "onnx_path": onnx_path,
                "deploy_yaml": deploy_yaml,
                "mode": rl_mode,
            }],
        ),
    ])

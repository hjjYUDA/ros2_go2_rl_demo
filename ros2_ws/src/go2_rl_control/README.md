# go2_rl_control (ROS 2)

ROS 2 deployment of a Unitree Go2 quadruped policy trained in IsaacLab via
[`unitree_rl_lab`](../../../unitree_rl_lab). Provides three nodes:

| Node                       | Lang | Topic in / out                                                          |
|----------------------------|------|--------------------------------------------------------------------------|
| `rl_inference_node`        | C++  | in: `/go2/joint_states`, `/go2/imu`, `/go2/cmd_vel`, `/go2/mode`<br>out: `/go2/cmd_torque` |
| `mujoco_sim_node.py`       | Py   | in: `/go2/cmd_torque`<br>out: `/go2/joint_states`, `/go2/imu`, `/go2/base_pose` |
| `keyboard_teleop_node.py`  | Py   | out: `/go2/cmd_vel` (Twist), `/go2/mode` (String "stand"/"walk"/"passive") |

Joint ordering on every ROS topic is the **Unitree SDK** order:

```
FR_hip, FR_thigh, FR_calf, FL_hip, FL_thigh, FL_calf,
RR_hip, RR_thigh, RR_calf, RL_hip, RL_thigh, RL_calf
```

`rl_inference_node` reorders to/from the IsaacLab "asset" order using the
`joint_ids_map` recorded in `policy/deploy.yaml`.

## Build prerequisites
* ROS 2 humble (Ubuntu 22.04)
* `pip install --user mujoco` (Python ≥ 3.10)
* ONNX Runtime C++ unpacked to `/opt/onnxruntime` (override with
  `colcon build --cmake-args -DONNXRUNTIME_ROOT=/path/to/onnxruntime`)
* `sudo apt install ros-humble-rclcpp ros-humble-sensor-msgs \
       ros-humble-geometry-msgs libyaml-cpp-dev libeigen3-dev`

## Build
```bash
cd ~/ros2_ws
colcon build --symlink-install --packages-select go2_rl_control
source install/setup.bash
```

## Drop in a new policy
1. Train with `unitree_rl_lab` (recommended task: `Unitree-Go2-Velocity-Robust`).
2. Export the policy:
   ```bash
   python ~/unitree_rl_lab/scripts/rsl_rl/export_onnx_standalone.py \
     --checkpoint <run_dir>/model_<iter>.pt
   ```
3. Copy the artefacts:
   ```bash
   cp <run_dir>/exported/policy.onnx policy/
   cp <run_dir>/params/deploy.yaml   policy/
   colcon build --symlink-install --packages-select go2_rl_control
   ```

## Run (sim2sim with the bundled MuJoCo bridge)
Terminal A:
```bash
ros2 launch go2_rl_control sim.launch.py rl_mode:=stand visualize:=true
```
Terminal B:
```bash
ros2 run go2_rl_control keyboard_teleop_node.py
```
Then press `2` to switch to WALK mode and use `W/S/A/D/Q/E` to drive the
robot. Press `1` to STAND, `3` for PASSIVE, Space to zero velocity.

To test disturbance rejection in the MuJoCo passive viewer, hold `Ctrl` and
right-click-drag the robot — the policy should recover.

## Layout
```
go2_rl_control/
  src/rl_inference_node.cpp         # C++ ONNX inference + PD torque
  scripts/mujoco_sim_node.py        # MuJoCo physics + ROS bridge (1 kHz/200 Hz)
  scripts/keyboard_teleop_node.py   # WASD/QE/123 -> /go2/cmd_vel + /go2/mode
  launch/sim.launch.py              # bring up mujoco_sim + rl_inference
  policy/policy.onnx                # exported actor MLP
  policy/deploy.yaml                # observation/action/joint mapping
  description/mujoco/scene.xml      # mujoco_menagerie unitree_go2 (Apache-2.0)
  description/urdf/go2.urdf         # placeholder for future RViz integration
```

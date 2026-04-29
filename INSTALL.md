# 安装与部署要求 / Installation & Deployment

> 本文档专门描述 **ros2_go2_rl_demo** 在「训练侧」与「部署侧」需要的环境与最小配置步骤。已经熟悉的读者可以直接跳到 [§4 编译运行](#4-编译运行)。

---

## 1. 硬件要求

| 角色 | 最低 | 推荐 |
|------|------|------|
| 训练 | NVIDIA GPU ≥ 8 GB 显存（可跑 IsaacSim/IsaacLab） | RTX 30/40 系，≥ 16 GB 显存 |
| 部署 / 仿真 | 任意 x86_64，2 GB 内存以上 | 4 核以上，便于 200 Hz 发布 |
| 键盘 | 真实 TTY 或带键盘焦点的终端 | — |
| 显示器（看 MuJoCo 窗口） | 桌面 X11/Wayland 或 VNC、`ssh -X` | 本机桌面体验最佳 |

> 服务器上**只看话题**不需要显示器；只有 `visualize:=true` 才依赖图形显示环境。

---

## 2. 操作系统与基础组件

| 组件 | 版本要求 | 说明 |
|------|---------|------|
| Ubuntu | **22.04 LTS（Jammy）** | 与 ROS 2 Humble 官方支持系统一致 |
| GPU 驱动 | NVIDIA Driver ≥ 535 | 训练侧需要 |
| CUDA | 11.8 / 12.x（按 IsaacLab 要求） | 训练侧需要 |
| Python | 系统 `/usr/bin/python3.10` | ROS 2 Python 节点必须使用 |
| ROS 2 | **Humble Hawksbill** | 部署/仿真侧 |
| Conda（可选） | Miniconda / Anaconda | 训练用，**仅训练时激活** |

> **重要：** Conda 自带的 Python 与 `libstdc++.so.6` 与 ROS 2 不兼容，会触发 `GLIBCXX_3.4.30 not found`。本仓库的 Python 节点 shebang 已固定为 `/usr/bin/python3`；运行 `ros2 launch` 前请先 `conda deactivate`。

---

## 3. 软件依赖与安装步骤

### 3.1 ROS 2 Humble（部署机）

```bash
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
  -o /usr/share/keyrings/ros-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" | \
  sudo tee /etc/apt/sources.list.d/ros2.list
sudo apt update
sudo apt install -y ros-humble-desktop \
  ros-humble-rclcpp ros-humble-rclpy \
  ros-humble-sensor-msgs ros-humble-std-msgs ros-humble-geometry-msgs \
  ros-humble-launch-ros
```

### 3.2 编译期 / 运行期 C++ 依赖

```bash
sudo apt install -y python3-colcon-common-extensions \
                    libyaml-cpp-dev libeigen3-dev cmake build-essential
```

### 3.3 ONNX Runtime（C++）

下载 **CPU x64** 预编译包并解压到 `/opt/onnxruntime`（CMake 默认查找此路径，可用 `-DONNXRUNTIME_ROOT=...` 覆盖）：

```bash
ORT_VER=1.18.1
cd /tmp && wget -q "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VER}/onnxruntime-linux-x64-${ORT_VER}.tgz"
sudo tar -xzf "onnxruntime-linux-x64-${ORT_VER}.tgz" -C /opt/
sudo ln -sfn "/opt/onnxruntime-linux-x64-${ORT_VER}" /opt/onnxruntime
ls /opt/onnxruntime/include/onnxruntime_cxx_api.h /opt/onnxruntime/lib/libonnxruntime.so
```

> 也支持 GPU 版（`onnxruntime-linux-x64-gpu-*.tgz`）；本项目策略很小，CPU 已绰绰有余。

### 3.4 MuJoCo Python（部署机；用于仿真节点）

```bash
sudo /usr/bin/pip3 install --break-system-packages mujoco
# 验证（必须用系统 python3）
/usr/bin/python3 -c "import mujoco; print('mujoco', mujoco.__version__)"
```

> 若 `pip3 install --user mujoco` 也行，只要 **系统 Python** 能 `import mujoco` 即可。**不要装到 conda 里**，否则 `ros2 run` 找不到。

### 3.5 IsaacLab + unitree_rl_lab（训练机；建议 Conda）

```bash
# 例：Miniconda + Python 3.10 环境
conda create -n env_isaaclab python=3.10 -y
conda activate env_isaaclab

# 1) 安装 IsaacSim/IsaacLab（参考官方文档）
#    https://isaac-sim.github.io/IsaacLab/main/
# 2) 克隆 unitree_rl_lab
git clone https://github.com/unitree-rl-lab/unitree_rl_lab.git ~/unitree_rl_lab
cd ~/unitree_rl_lab && pip install -e source/unitree_rl_lab

# 3) 应用本仓库训练侧补丁
ROS2_GO2_RL_DEMO=$HOME/ros2_go2_rl_demo
cp $ROS2_GO2_RL_DEMO/unitree_rl_lab/tasks/velocity_robust_env_cfg.py \
   ~/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/
cp $ROS2_GO2_RL_DEMO/unitree_rl_lab/tasks/__init__.py \
   ~/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/__init__.py
cp $ROS2_GO2_RL_DEMO/unitree_rl_lab/scripts/play_keyboard.py        ~/unitree_rl_lab/scripts/rsl_rl/
cp $ROS2_GO2_RL_DEMO/unitree_rl_lab/scripts/export_onnx_standalone.py ~/unitree_rl_lab/scripts/rsl_rl/
```

> 训练侧也需要 `pip install onnx onnxruntime numpy pyyaml torch`，IsaacLab 的环境通常已包含。

---

## 4. 编译运行

### 4.1 克隆并编译 ROS 2 包

```bash
git clone https://github.com/hjjYUDA/ros2_go2_rl_demo.git ~/ros2_go2_rl_demo
cd ~/ros2_go2_rl_demo/ros2_ws

# 重要：先退出 conda，再 source ROS
conda deactivate || true
source /opt/ros/humble/setup.bash

colcon build --symlink-install --packages-select go2_rl_control
source install/setup.bash
```

### 4.2 训练（可选；仓库已带 v1 训练好的 `policy.onnx`）

```bash
conda activate env_isaaclab
cd ~/unitree_rl_lab

python scripts/rsl_rl/train.py --task Unitree-Go2-Velocity-Robust \
       --num_envs 4096 --max_iterations 5000 --headless 2>&1 | tee /tmp/go2_train.log
```

### 4.3 训练完成后导出并刷新策略

```bash
~/ros2_go2_rl_demo/ros2_ws/src/go2_rl_control/scripts/refresh_policy.sh
```

或自动等待训练结束并刷新（同时拉起仿真栈）：

```bash
RUN_LAUNCH=1 RUN_VISUALIZE=false \
  nohup ~/ros2_go2_rl_demo/ros2_ws/src/go2_rl_control/scripts/wait_train_then_refresh.sh \
    >> /tmp/wait_train.log 2>&1 &
```

### 4.4 启动仿真 + 推理 + 键盘

```bash
# 终端 A
conda deactivate || true
source /opt/ros/humble/setup.bash
source ~/ros2_go2_rl_demo/ros2_ws/install/setup.bash
ros2 launch go2_rl_control sim.launch.py rl_mode:=stand visualize:=true

# 终端 B
conda deactivate || true
source /opt/ros/humble/setup.bash
source ~/ros2_go2_rl_demo/ros2_ws/install/setup.bash
ros2 run go2_rl_control keyboard_teleop_node.py
```

---

## 5. 跨主机查看（同局域网）

两台机器都装 ROS 2 Humble 并在同一 `ROS_DOMAIN_ID`：

```bash
export ROS_DOMAIN_ID=42         # 两台一致
ros2 topic list                  # 应能看到 /go2/* 系列话题
ros2 topic echo /go2/base_pose
```

如果话题看不见，检查防火墙（DDS 走 UDP）、多网卡，或试 CycloneDDS：

```bash
sudo apt install -y ros-humble-rmw-cyclonedds-cpp
export RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
```

要看 **MuJoCo 仿真画面**：必须在运行 `mujoco_sim_node` 的那台机器上有显示环境，可选 `ssh -X`、VNC、远程桌面，或在本机桌面运行。

---

## 6. v1 验证清单（用于自检）

| 检查项 | 期望 | 命令 |
|--------|------|------|
| MuJoCo 节点活着 | 无异常退出 | 看 launch 日志 |
| 关节状态发布 | ≈200 Hz | `ros2 topic hz /go2/joint_states` |
| 力矩输出 | 50 Hz | `ros2 topic hz /go2/cmd_torque` |
| WALK 前进 | x 增长，z≈0.31 m | `ros2 topic echo /go2/base_pose` |
| 抗干扰 | 5 N·s 后 z 保持 | `ros2 topic pub --once /go2/disturb …` |

---

## 7. 常见错误速查

| 现象 | 处理 |
|------|------|
| `GLIBCXX_3.4.30 not found` | **`conda deactivate`** 后重新 source ROS；本仓库 shebang 已固定为 `/usr/bin/python3` |
| `import mujoco` 失败（仅 Python 节点） | `sudo /usr/bin/pip3 install --break-system-packages mujoco` |
| 找不到 `onnxruntime_cxx_api.h` | 解压 ONNX Runtime 到 `/opt/onnxruntime` 或 `colcon build --cmake-args -DONNXRUNTIME_ROOT=...` |
| `visualize:=true` 无窗口 | 当前 shell 无 `DISPLAY`：用桌面终端、`ssh -X`、VNC，或装 `xvfb`（仅自动化用） |
| 跨机看不到话题 | 确认 `ROS_DOMAIN_ID` 一致、放行 UDP，必要时切换 RMW（CycloneDDS） |

更多细节请看 `ros2_ws/src/go2_rl_control/docs/实现与应用指南.md`。

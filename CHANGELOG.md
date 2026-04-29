# Changelog

## v1.0 — 2026-04-29

首个端到端可复现版本。

### 训练侧

- 新增 IsaacLab 任务 `Unitree-Go2-Velocity-Robust`，覆盖 `RobotEnvCfg` 引入更强的域随机化：
  - `events.push_robot`：interval 5–8 s，6 自由度速度扰动
  - `events.add_base_mass`：质量分布 `(-2.5, +5.0)` kg
  - `events.physics_material`：摩擦 0.3–1.2（静）/ 0.3–1.0（动）
  - `events.reset_base`：非零初始速度
- `tasks/locomotion/robots/go2/__init__.py` 注册新任务及其 `play_env_cfg`
- 新增 `scripts/rsl_rl/export_onnx_standalone.py`：**无需 IsaacSim** 即可从 `model_*.pt + agent.yaml` 导出 `policy.onnx` / `policy.pt`
- 新增 `scripts/rsl_rl/play_keyboard.py`：在 IsaacSim 中通过 `carb.input` 增量键盘控制 `base_velocity`，含 STAND/WALK/PASSIVE 状态机

### 部署侧（ROS 2 包 `go2_rl_control`，ament_cmake）

- C++ 推理节点 `rl_inference_node.cpp`：
  - 加载 `policy.onnx`（onnxruntime）+ `deploy.yaml`（yaml-cpp）
  - 严格按 `deploy.yaml` 顺序拼接 45 维观测
  - 修正 `joint_ids_map` 方向（`joint_ids_map[asset_idx] = sdk_idx`），SDK ↔ asset 重排
  - PD 控制（SDK 顺序）：`tau = kp*(q_des - q) - kd*dq`
  - 50 Hz 输出 `/go2/cmd_torque`
  - 模式切换：STAND / WALK / PASSIVE
- Python 节点 `mujoco_sim_node.py`：
  - 1 kHz 物理步、~200 Hz 发布 `/go2/joint_states /go2/imu /go2/base_pose`
  - 订阅 `/go2/cmd_torque`，按 actuator 写入 `data.ctrl`
  - 订阅 `/go2/disturb`（`Vector3`，N·s 冲量），用于抗干扰测试
  - `visualize` 字符串安全解析；无 `DISPLAY` 时给出明确诊断日志
- Python 节点 `keyboard_teleop_node.py`：
  - WASD/QE 增量速度、空格清零、1/2/3 切换模式
  - 发布 `/go2/cmd_vel` (`Twist`) 与 `/go2/mode` (`String`)
- launch：`launch/sim.launch.py` 一键拉起 `mujoco_sim` + `rl_inference`
- 资产：内置 `mujoco_menagerie/unitree_go2` MJCF + meshes
- 脚本：
  - `scripts/refresh_policy.sh`：一键导出 + 拷贝 + colcon build
  - `scripts/wait_train_then_refresh.sh`：监听训练完成自动刷新（可选自动起 launch）
- 文档：`docs/实现与应用指南.md`、顶层 `INSTALL.md`、`README.md`

### 关键修正

- shebang 固定为 `#!/usr/bin/python3`，避免 conda 激活时 `rclpy` 加载失败
- `refresh_policy.sh` 中 `WS_DIR` 路径修正（之前多上溯一级）
- `CMakeLists.txt` 移除已删除的 `config/` 安装目录

### 已验证（model_4999）

| 项目 | 结果 |
|------|------|
| `cmd_torque` 频率 | 50 Hz |
| `joint_states` 频率 | ~200 Hz |
| WALK `vx=0.5` | 前进 ~2.55 m，z≈0.31 m |
| 5 N·s 冲量 | 不摔倒，z 保持 |
| 端到端 colcon build | 通过 |

# 训练侧补丁 / unitree_rl_lab patches

> 这里只放本项目相对上游 [`unitree-rl-lab/unitree_rl_lab`](https://github.com/unitree-rl-lab/unitree_rl_lab) **新增或修改** 的文件，**不是完整训练工程**。请先按上游文档安装好 `unitree_rl_lab` 与 IsaacLab，再把下列文件覆盖到对应位置。

## 文件清单与目标位置

| 本仓库路径 | 拷贝到 |
|------|------|
| `unitree_rl_lab/tasks/velocity_robust_env_cfg.py` | `<unitree_rl_lab>/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_robust_env_cfg.py` |
| `unitree_rl_lab/tasks/__init__.py` | `<unitree_rl_lab>/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/__init__.py`（**会覆盖原文件，建议先备份**） |
| `unitree_rl_lab/scripts/play_keyboard.py` | `<unitree_rl_lab>/scripts/rsl_rl/play_keyboard.py` |
| `unitree_rl_lab/scripts/export_onnx_standalone.py` | `<unitree_rl_lab>/scripts/rsl_rl/export_onnx_standalone.py` |

**一键拷贝示例：**

```bash
ROS2_GO2_RL_DEMO=$HOME/ros2_go2_rl_demo
RL_LAB=$HOME/unitree_rl_lab

cp $ROS2_GO2_RL_DEMO/unitree_rl_lab/tasks/velocity_robust_env_cfg.py \
   $RL_LAB/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/

cp $ROS2_GO2_RL_DEMO/unitree_rl_lab/tasks/__init__.py \
   $RL_LAB/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/__init__.py

cp $ROS2_GO2_RL_DEMO/unitree_rl_lab/scripts/play_keyboard.py        $RL_LAB/scripts/rsl_rl/
cp $ROS2_GO2_RL_DEMO/unitree_rl_lab/scripts/export_onnx_standalone.py $RL_LAB/scripts/rsl_rl/
```

## 使用

### 训练（启用增强域随机化）

```bash
conda activate env_isaaclab
cd $RL_LAB
python scripts/rsl_rl/train.py --task Unitree-Go2-Velocity-Robust \
       --num_envs 4096 --max_iterations 5000 --headless 2>&1 | tee /tmp/go2_train.log
```

### 在 IsaacSim 内键盘把玩

```bash
python scripts/rsl_rl/play_keyboard.py --task Unitree-Go2-Velocity-Robust \
       --checkpoint logs/rsl_rl/unitree_go2_velocity_robust/<run>/model_5000.pt --num_envs 1
```

### 无 IsaacSim 直接导出 ONNX

```bash
python scripts/rsl_rl/export_onnx_standalone.py \
       --checkpoint logs/rsl_rl/unitree_go2_velocity_robust/<run>/model_5000.pt
# 产物：<run>/exported/policy.onnx + policy.pt
```

随后用部署侧 `ros2_ws/src/go2_rl_control/scripts/refresh_policy.sh` 把 `policy.onnx` 与 `params/deploy.yaml` 一同更新到 ROS 包内。

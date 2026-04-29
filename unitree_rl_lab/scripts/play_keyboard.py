# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a trained Go2 RL policy in IsaacSim with keyboard teleop.

Mirrors the key bindings of the ROS-side ``keyboard_teleop`` node so that the
operator's mental model is the same on both sides:

  * W/S        : +/- linear velocity along x  (each press +/- step_v)
  * A/D        : +/- linear velocity along y
  * Q/E        : +/- angular velocity around z
  * Space      : zero all velocity commands
  * 1          : STAND  mode (zero command + freeze)
  * 2          : WALK   mode (use keyboard command)
  * 3          : PASSIVE mode (zero torque, robot drops)
  * R          : reset envs

The script overrides ``env.command_manager.get_term("base_velocity").vel_command_b``
every step so that the environment never resamples a random command.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Play a Go2 RL policy with keyboard teleop.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Unitree-Go2-Velocity-Robust", help="Name of the task.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--step_v", type=float, default=0.1, help="Per-keypress linear velocity increment.")
parser.add_argument("--step_w", type=float, default=0.2, help="Per-keypress angular velocity increment.")
parser.add_argument("--max_vx", type=float, default=1.0, help="|lin_vel_x| upper bound.")
parser.add_argument("--max_vy", type=float, default=0.4, help="|lin_vel_y| upper bound.")
parser.add_argument("--max_wz", type=float, default=1.0, help="|ang_vel_z| upper bound.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
import os
import time
import weakref

import carb
import numpy as np
import omni
import torch

from isaaclab.envs import DirectMARLEnv, ManagerBasedRLEnv, multi_agent_to_single_agent
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from rsl_rl.runners import OnPolicyRunner

import unitree_rl_lab.tasks  # noqa: F401
from unitree_rl_lab.utils.parser_cfg import parse_env_cfg


# ----------------------------------------------------------------------------- 
# Incremental keyboard
# -----------------------------------------------------------------------------


class IncrementalSe2Keyboard:
    """Keyboard handler with per-press velocity increments.

    Differs from :class:`isaaclab.devices.keyboard.Se2Keyboard` in that the
    command is accumulated on each key-press (the ROS ``keyboard_teleop`` node
    behaves the same way), so brief taps lead to a smooth speed ramp rather
    than dropping back to zero on key-release.
    """

    MODE_STAND = "stand"
    MODE_WALK = "walk"
    MODE_PASSIVE = "passive"

    def __init__(self, step_v: float, step_w: float, max_vx: float, max_vy: float, max_wz: float):
        self.step_v = step_v
        self.step_w = step_w
        self.max_vx = max_vx
        self.max_vy = max_vy
        self.max_wz = max_wz

        self.cmd = np.zeros(3, dtype=np.float32)  # (vx, vy, wz)
        self.mode = self.MODE_WALK
        self.reset_request = False

        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_event(event, *args),
        )

    def __del__(self):
        try:
            self._input.unsubscribe_from_keyboard_events(self._keyboard, self._sub)
        except Exception:
            pass

    def help_str(self) -> str:
        return (
            "\n"
            "Go2 IsaacSim Keyboard Teleop\n"
            "----------------------------\n"
            "  W / S : forward / backward    (+/- step_v on each press)\n"
            "  A / D : strafe left / right\n"
            "  Q / E : yaw  left / right     (+/- step_w on each press)\n"
            "  Space : zero all velocity\n"
            "  1     : STAND   mode (no command, hold default pose)\n"
            "  2     : WALK    mode (apply RL policy with keyboard command)\n"
            "  3     : PASSIVE mode (zero torque)\n"
            "  R     : reset environment\n"
            f"  Limits: |vx|<={self.max_vx} |vy|<={self.max_vy} |wz|<={self.max_wz}\n"
        )

    def _on_event(self, event, *args, **kwargs):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return True
        name = event.input.name
        if name in ("W",):
            self.cmd[0] = float(np.clip(self.cmd[0] + self.step_v, -self.max_vx, self.max_vx))
        elif name in ("S",):
            self.cmd[0] = float(np.clip(self.cmd[0] - self.step_v, -self.max_vx, self.max_vx))
        elif name in ("A",):
            self.cmd[1] = float(np.clip(self.cmd[1] + self.step_v, -self.max_vy, self.max_vy))
        elif name in ("D",):
            self.cmd[1] = float(np.clip(self.cmd[1] - self.step_v, -self.max_vy, self.max_vy))
        elif name in ("Q",):
            self.cmd[2] = float(np.clip(self.cmd[2] + self.step_w, -self.max_wz, self.max_wz))
        elif name in ("E",):
            self.cmd[2] = float(np.clip(self.cmd[2] - self.step_w, -self.max_wz, self.max_wz))
        elif name == "SPACE":
            self.cmd[:] = 0.0
        elif name == "KEY_1" or name == "1":
            self.mode = self.MODE_STAND
            self.cmd[:] = 0.0
            print("[teleop] mode: STAND")
        elif name == "KEY_2" or name == "2":
            self.mode = self.MODE_WALK
            print("[teleop] mode: WALK")
        elif name == "KEY_3" or name == "3":
            self.mode = self.MODE_PASSIVE
            self.cmd[:] = 0.0
            print("[teleop] mode: PASSIVE")
        elif name == "R":
            self.reset_request = True
            self.cmd[:] = 0.0
            print("[teleop] reset")
        else:
            return True
        print(f"[teleop] cmd vx={self.cmd[0]:+.2f} vy={self.cmd[1]:+.2f} wz={self.cmd[2]:+.2f}  mode={self.mode}")
        return True


# ----------------------------------------------------------------------------- 
# Main
# -----------------------------------------------------------------------------


def main():
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
        entry_point_key="play_env_cfg_entry_point",
    )
    agent_cfg = cli_args.parse_rsl_rl_cfg(args_cli.task, args_cli)

    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # cache references
    base_velocity_term = env.unwrapped.command_manager.get_term("base_velocity")
    inner_env: ManagerBasedRLEnv = env.unwrapped
    default_q = inner_env.scene["robot"].data.default_joint_pos.clone()  # (num_envs, num_dofs)

    teleop = IncrementalSe2Keyboard(
        step_v=args_cli.step_v,
        step_w=args_cli.step_w,
        max_vx=args_cli.max_vx,
        max_vy=args_cli.max_vy,
        max_wz=args_cli.max_wz,
    )
    print(teleop.help_str())

    obs = env.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    dt = inner_env.step_dt
    while simulation_app.is_running():
        loop_start = time.time()

        if teleop.reset_request:
            obs, _ = env.reset()
            teleop.reset_request = False

        # ---- override the command so that the env never re-samples random ones ----
        cmd = torch.tensor(teleop.cmd, device=env.unwrapped.device, dtype=torch.float32)
        base_velocity_term.vel_command_b[:, :3] = cmd
        base_velocity_term.is_standing_env[:] = False

        with torch.inference_mode():
            if teleop.mode == teleop.MODE_PASSIVE:
                # zero "delta-from-default" => action_processed = default_q
                # JointPositionAction.processed_actions = default_q + scale * raw, so
                # raw = 0 means hold default pose. Combined with kp=25 / kd=0.5 this
                # is "soft stand" rather than truly zero torque, which is the closest
                # thing we can do at the manager API level without reaching into the
                # actuator. Real "passive" / zero-torque is implemented in the ROS
                # rl_inference_node.cpp instead.
                actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
            elif teleop.mode == teleop.MODE_STAND:
                actions = torch.zeros(env.unwrapped.action_space.shape, device=env.unwrapped.device)
            else:  # walk
                actions = policy(obs)
            obs, _, _, _ = env.step(actions)

        # real-time pacing
        sleep = dt - (time.time() - loop_start)
        if sleep > 0:
            time.sleep(sleep)

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

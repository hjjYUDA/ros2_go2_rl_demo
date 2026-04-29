"""Robust velocity-tracking task for Unitree Go2.

Inherits the official ``RobotEnvCfg`` and applies stronger domain
randomization / perturbation, inspired by the Flamingo two-wheeled-legged
training pipeline at ``Isaac-RL-Two-wheel-Legged-Bot`` (the
``flamingo_env/flat_env/stand_drive`` variant). The goal is a Go2 policy
that:

  * tracks linear/angular velocity commands (walk),
  * stays upright when commanded zero velocity (stand),
  * resists pushes, mass changes and friction changes (anti-disturbance).
"""

from __future__ import annotations

from isaaclab.utils import configclass

from .velocity_env_cfg import RobotEnvCfg, RobotPlayEnvCfg


@configclass
class RobotRobustEnvCfg(RobotEnvCfg):
    """Stronger-disturbance training cfg for Go2 velocity tracking."""

    def __post_init__(self):
        super().__post_init__()

        # --- Stronger periodic push (Flamingo-style) ---
        # Flamingo: interval (13, 15)s, velocity_range x/y +-1.0, z +-1.5
        # We pick a faster interval so the policy is pushed more often.
        self.events.push_robot.interval_range_s = (5.0, 8.0)
        self.events.push_robot.params = {
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }

        # --- Wider base-mass randomization ---
        self.events.add_base_mass.params["mass_distribution_params"] = (-2.5, 5.0)

        # --- Wider friction randomization ---
        self.events.physics_material.params["static_friction_range"] = (0.3, 1.2)
        self.events.physics_material.params["dynamic_friction_range"] = (0.3, 1.0)

        # --- Reset with non-zero base velocity to learn fall-recovery / stabilization ---
        self.events.reset_base.params["velocity_range"] = {
            "x": (-0.5, 0.5),
            "y": (-0.5, 0.5),
            "z": (-0.2, 0.2),
            "roll": (-0.5, 0.5),
            "pitch": (-0.5, 0.5),
            "yaw": (-0.5, 0.5),
        }


@configclass
class RobotRobustPlayEnvCfg(RobotPlayEnvCfg):
    """Play-time variant (smaller scene, full velocity range)."""

    def __post_init__(self):
        super().__post_init__()

        # Keep pushes on during play so we can showcase robustness.
        self.events.push_robot.interval_range_s = (4.0, 6.0)
        self.events.push_robot.params = {
            "velocity_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        }

"""Go2 velocity tracking with **multi-terrain** + **gait-friendly** rewards.

Motivation (issues addressed)
-----------------------------
1. **Sliding at low speed**: With weak ``feet_slide`` / ``feet_air_time`` relative to
   velocity tracking, policies often satisfy small ``v_cmd`` by skidding instead of
   stepping. We strengthen swing/contact shaping and widen the *initial* command
   curriculum band so early training sees meaningful forward speeds.

2. **“Rear legs not learned”**: Usually not separate RL weights per leg—often it is a
   **gait / friction / velocity-tracking shortcut** (front-heavy thrust + slide).
   Stronger foot-slide penalty + air-time bonus + rough terrain forces **all** feet
   to participate. If deployment joint order were wrong you would see systematic
   asymmetry; verify ``deploy.yaml`` ``joint_ids_map`` matches training.

3. **Different terrains**: Replace the mostly-flat cobblestone generator with Isaac
   Lab's ``ROUGH_TERRAINS_CFG``, scaled down for small quadrupeds (same idea as
   ``UnitreeGo2RoughEnvCfg`` in Isaac Lab). **Updates (anti-slide v2)**: taller boxes,
   rougher HF noise, higher stairs, steeper slopes; stronger ``feet_slide`` /
   ``feet_air_time`` / ``air_time_variance`` shaping; slightly softer velocity
   tracking; wider friction randomization; higher ``max_init_terrain_level``.

This builds on ``RobotRobustEnvCfg`` (push / mass / friction DR) and overrides
terrain + reward + command sampling only.
"""

from __future__ import annotations

import math

from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils import configclass

from .velocity_robust_env_cfg import RobotRobustEnvCfg, RobotRobustPlayEnvCfg


def _go2_scaled_rough_generator():
    """Isaac Lab rough terrain pack, scaled for Go2 — **v2 harder** than first gait release."""
    st = ROUGH_TERRAINS_CFG.sub_terrains
    # Boxes: taller discrete obstacles → must lift feet instead of sliding across.
    boxes = st["boxes"].replace(grid_height_range=(0.04, 0.22))
    # HF rough: deeper noise → more unpredictable footholds.
    rough = st["random_rough"].replace(noise_range=(0.018, 0.12), noise_step=0.012)
    # Stairs / slopes: higher steps & steeper ramps (still small-robot safe via curriculum).
    stairs = st["pyramid_stairs"].replace(step_height_range=(0.055, 0.26))
    stairs_inv = st["pyramid_stairs_inv"].replace(step_height_range=(0.055, 0.26))
    slope = st["hf_pyramid_slope"].replace(slope_range=(0.0, 0.48))
    slope_inv = st["hf_pyramid_slope_inv"].replace(slope_range=(0.0, 0.48))

    new_sub = {
        **st,
        "boxes": boxes,
        "random_rough": rough,
        "pyramid_stairs": stairs,
        "pyramid_stairs_inv": stairs_inv,
        "hf_pyramid_slope": slope,
        "hf_pyramid_slope_inv": slope_inv,
    }
    return ROUGH_TERRAINS_CFG.replace(sub_terrains=new_sub)


@configclass
class RobotGaitTerrainRobustEnvCfg(RobotRobustEnvCfg):
    """Robust DR + rough terrain + gait-focused reward tweaks."""

    def __post_init__(self):
        super().__post_init__()

        # --- Terrain: multi-patch generator + curriculum (see RobotEnvCfg.__post_init__) ---
        self.scene.terrain.terrain_generator = _go2_scaled_rough_generator()
        # Higher level → spawn on harder tiles earlier in training (still bounded by grid).
        self.scene.terrain.max_init_terrain_level = 8
        # Slightly more separation between envs when obstacles get taller.
        self.scene.env_spacing = 2.85

        # --- Command curriculum: start closer to a “walking” regime than ±0.1 m/s ---
        # Too-narrow initial ranges encourage skating to track tiny velocities.
        r = self.commands.base_velocity.ranges
        r.lin_vel_x = (-0.35, 0.35)
        r.lin_vel_y = (-0.28, 0.28)
        # yaw rate curriculum already expands via lin_vel_cmd_levels / ang_vel_cmd_levels as configured upstream

        # --- Friction DR: wider range → policy cannot rely on a single friction for “skating”. ---
        self.events.physics_material.params["static_friction_range"] = (0.22, 1.45)
        self.events.physics_material.params["dynamic_friction_range"] = (0.22, 1.2)

        # --- Rewards: stronger anti-slide + swing; softer velocity tracking; symmetric air-time ---
        self.rewards.feet_slide.weight = -0.62
        self.rewards.feet_air_time.weight = 0.38
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.air_time_variance.weight = -1.35  # penalize one-foot-dominant gait harder

        # Softer tracking + slightly lower weight → less incentive to “scrub” ground for exact vx.
        self.rewards.track_lin_vel_xy.weight = 1.35
        self.rewards.track_lin_vel_xy.params["std"] = math.sqrt(0.48)

        # Allow quicker foot motions (motor-side smoothing penalty was dominating low-speed stepping).
        self.rewards.action_rate.weight = -0.048

        # Slightly relax generic joint deviation penalty magnitude—helps alternating stance/swing.
        self.rewards.joint_pos.weight = -0.48

        # On rough terrain the policy needs more torque / energy spikes — ease penalties a hair.
        self.rewards.joint_torques.weight = -1.55e-4
        self.rewards.energy.weight = -1.4e-5

        # More PhysX contact patches for rough meshes (same order of magnitude as Isaac rough examples).
        self.sim.physx.gpu_max_rigid_patch_count = 12 * 2**15


@configclass
class RobotGaitTerrainRobustPlayEnvCfg(RobotRobustPlayEnvCfg):
    """Play config on rough terrain with full command ranges."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = _go2_scaled_rough_generator()
        self.scene.terrain.max_init_terrain_level = None  # random spawn across grid like Isaac rough PLAY cfgs
        self.scene.env_spacing = 2.85
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.events.physics_material.params["static_friction_range"] = (0.22, 1.45)
        self.events.physics_material.params["dynamic_friction_range"] = (0.22, 1.2)
        self.sim.physx.gpu_max_rigid_patch_count = 12 * 2**15

        # Align reward shaping with training variant for realistic play behaviour.
        self.rewards.feet_slide.weight = -0.62
        self.rewards.feet_air_time.weight = 0.38
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.air_time_variance.weight = -1.35
        self.rewards.track_lin_vel_xy.weight = 1.35
        self.rewards.track_lin_vel_xy.params["std"] = math.sqrt(0.48)
        self.rewards.action_rate.weight = -0.048
        self.rewards.joint_pos.weight = -0.48
        self.rewards.joint_torques.weight = -1.55e-4
        self.rewards.energy.weight = -1.4e-5

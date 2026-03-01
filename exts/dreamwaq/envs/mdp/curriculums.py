from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# =============================================================================
# Terrain curriculum
# =============================================================================

def terrain_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    yaw_tracking_tol: float = 0.25,
    min_yaw_cmd_for_tracking: float = 0.25,
) -> torch.Tensor:
    """Terrain curriculum based on how well the robot "covers" the commanded motion.

    - Primary: walk far enough -> move up.
    - Secondary: if yaw command exists and yaw is tracked well -> allow move up even if distance is small
      (useful for "rotate-in-place" commands).

    Notes
    -----
    Works only with TerrainImporter + terrain_type="generator".
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command(command_name)  # shape: (num_envs, 3) in base frame

    env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    # --- distance traveled in XY (world frame)
    distance = torch.norm(asset.data.root_pos_w[env_ids_t, :2] - env.scene.env_origins[env_ids_t, :2], dim=1)

    # --- expected distance based on commanded linear velocity magnitude
    lin_cmd_mag = torch.norm(command[env_ids_t, :2], dim=1)
    # expected_distance = lin_cmd_mag * env.max_episode_length_s
    episode_steps = env.episode_length_buf[env_ids_t].float()
    episode_time  = episode_steps * env.step_dt
    expected_distance = lin_cmd_mag * episode_time

    # --- yaw tracking (base frame yaw rate)
    yaw_vel = asset.data.root_ang_vel_b[env_ids_t, 2]
    yaw_cmd = command[env_ids_t, 2]
    yaw_cmd_active = torch.abs(yaw_cmd) > min_yaw_cmd_for_tracking
    yaw_tracked = torch.abs(yaw_vel - yaw_cmd) < yaw_tracking_tol

    # Move up condition:
    # 1) Walked more than half a tile length in X (terrain tile size[0] is the tile length)
    # 2) OR (yaw cmd is meaningful AND yaw is tracked well)
    level_distance = terrain.cfg.terrain_generator.size[0] / 2.5
    rot_only = lin_cmd_mag < 0.05   # 0.05 m/s -> rotate command 
    move_up = (distance > level_distance) | (rot_only & yaw_cmd_active & yaw_tracked)

    # Move down condition:
    # If robot is commanded to translate and it didn't move enough.
    # (Guard: if lin command is near zero, don't punish by moving down purely due to expected_distance ~ 0.)
    lin_cmd_active = lin_cmd_mag > 1e-3
    move_down = lin_cmd_active & (distance < expected_distance * 0.5)
    move_down &= ~move_up  # never move down if move_up

    # Update terrain levels / origins
    terrain.update_env_origins(env_ids_t, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())


# =============================================================================
# Optional: Event parameter modulation (only works if events are enabled in env cfg)
# =============================================================================

def modify_friction(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    values: list[float],
    interval: int,  # number of "ticks" between changes
    dt: float,
    cold_start_steps: int = 0,
    term_name: str = "physics_material",
):
    """Periodically change friction by updating an EventTerm cfg.

    IMPORTANT: This only has an effect if the corresponding event term exists AND is enabled in env cfg.
    """
    # "common_step_counter" is in sim steps; normalize by dt for a stable schedule
    num_steps = round(env.common_step_counter / dt)
    phase = (num_steps // interval) % len(values)
    value = float(values[int(phase)])

    tol = 3  # change window
    if (num_steps % interval) < tol and num_steps > cold_start_steps:
        term_cfg = env.event_manager.get_term_cfg(term_name)

        old_value, _ = term_cfg.params["static_friction_range"]
        if float(old_value) != value:
            print(
                f"[INFO] --------------- Friction changed from {old_value} to {value} at step {num_steps} ---------------"
            )
            term_cfg.params["static_friction_range"] = (value + 0.05, value + 0.06)
            term_cfg.params["dynamic_friction_range"] = (value + 0.00, value + 0.01)
            env.event_manager.set_term_cfg(term_name, term_cfg)


def curr_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    max_level: int = 2,
    friction_range: tuple[float, float] = (0.1, 2.5),
    restitution_range: tuple[float, float] = (0.0, 1.0),
    mass_range: tuple[float, float] = (-1.0, 10.0),
    external_push_velocity_range: tuple[float, float] = (-0.5, 0.5),
):
    """Example of coupling terrain level with domain randomization magnitude.

    IMPORTANT: Only affects training if the corresponding event terms exist AND are enabled in env cfg.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command(command_name)

    env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    # Terrain progression (same as distance-only baseline)
    distance = torch.norm(asset.data.root_pos_w[env_ids_t, :2] - env.scene.env_origins[env_ids_t, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids_t, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down &= ~move_up
    terrain.update_env_origins(env_ids_t, move_up, move_down)

    # Current curriculum level (mean across env_ids)
    terrain_levels = terrain.terrain_levels.float()
    curr_level = torch.mean(terrain_levels)

    def scale_range(range_tuple, level, max_level, device=None):
        span = float(range_tuple[1] - range_tuple[0])
        if float(level) > float(max_level):
            sub = 0.0
        else:
            sub = span * float(max_level - float(level)) / (2.0 * float(max_level))
        return torch.tensor([range_tuple[0] + sub, range_tuple[1] - sub], device=(device or level.device))

    # event params are often expected on CPU in some backends
    friction_scaled = scale_range(friction_range, curr_level, max_level, device="cpu")
    restitution_scaled = scale_range(restitution_range, curr_level, max_level, device="cpu")
    mass_scaled = scale_range(mass_range, curr_level, max_level, device="cpu")
    push_vel_scaled = tuple(scale_range(external_push_velocity_range, curr_level, max_level).tolist())

    term_updates = [
        (
            "physics_material",
            {
                "static_friction_range": tuple(friction_scaled.tolist()),
                "dynamic_friction_range": (float(friction_scaled[0] - 0.1), float(friction_scaled[1] - 0.1)),
                "restitution_range": tuple(restitution_scaled.tolist()),
            },
        ),
        (
            "add_base_mass",
            {"mass_distribution_params": tuple(mass_scaled.tolist())},
        ),
        (
            "push_robot",
            {"velocity_range": {"x": push_vel_scaled, "y": push_vel_scaled}},
        ),
    ]

    for term_name, params in term_updates:
        term_cfg = env.event_manager.get_term_cfg(term_name)
        assert term_cfg.mode != "startup", f"[ERROR] Term {term_name} is in startup mode. Change it to 'reset'/'interval'."
        term_cfg.params.update(params)
        env.event_manager.set_term_cfg(term_name, term_cfg)

    return curr_level


# =============================================================================
# Velocity command curriculum (matches your env_cfg naming)
# =============================================================================

def lin_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "lin_vel_tracking",
    command_name: str = "base_velocity",
    max_curriculum: float = 1.0,
    increment: float = 0.1,
    success_ratio: float = 0.8,
) -> torch.Tensor:
    """Increase (or keep) linear velocity command ranges based on tracking reward.

    This matches your RewardsCfg naming:
      - lin_vel_tracking: weight=1.0 (example)
    And your CommandsCfg structure:
      - base_velocity.cfg.ranges
      - base_velocity.cfg.limit_ranges
    """
    env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    # episode_sums: (num_envs,) total return for that term over current episode
    episode_sums = env.reward_manager._episode_sums[reward_term_name]
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

    cmd_term = env.command_manager.get_term(command_name)
    ranges = cmd_term.cfg.ranges
    limit_ranges = getattr(cmd_term.cfg, "limit_ranges", None)

    if limit_ranges is None:
        raise AttributeError(f"{command_name} cfg has no 'limit_ranges'. Add it or clamp manually.")

    # Mean per-step reward for the given env_ids
    mean_reward = torch.mean(episode_sums[env_ids_t]) / env.max_episode_length

    # Track cumulative delta for logging if desired
    if not hasattr(env, "delta_lin_vel"):
        env.delta_lin_vel = torch.tensor(0.0, device=env.device)

    if mean_reward > success_ratio * reward_term_cfg.weight:
        delta = torch.tensor([-increment, increment], device=env.device)

        lin_x = torch.tensor(ranges.lin_vel_x, device=env.device)
        lin_y = torch.tensor(ranges.lin_vel_y, device=env.device)

        ranges.lin_vel_x = torch.clamp(
            lin_x + delta,
            min=limit_ranges.lin_vel_x[0],
            max=limit_ranges.lin_vel_x[1],
        ).tolist()
        ranges.lin_vel_y = torch.clamp(
            lin_y + delta,
            min=limit_ranges.lin_vel_y[0],
            max=limit_ranges.lin_vel_y[1],
        ).tolist()

        env.delta_lin_vel = torch.clamp(env.delta_lin_vel + delta[1], 0.0, max_curriculum)

    return env.delta_lin_vel


def ang_vel_cmd_levels(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str = "ang_vel_tracking",
    command_name: str = "base_velocity",
    max_curriculum: float = 1.0,
    increment: float = 0.1,
    success_ratio: float = 0.8,
) -> torch.Tensor:
    """Increase angular velocity command range (ang_vel_z) based on yaw tracking reward."""
    env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    episode_sums = env.reward_manager._episode_sums[reward_term_name]
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

    cmd_term = env.command_manager.get_term(command_name)
    ranges = cmd_term.cfg.ranges
    limit_ranges = getattr(cmd_term.cfg, "limit_ranges", None)

    if limit_ranges is None:
        raise AttributeError(f"{command_name} cfg has no 'limit_ranges'. Add it or clamp manually.")

    mean_reward = torch.mean(episode_sums[env_ids_t]) / env.max_episode_length

    if not hasattr(env, "delta_ang_vel"):
        env.delta_ang_vel = torch.tensor(0.0, device=env.device)

    if mean_reward > success_ratio * reward_term_cfg.weight:
        delta = torch.tensor([-increment, increment], device=env.device)
        ang = torch.tensor(ranges.ang_vel_z, device=env.device)

        ranges.ang_vel_z = torch.clamp(
            ang + delta,
            min=limit_ranges.ang_vel_z[0],
            max=limit_ranges.ang_vel_z[1],
        ).tolist()

        env.delta_ang_vel = torch.clamp(env.delta_ang_vel + delta[1], 0.0, max_curriculum)

    return env.delta_ang_vel


# =============================================================================
# (Optional) Keep legacy-style helper (works but now parameterized)
# =============================================================================

def command_levels_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    command_name: str = "base_velocity",
    max_curriculum: float = 1.0,
    increment: float = 0.1,
) -> torch.Tensor:
    """Legacy helper: expand lin_vel_x/lin_vel_y together if tracking is good."""
    env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    episode_sums = env.reward_manager._episode_sums[reward_term_name]
    reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)

    cmd_term = env.command_manager.get_term(command_name)
    ranges = cmd_term.cfg.ranges
    limit_ranges = getattr(cmd_term.cfg, "limit_ranges", None)

    if limit_ranges is None:
        raise AttributeError(f"{command_name} cfg has no 'limit_ranges'. Add it or clamp manually.")

    delta_range = torch.tensor([-increment, increment], device=env.device)

    if not hasattr(env, "delta_lin_vel"):
        env.delta_lin_vel = torch.tensor(0.0, device=env.device)

    if torch.mean(episode_sums[env_ids_t]) / env.max_episode_length > 0.8 * reward_term_cfg.weight:
        lin_x = torch.tensor(ranges.lin_vel_x, device=env.device)
        lin_y = torch.tensor(ranges.lin_vel_y, device=env.device)

        ranges.lin_vel_x = torch.clamp(
            lin_x + delta_range, limit_ranges.lin_vel_x[0], limit_ranges.lin_vel_x[1]
        ).tolist()
        ranges.lin_vel_y = torch.clamp(
            lin_y + delta_range, limit_ranges.lin_vel_y[0], limit_ranges.lin_vel_y[1]
        ).tolist()

        env.delta_lin_vel = torch.clamp(env.delta_lin_vel + delta_range[1], 0.0, max_curriculum)

    return env.delta_lin_vel


def base_velocity_range(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    max_forward_curriculum: float,
    max_backward_curriculum: float,
    max_lat_curriculum: float,
    increment: float = 0.1,
    command_name: str = "base_velocity",
    reward_term_name: str = "lin_vel_tracking",
):
    """(Optional) Range expansion/reduction using numpy clip logic (kept for format parity)."""
    env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)

    tracking_reward = env.reward_manager._episode_sums[reward_term_name][env_ids_t]
    reward_scale = env.reward_manager.get_term_cfg(reward_term_name).weight

    mean_tracking_reward = torch.mean(tracking_reward) / env.max_episode_length_s
    cmd_term = env.command_manager.get_term(command_name)

    if mean_tracking_reward > 0.8 * reward_scale:
        cmd_term.cfg.ranges.lin_vel_x = (
            np.clip(cmd_term.cfg.ranges.lin_vel_x[0] - increment, -max_backward_curriculum, -0.1),
            np.clip(cmd_term.cfg.ranges.lin_vel_x[1] + increment, 0.1, max_forward_curriculum),
        )
        cmd_term.cfg.ranges.lin_vel_y = (
            np.clip(cmd_term.cfg.ranges.lin_vel_y[0] - increment, -max_lat_curriculum, -0.1),
            np.clip(cmd_term.cfg.ranges.lin_vel_y[1] + increment, 0.1, max_lat_curriculum),
        )

    elif mean_tracking_reward < 0.2 * reward_scale:
        cmd_term.cfg.ranges.lin_vel_x = (
            np.clip(cmd_term.cfg.ranges.lin_vel_x[0] + increment, -max_backward_curriculum, -0.1),
            np.clip(cmd_term.cfg.ranges.lin_vel_x[1] - increment, 0.1, max_forward_curriculum),
        )
        cmd_term.cfg.ranges.lin_vel_y = (
            np.clip(cmd_term.cfg.ranges.lin_vel_y[0] + increment, -max_lat_curriculum, -0.1),
            np.clip(cmd_term.cfg.ranges.lin_vel_y[1] - increment, 0.1, max_lat_curriculum),
        )

    return cmd_term.cfg.ranges.lin_vel_x[-1]

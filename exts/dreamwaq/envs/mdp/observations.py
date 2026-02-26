# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES
#
# This file is based on code from the isaaclab repository:
# https://github.com/isaac-sim/IsaacLab/
#
# The original code is licensed under the BSD 3-Clause License.
# See the `licenses/` directory for details.
#
# This version includes significant modifications.

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.envs.mdp.observations import *
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_index(asset: Articulation, body_name_candidates: list[str] | tuple[str, ...] | str) -> int:
    if isinstance(body_name_candidates, str):
        body_name_candidates = [body_name_candidates]

    for name in body_name_candidates:
        found = asset.find_bodies(name)
        if len(found) > 0:
            return found[0]

    available = getattr(asset.data, "body_names", None)
    raise ValueError(
        f"None of base body candidates {list(body_name_candidates)} found in asset '{asset.cfg.prim_path}'. "
        f"Available bodies: {available}"
    )

# policy observation : body angular velocity, gravity vector in the body frame, 
# body velocity command, joint angle, joint angular velocity, and previous action
# ωt gt ct θt ̇θt at−1
# privileged observation : policy observation, disturbance force applied randomly on the robot’s body,
# height map scan of the robot’s surroundings as an exteroceptive cue
# ot vt dt ht

# =============================================================================
# Policy Observations
# =============================================================================


def body_ang_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_body_names: tuple[str, ...] = ("base", "trunk", "torso", "pelvis", "base_link"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    # 1) prefer explicit root signals if they exist
    root_ang = getattr(asset.data, "root_ang_vel_w", None)
    if root_ang is not None:
        return root_ang.view(env.num_envs, -1)

    # 2) fallback: index into per-body tensor
    base_idx = _get_body_index(asset, base_body_names)
    return asset.data.body_com_ang_vel_w[:, base_idx, :].view(env.num_envs, -1)


def gravity(
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
        ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return (asset.data.projected_gravity_b).view(env.num_envs, -1)


def command(
        env: ManagerBasedRLEnv,
        command_name: str
        ) -> torch.Tensor:
    return env.command_manager.get_command(command_name).view(env.num_envs, -1)


def joint_pos_normalized(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
        ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return (asset.data.joint_pos - asset.data.default_joint_pos).view(env.num_envs, -1)


def joint_vel(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
        ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_vel.view(env.num_envs, -1)

def prev_actions(
        env: ManagerBasedRLEnv
        ) -> torch.Tensor:
    return env.action_manager.action.view(env.num_envs, -1)


# =============================================================================
# Privileged Observations (Fixed for Batch Dimensions)
# =============================================================================
def body_vel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_body_names: tuple[str, ...] = ("base", "trunk", "torso", "pelvis", "base_link"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]

    # 1) prefer explicit root signals if they exist
    root_lin = getattr(asset.data, "root_lin_vel_w", None)
    if root_lin is not None:
        return root_lin.view(env.num_envs, -1)

    # 2) fallback: index into per-body tensor
    base_idx = _get_body_index(asset, base_body_names)
    return asset.data.body_lin_vel_w[:, base_idx, :].view(env.num_envs, -1)

def disturbance_force(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_body_names: tuple[str, ...] = ("base", "trunk", "torso", "pelvis", "base_link"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    force = asset._external_force_b  # type: ignore

    if force.ndim == 2:
        return force.view(env.num_envs, -1)
    base_idx = _get_body_index(asset, base_body_names)
    return force[:, base_idx, :].view(env.num_envs, -1)

def height_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
) -> torch.Tensor:
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    # height = sensor_height - hit_point_z - offset
    return sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset




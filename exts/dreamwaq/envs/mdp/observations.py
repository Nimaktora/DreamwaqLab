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

def _get_robot_indices(asset: Articulation, body_names: list[str] | str) -> list[int]:
    if isinstance(body_names, str):
        body_names = [body_names]
    indices = []
    for name in body_names:
        found = asset.find_bodies(name)
        if len(found) > 0:
            indices.append(found[0])
        else:
            raise ValueError(f"Body name '{name}' not found in asset '{asset.cfg.prim_path}'")
    return indices

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
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
        ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return (asset.data.body_com_ang_vel_w).view(env.num_envs, -1)


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
        ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.body_lin_vel_w.view(env.num_envs, -1)

def disturbance_force(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset._external_force_b.view(env.num_envs, -1) # type: ignore

def height_scan(
        env: ManagerBasedRLEnv, 
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        ) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_state_w[:, 2:3]

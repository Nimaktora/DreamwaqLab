# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import wrap_to_pi
from typing import Sequence, Union

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
def _get_robot(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> Articulation:
    return env.scene[asset_cfg.name]


# def _safe_body_index(asset: Articulation, body_name: str, fallback_regex: str | None = None, default: int = 0) -> int:
#     ids = asset.find_bodies(body_name)
#     if len(ids) > 0:
#         return ids[0]
#     if fallback_regex is not None:
#         ids2 = asset.find_bodies(fallback_regex)
#         if len(ids2) > 0:
#             return ids2[0]
#     return default


# def _get_dt(env: ManagerBasedRLEnv) -> float:
#     if hasattr(env, "step_dt"):
#         return float(env.step_dt)
#     return 1.0 / 60.0 # fallback


# ------------------------------------------------------------
# Reward terms (match the table)
# Each function returns shape: (num_envs,)
# Weight number -> cfg
# ------------------------------------------------------------

# == 1.REWARDS ==
# 1) Lin. velocity tracking: exp{-4 (v_xy^cmd - v_xy)^2}
def reward_lin_vel_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 4.0,
) -> torch.Tensor:
    robot = _get_robot(env, asset_cfg)

    cmd = env.command_manager.get_command(command_name).view(env.num_envs, -1)
    v_cmd_xy = cmd[:, 0:2]

    # base linear velocity (body frame). field명은 IsaacLab 버전에 따라 다를 수 있음
    if hasattr(robot.data, "root_lin_vel_b"):
        v_xy = robot.data.root_lin_vel_b[:, 0:2]
    else:
        # fallback: world frame -> xy만 사용
        v_xy = robot.data.root_lin_vel_w[:, 0:2]

    err = v_cmd_xy - v_xy
    err_sq = torch.sum(err * err, dim=1)
    return torch.exp(-scale * err_sq)


# 2) Ang. velocity tracking: exp{-4 (w_yaw^cmd - w_yaw)^2}
def reward_ang_vel_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    scale: float = 4.0,
) -> torch.Tensor:
    robot = _get_robot(env, asset_cfg)

    cmd = env.command_manager.get_command(command_name).view(env.num_envs, -1)
    # generally, assume that base vel cmd is [vx, vy, wz]
    w_cmd_yaw = cmd[:, 2]

    if hasattr(robot.data, "root_ang_vel_b"):
        w_yaw = robot.data.root_ang_vel_b[:, 2]
    else:
        w_yaw = robot.data.root_ang_vel_w[:, 2]

    err = w_cmd_yaw - w_yaw
    return torch.exp(-scale * (err * err))

# == 2.PENALTY ==

# 3) Linear velocity (z): v_z^2 
def penalty_lin_vel_z(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = _get_robot(env, asset_cfg)

    if hasattr(robot.data, "root_lin_vel_b"):
        v_z = robot.data.root_lin_vel_b[:, 2]
    else:
        v_z = robot.data.root_lin_vel_w[:, 2]
    return v_z * v_z


# 4) Angular velocity (xy): w_xy^2  
def penalty_ang_vel_xy(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = _get_robot(env, asset_cfg)

    if hasattr(robot.data, "root_ang_vel_b"):
        w_xy = robot.data.root_ang_vel_b[:, 0:2]
    else:
        w_xy = robot.data.root_ang_vel_w[:, 0:2]
    return torch.sum(w_xy * w_xy, dim=1)


# 5) Orientation: (||g_xy||^2)
def penalty_orientation_gravity(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = _get_robot(env, asset_cfg)

    # projected gravity in body frame: (N,3)
    if hasattr(robot.data, "projected_gravity_b"):
        g_b = robot.data.projected_gravity_b
    else:
        g_b = torch.zeros((env.num_envs, 3), device=robot.device)
        g_b[:, 2] = -1.0
    # ||g_xy||^2
    return torch.sum(g_b[:, 0:2] * g_b[:, 0:2], dim=1)


# 6) Joint accelerations: qddot^2
def penalty_joint_accel(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = _get_robot(env, asset_cfg)

    if hasattr(robot.data, "joint_acc"):
        qdd = robot.data.joint_acc
    else:
        print("penalty_joint_acc_error")
        return torch.zeros(env.num_envs, device=env.device)
    return torch.sum(qdd * qdd, dim=1)


# 7) Joint power: |tau| * |qdot|
def penalty_joint_power(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = _get_robot(env, asset_cfg)

    tau = robot.data.applied_torque
    qd = robot.data.joint_vel
    return torch.sum(torch.abs(tau) * torch.abs(qd), dim=1)


# 8) Body height: (h_des - h)^2
def penalty_body_height(
    env: ManagerBasedRLEnv,
    h_des: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = _get_robot(env, asset_cfg)

    # root position z
    if hasattr(robot.data, "root_pos_w"):
        h = robot.data.root_pos_w[:, 2]
    else:
        # fallback: base body pos
        h = robot.data.body_pos_w[:, 0, 2]

    err = h_des - h
    return err * err


# 9) Foot clearance : (p_f,z,des - p_f,z)^2 * ||v_f,xy||
# Foot : FL_foot, FR_foot, RL_foot, RR_foot
# foot clearance 계산 시 속도 항을 velocity로할 것인가 speed로 할 것인가?
# 논리적으로는 speed가 맞는 것으로 보이나 논문 상에서는 velocity로 되어 있음?? 머임 일단 난 speed로 햇음
def penalty_foot_clearance(
    env: ManagerBasedRLEnv,
    foot_body_names: Sequence[str],
    clearance_des: Union[Sequence[float], torch.Tensor],  # (F,) or (N,F) only
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = _get_robot(env, asset_cfg)

    # --- body indices ---
    foot_ids: list[int] = []
    for name in foot_body_names:
        ids = robot.find_bodies(name)
        if len(ids) > 0:
            foot_ids.append(int(ids[0][0]))

    if len(foot_ids) == 0:
        ids = robot.find_bodies(".*foot.*")
        foot_ids = [int(i) for i in ids] if len(ids) > 0 else []

    if len(foot_ids) == 0:
        raise RuntimeError(
            "No foot bodies found. Check foot_body_names or robot body names."
        )

    # --- kinematics ---
    foot_pos = robot.data.body_pos_w[:, foot_ids, :]      # (N, F, 3)
    foot_vel = robot.data.body_lin_vel_w[:, foot_ids, :]  # (N, F, 3)

    pz = foot_pos[:, :, 2]                                # (N, F)
    vxy = foot_vel[:, :, 0:2]                             # (N, F, 2)
    speed_xy = torch.linalg.norm(vxy, dim=2)              # (N, F)

    N, F = pz.shape

    # --- normalize clearance_des to shape (N, F) ---
    if isinstance(clearance_des, torch.Tensor):
        if clearance_des.ndim == 1:
            if clearance_des.numel() != F:
                raise ValueError(f"clearance_des shape must be (F,) with F={F}")
            clearance_des_t = clearance_des.to(
                device=pz.device, dtype=pz.dtype
            ).view(1, F).expand(N, F)

        elif clearance_des.ndim == 2:
            if clearance_des.shape != (N, F):
                raise ValueError(
                    f"clearance_des shape must be (N,F)={(N,F)}, got {tuple(clearance_des.shape)}"
                )
            clearance_des_t = clearance_des.to(
                device=pz.device, dtype=pz.dtype
            )

        else:
            raise ValueError("clearance_des tensor must be 1D (F,) or 2D (N,F)")

    else:
        # Sequence[float] -> (F,)
        if len(clearance_des) != F:
            raise ValueError(f"clearance_des length must be F={F}")
        clearance_des_t = torch.tensor(
            clearance_des, device=pz.device, dtype=pz.dtype
        ).view(1, F).expand(N, F)

    # --- compute penalty ---
    err = clearance_des_t - pz                             # (N, F)
    penalty = (err ** 2) * speed_xy                        # (N, F)

    return torch.sum(penalty, dim=1)                       # (N,)


# 10) Action rate: (a_t - a_{t-1})^2
def penalty_action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    a = env.action_manager.action
    a_prev = env.action_manager.prev_action
    diff = a - a_prev
    return torch.sum(diff * diff, dim=1)


# 11) Smoothness: (a_t - 2 a_{t-1} + a_{t-2})^2
def penalty_action_smoothness(env: ManagerBasedRLEnv, action_term_name: str = "joint_pos") -> torch.Tensor:
    term = env.action_manager.get_term(action_term_name)
    # IsaacLab 내부에서 _prev_prev_action을 쓰는 경우가 있어 접근
    a = term.action
    a1 = term.prev_action
    a2 = term._prev_prev_action
    diff2 = a - 2.0 * a1 + a2
    return torch.sum(diff2 * diff2, dim=1)


# 12) Power distribution: var(tau * qdot)^2
def penalty_power_distribution_var(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    unbiased: bool = False,
) -> torch.Tensor:
    robot = _get_robot(env, asset_cfg)

    tau = robot.data.applied_torque
    qd = robot.data.joint_vel
    power = tau * qd  # (N, J)

    # joint-wise variance per env
    var = torch.var(power, dim=-1)
    return torch.square(var)
import os
import torch
import math
import isaaclab.sim as sim_utils
from isaaclab.sim import PhysxCfg, SimulationCfg, RenderCfg 
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass

import isaaclab.terrains as terrain_gen
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

from exts.dreamwaq.tasks.envs.unitree import UNITREE_GO1_CFG
import isaaclab.envs.mdp as mdp_std
import exts.dreamwaq.envs.mdp as mdp
from exts.dreamwaq.utils.terrains_cfg import RailwayTracksTerrainCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
# from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.terrains.config.rough import TerrainGeneratorCfg
from typing import Dict, Tuple, Sequence


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


from isaaclab.assets import Articulation, RigidObject
# if TYPE_CHECKING:
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


# cobblestone road (optional)
COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    # max_init_terrain_level=4,
    # curriculum=True,
    difficulty_range=(0.1, 0.9),
    use_cache=True,
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.1, noise_range=(0.01, 0.06), noise_step=0.01, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.23), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.23), platform_width=2.0, border_width=0.25
        ),
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2, grid_width=0.45, grid_height_range=(0.025, 0.1), platform_width=2.0
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.2,
            step_height_range=(0.0, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "random_tracks": RailwayTracksTerrainCfg(
            proportion=0.1,
            gauge_range=(0.60, 1.80),
            rail_height_range=(0.00, 0.15),
            rail_width_range=(0.04, 0.09),
            sleeper_spacing_range=(0.50, 0.70),
            sleeper_height_range=(0.0, 0.05),
            sleeper_width_range=(0.20, 0.30),
        ),
    },
)


# This cfg is modified version of IsaacLab source terrain.
ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.3,
            step_height_range=(0.05, 0.23),
            step_width=0.3,
            platform_width=3.0,
            border_width=1.0,
            holes=False,
        ),
        # "boxes": terrain_gen.MeshRandomGridTerrainCfg(
        #     proportion=0.2, grid_width=0.45, grid_height_range=(0.05, 0.2), platform_width=2.0
        # ),
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=0.2, noise_range=(0.02, 0.10), noise_step=0.02, border_width=0.25
        ),
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.1, slope_range=(0.0, 0.4), platform_width=2.0, border_width=0.25
        ),
    },
)


##################
# Scene definition
##################
@configclass
class RobotSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",  # "plane", "generator"
        terrain_generator=ROUGH_TERRAINS_CFG,  # None, ROUGH_TERRAINS_CFG
        max_init_terrain_level=1,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    # robots
    robot: ArticulationCfg = UNITREE_GO1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # sensors
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True)
    # lights
    # light = AssetBaseCfg(
    #     prim_path="/World/light",
    #     spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    # )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

##################
# MDP settings
##################

@configclass
class EventCfg:
    """Configuration for events."""
    # pass
    
    # def reset_policy_obs_delay(env, env_ids: Sequence[int], lag_ranges: dict):
    #     """
    #     lag_ranges: {"term_key": (min_lag, max_lag), ...}
    #     """

    #     if isinstance(env_ids, torch.Tensor):
    #         env_ids_list = env_ids.to("cpu").tolist()
    #     else:
    #         env_ids_list = list(env_ids)

    #     bufs = mdp.observations._get_delay_buffers(env)
    #     for key, (min_lag, max_lag) in lag_ranges.items():
    #         max_lag = int(max_lag)
    #         buf = mdp.observations._get_or_create_delay_buf(env, key, max_lag_steps=max_lag)

    #         # 1) 
    #         buf.reset(env_ids_list)  # :contentReference[oaicite:8]{index=8}

    #         # 2)
    #         lags = torch.randint(
    #             low=int(min_lag),
    #             high=int(max_lag) + 1,
    #             size=(len(env_ids_list),),
    #             device=buf.time_lags.device,
    #             dtype=torch.int,
    #         )

            # 3) 
            # buf.set_time_lag(lags, batch_ids=env_ids_list)  # :contentReference[oaicite:9]{index=9}

    # reset_policy_obs_delay = EventTerm(
    #     func=reset_policy_obs_delay,
    #     mode="reset",
    #     min_step_count_between_reset=0,
    #     params={
    #         "lag_ranges": {
    #             "base_ang_vel": (0, 2),
    #             "projected_gravity": (0, 2),
    #             "joint_pos_rel": (0, 1),
    #             "joint_vel_rel": (0, 1),
    #             "velocity_commands": (0, 1),
    #         }
    #     },
    # )

# -------------------
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "static_friction_range": (0.10, 2.5),
            "dynamic_friction_range": (0.1, 2.0),
            "restitution_range": (0.0, 1.00),
            "num_buckets": 64,
        },
    )

    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "mass_distribution_params": (-2.0, 3.0),
            "operation": "add",
        },
    )
    randomize_rigid_body_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )
# -------------------
    # random_joint_params = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="startup",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
    #         "friction_distribution_params": (0.0, 0.05),
    #         "operation": "add",
    #     },
    # )

    random_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        # min_step_count_between_reset=720
        # mode="reset",
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.9, 1.1),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # base_external_force_torque = EventTerm(
    #     func=mdp.apply_external_force_torque,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
    #         "force_range": (-8.0, 8.0),
    #         "torque_range": (-2.0, 2.0),
    #     },
    # )
# -------------------
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "roll": (-0.01, 0.01), "pitch": (-0.01, 0.01), "yaw": (-3.14, 3.14)},
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "position_range": (0.5, 1.5),
            "velocity_range": (0.0, 0.0),
        },
    )

    # # interval
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(8.0, 12.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk"),
            "velocity_range": {"x": (-1.5, 1.5), "y": (-1.5, 1.5)},
        },
    )
# -------------------

@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    # base_velocity = mdp.UniformLevelVelocityCommandCfg(
    #     asset_name="robot",
    #     resampling_time_range=(5.0, 10.0),
    #     rel_standing_envs=0.1,
    #     rel_heading_envs=0.05,
    #     # rel_rotate_only_envs=0.1,
    #     debug_vis=True,standalone/dreamwaq/logs/rsl_rl/dwaq/2026-03-17_17-46-35/model_2750.pt
    #     # ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
    #     #     lin_vel_x=(-0.1, 0.1), lin_vel_y=(-0.1, 0.1), ang_vel_z=(-1, 1)
    #     # ),
    #     limit_ranges=mdp.UniformLevelVelocityCommandCfg.Ranges(
    #         lin_vel_x=(-1.1, 1.1), lin_vel_y=(-0.4, 0.4), ang_vel_z=(-1.0, 1.0)
    #     ),
    # )
    base_velocity = mdp.VelocityCommandWithRotateCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.1,
        rel_heading_envs=0.05,
        rel_rotate_only_envs=0.1,
        heading_command=True,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-1.0, 1.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    JointPositionAction = mdp.JointPositionActionWithHistoryCfg(
        asset_name="robot", joint_names=UNITREE_GO1_CFG.joint_sdk_names, 
        scale={
            ".*_thigh_joint": 0.25,
            ".*_calf_joint" : 0.25,
            ".*hip_joint": 0.5*0.25,                 
        }, 
        use_default_offset=True, clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

# =============================================================================
# Policy Observations
# =============================================================================
# policy observation : body angular velocity, gravity vector in the body frame, 
# body velocity command, joint angle, joint angular velocity, and previous action
# ωt gt ct θt ̇θt at−1

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        base_ang_vel = ObsTerm(
            func=mdp_std.base_ang_vel,
            clip=(-100, 100),
            # scale=0.25,
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )

        projected_gravity = ObsTerm(
            func=mdp_std.projected_gravity,
            clip=(-100, 100),
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )

        velocity_commands = ObsTerm(
            func=mdp_std.generated_commands,
            params={"command_name": "base_velocity"},
            # scale=(2.0, 2.0, 0.25),
            clip=(-100, 100),
        )

        joint_pos_rel = ObsTerm(
            func=mdp_std.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_GO1_CFG.joint_sdk_names, preserve_order=True)
            },
            clip=(-100, 100),
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )

        joint_vel_rel = ObsTerm(
            func=mdp_std.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_GO1_CFG.joint_sdk_names, preserve_order=True)
            },
            clip=(-100, 100),
            # scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )

        last_action = ObsTerm(
            func=mdp_std.last_action,  # was: mdp.last_action
            clip=(-100, 100),
        )  # :contentReference[oaicite:9]{index=9}

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()

# =============================================================================
# Privileged Observations (Fixed for Batch Dimensions)
# =============================================================================
# privileged observation : policy observation, disturbance force applied randomly on the robot’s body,
# height map scan of the robot’s surroundings as an exteroceptive cue
# ot vt dt ht

    @configclass
    class CriticCfg(ObsGroup):
        """Observations for critic group."""

        # policy obs (no noise by default for critic)
        base_ang_vel = ObsTerm(
            func=mdp_std.base_ang_vel,
            # scale=0.25,
            clip=(-100, 100),
        )

        projected_gravity = ObsTerm(
            func=mdp_std.projected_gravity,
            clip=(-100, 100),
        )

        velocity_commands = ObsTerm(
            func=mdp_std.generated_commands,
            params={"command_name": "base_velocity"},
            # scale=(2.0, 2.0, 0.25),
            clip=(-100, 100),
        )

        joint_pos_rel = ObsTerm(
            func=mdp_std.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_GO1_CFG.joint_sdk_names, preserve_order=True)
            },
            clip=(-100, 100),
        )

        joint_vel_rel = ObsTerm(
            func=mdp_std.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=UNITREE_GO1_CFG.joint_sdk_names, preserve_order=True)
            },
            # scale=0.05,
            clip=(-100, 100),
        )

        last_action = ObsTerm(
            func=mdp_std.last_action,  # was: mdp.last_action
            clip=(-100, 100),
        )  # :contentReference[oaicite:9]{index=9}

        # privileged obs
        base_lin_vel = ObsTerm(
            func=mdp_std.base_lin_vel,
            clip=(-100, 100),
        )  # :contentReference[oaicite:16]{index=16}

        disturbance_force = ObsTerm(
            func=mdp.trunk_disturbance_force,
            clip=(-100, 100),
        )  # :contentReference[oaicite:17]{index=17}

        height_scanner = ObsTerm(
            func=mdp_std.height_scan,
            params={
                "sensor_cfg": SceneEntityCfg("height_scanner"),
                # "offset": 0.3,
            },
            # clip=(-2.0, 5.0),
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-4.0, 5.0),
        )  # :contentReference[oaicite:18]{index=18}

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    critic: CriticCfg = CriticCfg()


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # == Rewards ==
    lin_vel_tracking = RewTerm(
        func=mdp_std.track_lin_vel_xy_exp,
        weight=1.5, 
        params={"command_name": "base_velocity",
                "std": math.sqrt(0.25)}
                )

    ang_vel_tracking = RewTerm(
        func=mdp_std.track_ang_vel_z_exp,
        weight=1.0, 
        params={"command_name": "base_velocity",
                "std": math.sqrt(0.25)}
                )

    # == Penalty ==
    lin_vel_z = RewTerm(
        func=mdp_std.lin_vel_z_l2,
        weight=-2.0
        )
    
    ang_vel_xy = RewTerm(
        func=mdp_std.ang_vel_xy_l2, 
        weight=-0.05
        )
    
    orientation_gravity = RewTerm(func=mdp_std.flat_orientation_l2, weight=-0.2)
    joint_accel = RewTerm(func=mdp_std.joint_acc_l2, weight=-2.5e-7)
    joint_power = RewTerm(func=mdp.penalty_joint_power, weight=-2e-5)
    body_height = RewTerm(
            func=mdp_std.base_height_l2, 
            weight=-1.0, 
            params={
                "target_height": 0.3,
                "asset_cfg": SceneEntityCfg("robot", body_names="base")
                # "sensor_cfg": SceneEntityCfg("base_height_scanner"),
            },
        )
    foot_clearance = RewTerm(func=mdp.penalty_foot_clearance, weight=-0.01,
                             params={
                                 "foot_body_names": ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                                 "clearance_des": [-0.2, -0.2, -0.2, -0.2],})
    action_rate = RewTerm(func=mdp_std.action_rate_l2, weight=-0.01)
    action_smoothness = RewTerm(func=mdp.penalty_action_smoothness, weight=-0.01, params={"action_term_name": "JointPositionAction"})
    # power_distribution_var = RewTerm(func=mdp.penalty_power_distribution_var, weight=-1e-6)
    stand_still = RewTerm(func=mdp.penalty_stand_still, weight=-0.2)

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp_std.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp_std.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=["trunk",".*_hip"]), "threshold": 1.0},
    )
    # bad_orientation = DoneTerm(func=mdp_std.bad_orientation, params={"limit_angle": 0.8})

@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)
    # lin_vel_cmd_levels = CurrTerm(func=mdp.lin_vel_cmd_levels)
    # ang_vel_cmd_levels = CurrTerm(func=mdp.ang_vel_cmd_levels)

@configclass
class BaseRobotEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene_cfg = RobotSceneCfg
    # Basic settings
    observations = ObservationsCfg()
    actions = ActionsCfg()
    commands = CommandsCfg()
    # MDP settings
    rewards = RewardsCfg()
    terminations = TerminationsCfg()
    events = EventCfg()
    curriculum = CurriculumCfg()

    

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        # self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.sim.disable_contact_processing = True
        #self.actions.joint_pos.scale = 0.25
        self.sim.physx = PhysxCfg(
                    # [핵심] 0: PGS (Gym과 동일, 빠름), 1: TGS (Isaac Lab 기본, 느림)
                    solver_type=1, 
                    
                    # [핵심] 반복 횟수 명시 (Gym 기본값)
                    min_position_iteration_count=4,
                    min_velocity_iteration_count=1,
                    
                    # [성능] CCD는 뚫림 방지용이나 매우 느립니다. 속도를 위해 끄세요.
                    enable_ccd=False, 
                    
                    # [안정성] 지형 위에서의 떨림 방지
                    enable_stabilization=True, 
                    
                    # [GPU 최적화] GPU 버퍼 및 쓰레드 설정 (추가 권장)
                    #num_threads=4,
                    #enable_pcm=True, 
                )

        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene = self.scene_cfg(num_envs=4096, env_spacing=2.5)
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        # check if terrain levels curriculum is enabled - if so, enable curriculum for terrain generator
        # this generates terrains with increasing difficulty and is useful for training
        if getattr(self.curriculum, "terrain_levels", None) is not None:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = True
        else:
            if self.scene.terrain.terrain_generator is not None:
                self.scene.terrain.terrain_generator.curriculum = False

        # self.scene.terrain.terrain_generator.sub_terrains["boxes"].grid_height_range = (0.025, 0.1)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_range = (0.01, 0.06)
        # self.scene.terrain.terrain_generator.sub_terrains["random_rough"].noise_step = 0.01

        self.sim.physics_material = self.scene.terrain.physics_material
        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = self.sim.dt

        # Set extended domain randomization parameters
        # self.events.physics_material.params["static_friction_range"] = (0.1, 3.16)
        # self.events.physics_material.params["dynamic_friction_range"] = (0.1, 3.0)
        # self.events.physics_material.params["restitution_range"] = (0.0, 1.00)
        # self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 10.0)
        # self.events.physics_material.params["static_friction_range"] = (0.15, 3.16)
        # self.events.physics_material.params["dynamic_friction_range"] = (0.1, 3.0)
        # self.events.physics_material.params["restitution_range"] = (0.0, 0.05)
        # self.events.add_base_mass.params["mass_distribution_params"] = (-2.0, 10.0)
        self.events.add_base_mass.params["recompute_inertia"] = True

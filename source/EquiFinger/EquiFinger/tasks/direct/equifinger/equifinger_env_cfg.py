# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
from typing import Tuple, Optional, Sequence
import os
from pathlib import Path
import math
import yaml
# Third-party
import torch
#import pytorch_kinematics as pk
# import pytorch_kinematics.transforms as tf


from isaaclab_tasks.direct.allegro_hand.allegro_hand_env_cfg import AllegroHandEnvCfg

from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.utils import configclass
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils

# from isaaclab.utils.math import euler_xyz_from_quat
# from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane


# CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# CONFIG_DIR  = f"{CURRENT_DIR}/config"
# ASSETS_DIR = '/home/zgao/EquiFinger/source/EquiFinger/EquiFinger/assets'
# ALLEGRO_URDF_DIR = f"{ASSETS_DIR}/allegro_xela"
# CUBOID_URDF_DIR = f"{ASSETS_DIR}/cuboid_insertion"


# uni michigan allegro hand cfg
ALLEGRO_HAND_UM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/zgao/EquiFinger/source/EquiFinger/EquiFinger/assets/allegro_xela/allegro_hand_right/allegro_hand_right.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            retain_accelerations=False,
            enable_gyroscopic_forces=False,

            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=64 / math.pi * 180.0,
            max_depenetration_velocity=1000.0,
            max_contact_impulse=1e32,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
            sleep_threshold=0.005,
            stabilization_threshold=0.0005,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.02, -0.35, 0.376),
        rot=(0.7071068, 0.0, 0.0, 0.7071068),
        joint_pos={"^(?!allegro_hand_hitosashi_finger_finger_joint_0).*": 0.28, "allegro_hand_hitosashi_finger_finger_joint_0": 0.28},
    ),
    actuators={
        "fingers": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            effort_limit_sim=0.5,
            stiffness=3.0,
            damping=0.1,
            friction=0.01,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


@configclass
class EquifingerEnvCfg(DirectRLEnvCfg):
    # Simulation & runtime
    decimation = 2                    # control at sim_dt * decimation
    episode_length_s = 10.0           # per-episode horizon (seconds)

    # Spaces (update if you change obs/action later)
    action_space = 16                 # Allegro has 16 actuated joints, 4 are fixed?
    observation_space = 50            # joint pos/vel + object pose/vel, etc.  12*2 + 1*2 +12 = 36?
    state_space = 0                   # no privileged state by default

    # Physics stepping
    sim: SimulationCfg = SimulationCfg(
        # device="cuda:0",  # GPU mode; PhysX GPU and pipeline are chosen automatically
        dt=1. / 120,
        render_interval=decimation,
        physx=PhysxCfg(
            solver_type=1,
            max_position_iteration_count=4,
            max_velocity_iteration_count=0,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2 ** 23,
            # other GPU buffer sizes
        )
    )
    # /home/zgao/cognarai/src/cognarai/mpc/mfr/models/allegro_xela/allegro_hand_right/allegro_hand_right.usd
    robot_cfg: ArticulationCfg = ALLEGRO_HAND_UM_CFG.replace(prim_path="/World/envs/env_.*/Robot")


    # Allegro hand with Xela sensors
    actuated_joint_names: list[str] = [
        'allegro_hand_hitosashi_finger_finger_joint_0',
        'allegro_hand_naka_finger_finger_joint_4',
        'allegro_hand_kusuri_finger_finger_joint_8',
        'allegro_hand_oya_finger_joint_12',
        'allegro_hand_hitosashi_finger_finger_joint_1',
        'allegro_hand_hitosashi_finger_finger_joint_2',
        'allegro_hand_hitosashi_finger_finger_joint_3',
        'allegro_hand_naka_finger_finger_joint_5',
        'allegro_hand_naka_finger_finger_joint_6',
        'allegro_hand_naka_finger_finger_joint_7',
        'allegro_hand_kusuri_finger_finger_joint_9',
        'allegro_hand_kusuri_finger_finger_joint_10',
        'allegro_hand_kusuri_finger_finger_joint_11',
        'allegro_hand_oya_finger_joint_13',
        'allegro_hand_oya_finger_joint_14',
        'allegro_hand_oya_finger_joint_15',
    ]
    fingers: list[str] = [] #['index', 'middle', 'thumb'] # 'ring'
    fingertip_body_names: list[str] = [
        "allegro_hand_hitosashi_finger_finger_0_aftc_base_link",
        "allegro_hand_naka_finger_finger_1_aftc_base_link",
        "allegro_hand_kusuri_finger_finger_2_aftc_base_link",
        "allegro_hand_oya_finger_3_aftc_base_link",
    ]
    finger_ee_names: dict[str, list[str]] = {
        'index': fingertip_body_names[0],
        'middle': fingertip_body_names[1],
        'ring': fingertip_body_names[2],
        'thumb': fingertip_body_names[3],
    }

    # Scene cloning (vectorized envs)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=2048, env_spacing=0.75, replicate_physics=True, #clone_in_fabric=True
    )

    # in-hand object
    valve_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Valve",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/home/zgao/EquiFinger/source/EquiFinger/EquiFinger/assets/valve/valve_cross/valve_cross.usd",
            activate_contact_sensors=False,
        ),

        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0, 0.4),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={
                "wall_valve_mount": 0.0,
            },
        ),
        actuators={
            "valve_screw": ImplicitActuatorCfg(
                joint_names_expr=["wall_valve_mount"],
                effort_limit_sim=300.0,
                stiffness=0.0,
                damping=0.1, # add some damping to stabilize
            ),
        },
    )


    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = -10.0
    # rot_reward_scale = 1.0
    # rot_eps = 0.1

    action_penalty_scale = -0.0002
    reach_goal_bonus = 250

    vel_obs_scale = 0.2
    success_tolerance = 0.05

    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 1.0
    force_torque_obs_scale = 10.0


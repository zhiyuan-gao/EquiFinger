# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
import numpy as np
import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

import pytorch_kinematics.transforms as tf
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate

from .equifinger_env_cfg import EquifingerEnvCfg


class EquifingerEnv(DirectRLEnv):
    """Allegro-hand manipulation task in Isaac Lab's direct RL workflow.

    Observations
    -----------
    Concatenated vector with:
    - Allegro joint positions/velocities (normalized)
    - Object pose (p, q) and velocities
    - (Optional) fingertip poses

    Actions
    -------
    - By default: joint position deltas in [-1, 1], scaled to per-joint range.
      Switch to velocity control by changing `_apply_action`.
    """
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: EquifingerEnvCfg

    def __init__(self, cfg: EquifingerEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_hand_dofs = self.hand.num_joints

        # buffers for position targets
        self.hand_dof_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.prev_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)
        self.cur_targets = torch.zeros((self.num_envs, self.num_hand_dofs), dtype=torch.float, device=self.device)

        # list of actuated joints
        self.actuated_dof_indices = list()
        for joint_name in cfg.actuated_joint_names:
            self.actuated_dof_indices.append(self.hand.joint_names.index(joint_name))
        self.actuated_dof_indices.sort()

        # finger bodies
        self.finger_bodies = list()
        for body_name in self.cfg.fingertip_body_names:
            self.finger_bodies.append(self.hand.body_names.index(body_name))
        self.finger_bodies.sort()
        self.num_fingertips = len(self.finger_bodies)

        # joint limits
        joint_pos_limits = self.hand.root_physx_view.get_dof_limits().to(self.device)
        self.hand_dof_lower_limits = joint_pos_limits[..., 0]
        self.hand_dof_upper_limits = joint_pos_limits[..., 1]

        # track successes
        self.successes = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

        # unit tensors
        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.goal_angle = torch.tensor(math.pi / 4, dtype=torch.float, device=self.device).repeat((self.num_envs, 1))  # 45 degrees


    # ---- Scene construction -------------------------------------------------

    def _setup_scene(self):
 
        self.hand = Articulation(self.cfg.robot_cfg)
        self.valve = Articulation(self.cfg.valve_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate (no need to filter for this environment)
        self.scene.clone_environments(copy_from_source=False)

        # add articulation to scene - we must register to scene to randomize with EventManager
        self.scene.articulations["robot"] = self.hand
        self.scene.articulations["valve"] = self.valve

        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clone()

    def _apply_action(self) -> None:
        self.cur_targets[:, self.actuated_dof_indices] = scale(
            self.actions,
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )
        self.cur_targets[:, self.actuated_dof_indices] = (
            self.cfg.act_moving_average * self.cur_targets[:, self.actuated_dof_indices]
            + (1.0 - self.cfg.act_moving_average) * self.prev_targets[:, self.actuated_dof_indices]
        )
        self.cur_targets[:, self.actuated_dof_indices] = saturate(
            self.cur_targets[:, self.actuated_dof_indices],
            self.hand_dof_lower_limits[:, self.actuated_dof_indices],
            self.hand_dof_upper_limits[:, self.actuated_dof_indices],
        )

        self.prev_targets[:, self.actuated_dof_indices] = self.cur_targets[:, self.actuated_dof_indices]

        self.hand.set_joint_position_target(
            self.cur_targets[:, self.actuated_dof_indices], joint_ids=self.actuated_dof_indices
        )

    def _compute_intermediate_values(self):

        # TODO: try different obervations in the future

        self.hand_dof_pos = self.hand.data.joint_pos
        self.hand_dof_vel = self.hand.data.joint_vel

        self.valve_dof_pos = self.valve.data.joint_pos
        self.valve_dof_vel = self.valve.data.joint_vel

    def _get_observations(self) -> dict:
        obs = torch.cat(
            (
                # hand
                unscale(self.hand_dof_pos, self.hand_dof_lower_limits, self.hand_dof_upper_limits),
                self.cfg.vel_obs_scale * self.hand_dof_vel,
                # valve
                # unscale(self.valve_dof_pos, -1.57079632679, 1.57079632679),  #valve_angle_min, valve_angle_max TODO: parametrize
                self.valve_dof_pos, #TODO unscale?
                self.cfg.vel_obs_scale * self.valve_dof_vel,
                self.actions,
            ),
            dim=-1,
        )

        observations = {"policy": obs}
        return observations

    
    def _get_rewards(self):
        self._compute_intermediate_values()
        valve_angle = self.valve.data.joint_pos[:, [0]]  # [N, 1]
  # [N, 1] or scalar broadcastable

        reward, successes = compute_valve_reward(
            valve_angle,
            self.goal_angle,
            self.actions,
            self.cfg.dist_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.reach_goal_bonus,
            self.cfg.success_tolerance,
        )

        self.successes += successes  # ✅ 不再报 shape mismatch
        return reward


    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:


        terminated = self.valve.data.joint_pos[:, 0] > 0.785398  # 45 degrees  TODO: parametrize
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        if truncated.sum().item() > 0:
            print("Episode truncated!")
            print('rotate ',self.valve.data.joint_pos[:, 0])
        if terminated.sum().item() > 0:
            print("Episode terminated due to success!")
            print('rotate ',self.valve.data.joint_pos[:, 0])

        return terminated, truncated
    # ---- Reset logic --------------------------------------------------------

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.hand._ALL_INDICES
        super()._reset_idx(env_ids)

        # reset valve
        valve_angle_noise = sample_uniform(-1.0, 1.0, (len(env_ids), 1), device=self.device)
        valve_angle = self.valve.data.default_joint_pos[env_ids] + 0.1 * valve_angle_noise  #TODO: parametrize noise range of initial valve angle, now it is +/-0.1 rad, around 5.7 degrees
        valve_vel = torch.zeros_like(valve_angle)

        self.valve.write_joint_state_to_sim(valve_angle, valve_vel, env_ids=env_ids)

        # reset hand
        delta_max = self.hand_dof_upper_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]
        delta_min = self.hand_dof_lower_limits[env_ids] - self.hand.data.default_joint_pos[env_ids]

        dof_pos_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        rand_delta = delta_min + (delta_max - delta_min) * 0.5 * dof_pos_noise
        dof_pos = self.hand.data.default_joint_pos[env_ids] + self.cfg.reset_dof_pos_noise * rand_delta

        dof_vel_noise = sample_uniform(-1.0, 1.0, (len(env_ids), self.num_hand_dofs), device=self.device)
        dof_vel = self.hand.data.default_joint_vel[env_ids] + self.cfg.reset_dof_vel_noise * dof_vel_noise

        self.prev_targets[env_ids] = dof_pos
        self.cur_targets[env_ids] = dof_pos
        self.hand_dof_targets[env_ids] = dof_pos

        self.hand.set_joint_position_target(dof_pos, env_ids=env_ids)
        self.hand.write_joint_state_to_sim(dof_pos, dof_vel, env_ids=env_ids)

        self.successes[env_ids] = 0
        self._compute_intermediate_values()




@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)


@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention

@torch.jit.script
def compute_valve_reward(
    valve_angle: torch.Tensor,
    goal_angle: torch.Tensor,
    actions: torch.Tensor,
    dist_reward_scale: float,
    action_penalty_scale: float,
    reach_goal_bonus: float,
    success_tolerance: float,
):
    # 1. 角度误差
    distance2goal = valve_angle - goal_angle         # [N, 1]

    # 2. 基础 reward
    reward = dist_reward_scale * torch.pow(distance2goal, 2).squeeze(-1)  # → [N]

    # 3. 动作惩罚
    reward += action_penalty_scale * torch.sum(actions**2, dim=-1)  # [N]

    # 4. 成功检测
    success_mask = torch.abs(distance2goal).squeeze(-1) < success_tolerance  # [N]
    reward = torch.where(success_mask, reward + reach_goal_bonus, reward)

    return reward, success_mask.float()

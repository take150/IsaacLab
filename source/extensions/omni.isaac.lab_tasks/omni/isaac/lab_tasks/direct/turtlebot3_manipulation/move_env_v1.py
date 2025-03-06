# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
import math
import os
from typing import Tuple

from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
import omni.isaac.lab.utils.math as math_utils
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
from omni.isaac.lab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.sensors import FrameTransformerCfg, FrameTransformer
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg

ASSET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../"))

@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_leftcaster_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names="caster_back_left_link"),
          "static_friction_range": (0.0, 0.0),
          "dynamic_friction_range": (0.0, 0.0),
          "restitution_range": (0.5, 0.5),
          "num_buckets": 1,
      },
    )

    reset_rightcaster_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names="caster_back_right_link"),
          "static_friction_range": (0.0, 0.0),
          "dynamic_friction_range": (0.0, 0.0),
          "restitution_range": (0.5, 0.5),
          "num_buckets": 1,
      },
    )

    reset_leftwheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names="wheel_left_link"),
          "static_friction_range": (1.2, 1.2),
          "dynamic_friction_range": (1.2, 1.2),
          "restitution_range": (0.5, 0.5),
          "num_buckets": 1,
      },
    )
    
    reset_rightwheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names="wheel_right_link"),
          "static_friction_range": (1.2, 1.2),
          "dynamic_friction_range": (1.2, 1.2),
          "restitution_range": (0.5, 0.5),
          "num_buckets": 1,
      },
    )

    reset_leftfinger_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names="gripper_left_link"),
          "static_friction_range": (1.0, 1.0),
          "dynamic_friction_range": (1.0, 1.0),
          "restitution_range": (0.5, 0.5),
          "num_buckets": 1,
      },
    )

    reset_rightfinger_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names="gripper_right_link"),
          "static_friction_range": (1.0, 1.0),
          "dynamic_friction_range": (1.0, 1.0),
          "restitution_range": (0.5, 0.5),
          "num_buckets": 1,
      },
    )

    # reset_object_friction = EventTerm(
    #     func=mdp.randomize_rigid_body_material,
    #     mode="reset",
    #     params={
    #       "asset_cfg": SceneEntityCfg("cube", body_names="object"),
    #       "static_friction_range": (1.0, 1.0),
    #       "dynamic_friction_range": (1.0, 1.0),
    #       "restitution_range": (0.5, 0.5),
    #       "num_buckets": 1,
    #   },
    # )

@configclass
class Turtlebot3MoveEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 7.2  # 1500 timesteps
    decimation = 2
    action_space = 7
    observation_space = 18
    state_space = 0
    seed = 42

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=decimation,
        disable_contact_processing=True,
        physx = sim_utils.PhysxCfg(
            bounce_threshold_velocity = 0.01,
            gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4,
            gpu_total_aggregate_pairs_capacity = 16 * 1024,
            friction_correlation_distance = 0.00625,
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.8,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=2.5, replicate_physics=True)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ASSET_ROOT, "omni.isaac.lab_assets/data/Robots/Turtlebot3_manipulation/turtlebot3_manipulation_merge_02.usd"),
            # usd_path=f"{ASSET_ROOT}/omni.isaac.lab_assets/data/Robots/Turtlebot3_manipulation/turtlebot3_manipulation.usd",
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": 0.0,
                "joint2": 0.0,
                "joint3": 0.0,
                "joint4": 0.0,
                "gripper_left_joint": -0.01,
                "gripper_right_joint": -0.01,
                "wheel_left_joint": 0.0,
                "wheel_right_joint": 0.0,
            },
            pos=(0.0, 1.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "turtlebot3_arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-4]"],
                effort_limit=4.1,
                velocity_limit=2.0,
                stiffness=40.0,
                damping=0.5,
            ),
            "turtlebot3_gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper_.*"],
                effort_limit=200.0,
                velocity_limit=0.2,
                stiffness=2e3,
                damping=1e2,
            ),
            "turtlebot3_wheel": ImplicitActuatorCfg(
                joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
                effort_limit=4.1,
                velocity_limit=4.8,
                stiffness=0.0,
                damping=0.5,
            ),
        },
    )
    
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",

        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ASSET_ROOT, "omni.isaac.lab_assets/data/Objects/red_cube.usd"),
            scale=(0.02, 0.02, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    goal: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal",

        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ASSET_ROOT, "omni.isaac.lab_assets/data/Objects/blue_cube.usd"),
            scale=(0.04, 0.4, 0.04),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    # ground plane
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.8,
        ),
    )

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"
    ee_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_footprint",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/link5",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.126, 0.0, 0.0],
                ),
            ),
        ],
    )
    lee_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_footprint",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/gripper_left_link",
                name="left_end_effector",
                offset=OffsetCfg(
                    pos=[0.045, 0.0, 0.0],
                    # pos=[0.053, -0.005, 0.0],
                ),
            ),
        ],
    )
    ree_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_footprint",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/gripper_right_link",
                name="right_end_effector",
                offset=OffsetCfg(
                    pos=[0.045, 0.0, 0.0],
                    # pos=[0.053, 0.005, 0.0],
                ),
            ),
        ],
    )

    goal_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/goal",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/goal",
                name="robot_goal",
                offset=OffsetCfg(
                    pos=[0.0, 0.38, 0.02],
                    # pos=[0.053, 0.005, 0.0],
                ),
            ),
        ],
    )

    events: EventCfg = EventCfg()

    action_scale = 1.0
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.0
    lift_reward_scale = 3.0
    angle_penalty_scale = 10.0
    dist_g_reward_scale = 5.0

class Turtlebot3MoveEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: Turtlebot3MoveEnvCfg

    def __init__(self, cfg: Turtlebot3MoveEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # シミュレーションの１ステップ時間を設定
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # 各関節の上限と下限の取得
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_vel_limits_tensor = self._robot.data.joint_velocity_limits[0, :].to(device=self.device)
        # ['joint1', 'wheel_left_joint', 'wheel_right_joint', 'joint2', 'joint3', 'joint4', 'gripper_left_joint', 'gripper_right_joint']

        # グリッパーの初期化
        self.robot_grasp_rot = torch.zeros((self.num_envs, 4), device=self.device)
        self.robot_grasp_pos = torch.zeros((self.num_envs, 3), device=self.device)

        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.joint_pos_names = ["joint.*", "gripper_.*"]
        self.arm_names = ["joint.*"]
        self.gripper_names = ["gripper_.*"]
        self.wheel_names = ["wheel_.*"]
        self.joint_pos_ids, _ = self._robot.find_joints(self.joint_pos_names, preserve_order=False)
        self.arm_ids, _ = self._robot.find_joints(self.arm_names, preserve_order=False)
        self.gripper_ids, _ = self._robot.find_joints(self.gripper_names, preserve_order=False)
        self.wheel_ids, _ = self._robot.find_joints(self.wheel_names, preserve_order=False)

        # 各関節の目標位置を初期化
        self.robot_arm_targets = torch.zeros((self.num_envs, len(self.arm_ids)), device=self.device)
        self.robot_gripper_targets = torch.zeros((self.num_envs, len(self.gripper_ids)), device=self.device)
        self.robot_wheel_targets = torch.zeros((self.num_envs, len(self.wheel_ids)), device=self.device)

        self.lift_reward = torch.zeros(self.num_envs, dtype=torch.bool)

    # シーン全体をセットアップ
    def _setup_scene(self):
        # ロボットを初期化
        self._robot = Articulation(self.cfg.robot)
        self._ee_frame = FrameTransformer(self.cfg.ee_frame)
        self._lee_frame = FrameTransformer(self.cfg.lee_frame)
        self._ree_frame = FrameTransformer(self.cfg.ree_frame)
        self._cube = RigidObject(self.cfg.cube)
        self._goal = RigidObject(self.cfg.goal)
        self._goal_frame = FrameTransformer(self.cfg.goal_frame)
        # self.goal_markers = VisualizationMarkers(self.cfg.goal)
        # ロボットをシーンに追加
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["cube"] = self._cube
        self.scene.sensors["ee_frame"] = self._ee_frame
        self.scene.sensors["lee_frame"] = self._lee_frame
        self.scene.sensors["ree_frame"] = self._ree_frame
        self.scene.rigid_objects["goal"] = self._goal
        self.scene.sensors["goal_frame"] = self._goal_frame
        
    
        # 地形の準備
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # 元のシーンを複製
        self.scene.clone_environments(copy_from_source=False)
        # 特定のオブジェクト間での衝突を無効化
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # 証明を追加
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        arm_actions = actions[:, :len(self.arm_ids)].clone().clamp(-1.0, 1.0)
        gripper_action = actions[:, len(self.arm_ids)].clone().clamp(-1.0, 1.0) 
        wheel_actions = actions[:, len(self.arm_ids)+1:].clone().clamp(-1.0, 1.0)

        arm_targets = self._robot.data.joint_pos[:, self.arm_ids] + self.robot_dof_vel_limits_tensor[self.arm_ids] * self.dt * arm_actions
        # arm_targets = self._robot.data.joint_pos[:, self.arm_ids]
        self.robot_arm_targets[:] = torch.clamp(arm_targets, self.robot_dof_lower_limits[self.arm_ids], self.robot_dof_upper_limits[self.arm_ids])

        gripper_actions = torch.zeros(self.num_envs, len(self.gripper_ids), device=self.device)
        gripper_actions[:, 0] = torch.where(gripper_action >= 0.0, self.robot_dof_upper_limits[self.gripper_ids[0]].item(),
                                      self.robot_dof_lower_limits[self.gripper_ids[0]].item())
        gripper_actions[:, 1] = torch.where(gripper_action >= 0.0, self.robot_dof_upper_limits[self.gripper_ids[1]].item(),
                                      self.robot_dof_lower_limits[self.gripper_ids[1]].item())
        self.robot_gripper_targets[:] = gripper_actions

        # wheel_actions[:, 0] = torch.full_like(wheel_actions[:, 0], -1.0)
        # wheel_actions[:, 1] = torch.full_like(wheel_actions[:, 1], 1.0)
        # wheel_actions[self.lift_reward, 0] = 1.0
        # wheel_actions[self.lift_reward, 1] = -1.0

        # print(0, self._robot.data.joint_pos[:, self.arm_ids])
        # print(1, arm_actions)
        # print(2, arm_targets)

        self.robot_wheel_targets[:] = wheel_actions * self.robot_dof_vel_limits_tensor[self.wheel_ids]
        
        # print(self.robot_wheel_targets)

    def _apply_action(self):
        # 制御
        self._robot.set_joint_position_target(self.robot_arm_targets, self.arm_ids)
        self._robot.set_joint_position_target(self.robot_gripper_targets, self.gripper_ids)
        self._robot.set_joint_velocity_target(self.robot_wheel_targets, self.wheel_ids)

    # post-physics step calls
    
    # 終了判定
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        return self._compute_rewards(
            self.ee_pos,
            self.lee_pos,
            self.ree_pos,
            self.cube_pos,
            self.goal_pos,
            self._robot.data.joint_pos[:, self.arm_ids],
            self._robot.data.default_joint_pos[:, self.arm_ids],
            self.robot_dof_upper_limits[self.arm_ids],
            self.robot_dof_lower_limits[self.arm_ids],
            self.cfg.dist_reward_scale,
            self.cfg.lift_reward_scale,
            self.cfg.angle_penalty_scale,
            self.cfg.dist_g_reward_scale,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        joint_pos = torch.clamp(self._robot.data.default_joint_pos[env_ids], self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel = torch.zeros_like(joint_pos)
        default_robot_state = self._robot.data.default_root_state[env_ids].clone()
        default_robot_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_link_pose_to_sim(default_robot_state[:, :7], env_ids=env_ids)
        self._robot.write_root_com_velocity_to_sim(default_robot_state[:, 7:], env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        default_cube_state = self._cube.data.default_root_state[env_ids].clone()
        default_cube_state[:, :3] += self.scene.env_origins[env_ids]
        default_cube_state[:, :3] += self.reset_root_state_uniform(
            env_ids=env_ids,
            pose_range = {"x": (0.4, 0.5), "y": (0.9, 1.1), "z": (0.0, 0.0)},
        )
        self._cube.write_root_link_pose_to_sim(default_cube_state[:, :7], env_ids=env_ids)

        default_goal_state = self._goal.data.default_root_state[env_ids, :7].clone()
        default_goal_state[:, :3] += self.scene.env_origins[env_ids]
        self._goal.write_root_link_pose_to_sim(default_goal_state)

        # Need to refresh the intermediate values so that _get_observations() can use the latest values
        self._compute_intermediate_values(env_ids)
    
    def reset_root_state_uniform(self, env_ids, pose_range):

        # poses
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)

        positions = rand_samples[:, 0:3]

        return positions

    def _get_observations(self) -> dict:

        joint_pos = self._robot.data.joint_pos[:, self.joint_pos_ids]
        joint_vel = self._robot.data.joint_vel[:, self.wheel_ids]

        object_pos_b, object_rot_b = math_utils.subtract_frame_transforms(
            self.base_pos, self.base_rot, self.cube_pos, self.cube_rot
        )
        object_b = torch.cat((object_pos_b, object_rot_b), dim=1)

        goal_pos_b, _ = math_utils.subtract_frame_transforms(
            self.base_pos, self.base_rot, self.goal_pos, self.goal_rot
        )
        # goal_b = torch.cat((goal_pos_b, goal_rot_b), dim=1)

        # print(goal_pos_b)
         
        obs = torch.cat(
            (
                joint_pos,
                joint_vel,
                object_b,
                goal_pos_b,
            ),
            dim=-1,
        )

        return obs

    # ロボットの把持位置、回転を最新状態に更新
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self.base_pos = self._robot.data.root_link_state_w[env_ids, :3]
        self.base_rot = self._robot.data.root_link_state_w[env_ids, 3:7]
        self.cube_pos = self._cube.data.root_link_state_w[env_ids, :3]
        self.cube_rot = self._cube.data.root_link_state_w[env_ids, 3:7]
        self.ee_pos = self._ee_frame.data.target_pos_w[env_ids, 0, :]
        self.lee_pos = self._lee_frame.data.target_pos_w[env_ids, 0, :]
        self.ree_pos = self._ree_frame.data.target_pos_w[env_ids, 0, :]
        self.goal_pos = self._goal_frame.data.target_pos_w[env_ids, 0, :]
        self.goal_rot = self._goal_frame.data.target_quat_w[env_ids, 0, :]

    def _compute_rewards(
        self,
        end_effector_pos,
        left_tip_pos,
        right_tip_pos,
        cube_pos,
        goal_pos,
        joint_pos,
        default_joint_pos,
        joint_pos_upper,
        joint_pos_lower,
        dist_reward_scale,
        lift_reward_scale,
        angle_penalty_scale,
        dist_g_reward_scale,
    ):
        d_c = torch.norm(cube_pos-end_effector_pos, dim=-1)
        d_l = torch.norm(cube_pos-left_tip_pos, dim=-1)
        d_r = torch.norm(cube_pos-right_tip_pos, dim=-1)
        dis = (d_c + d_l + d_r) / 3
        # dis_reward = torch.exp(-10*dis)
        dis_reward_0 = 1 - torch.tanh(10*dis)
        # dis_reward_1 = 1 - torch.tanh(3*dis)

        # lift_reward = 1 - torch.exp(-100*(cube_pos[:, 2]-0.025/2))
        lift_reward = torch.where(cube_pos[:, 2] > 0.04, 1.0, 0.0)
        # lift_reward = torch.where(cube_pos[:, 2] > 0.04, 1.0, 0.0) * torch.where(dis < 0.025, 1.0, 0.0)
        # lift_reward = (cube_pos[:, 2] - 0.025/2) > 0.04

        # self.lift_reward = lift_reward.bool()

        # angle = joint_pos - default_joint_pos

        # angle_1 = angle / joint_pos_upper
        # angle_2 = angle / abs(joint_pos_lower)

        # # print(0, angle)

        # angle = torch.where(angle >= 0, angle_1, angle_2)

        # # print(1, angle)
        
        # angle_penalty = - torch.sum(torch.abs(angle), dim=1) * lift_reward / angle.shape[1]
        
        # dis_g = torch.norm(goal_pos-end_effector_pos, dim=-1)
        # dis_g_reward_0 = (1 - torch.tanh(10*dis_g)) * lift_reward
        # dis_g_reward_1 = (1 - torch.tanh(3*dis_g)) * lift_reward

        # dis_g_reward = torch.exp(-10*dis_g)

        # print(0, dist_g_reward_scale * dis_g_reward_0)
        # print(1, dist_g_reward_scale * dis_g_reward_1)

        # print(dis_reward_0, dis_reward_1)

        rewards = (
            dist_reward_scale * dis_reward_0
            # + dist_reward_scale * dis_reward_1
            + lift_reward_scale * lift_reward
            # + angle_penalty_scale * angle_penalty
            # + dist_g_reward_scale * dis_g_reward_0
            # + dist_g_reward_scale * dis_g_reward_1
        )

        return rewards
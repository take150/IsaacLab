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
import csv
import tqdm

import omni.kit.commands
import isaacsim.core.utils.prims as prims_utils
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg, IdealPDActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import NUCLEUS_ASSET_ROOT_DIR
import isaaclab.utils.math as math_utils
import isaaclab.utils.noise as noise_utils
from isaaclab.utils.math import sample_uniform
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg, FrameTransformer
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg
from isaaclab.sim.schemas import activate_contact_sensors
from isaaclab.utils.buffers import CircularBuffer

ASSET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))

@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # randomize_joints_gain_1 = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["joint1"]),
    #         "stiffness_distribution_params": (380.0, 420.0),
    #         "damping_distribution_params": (38.0, 42.0),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )

    # randomize_joints_gain_2 = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["joint2"]),
    #         "stiffness_distribution_params": (380.0, 420.0),
    #         "damping_distribution_params": (38.0, 42.0),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )

    # randomize_joints_gain_3 = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["joint3"]),
    #         "stiffness_distribution_params": (380.0, 420.0),
    #         "damping_distribution_params": (38.0, 42.0),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )

    # randomize_joints_gain_4 = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot", joint_names=["joint4"]),
    #         "stiffness_distribution_params": (380.0, 420.0),
    #         "damping_distribution_params": (38.0, 42.0),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )

    # randomize_friction_and_armature = EventTerm(
    #     func=mdp.randomize_joint_parameters,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot"),
    #         "friction_distribution_params": (0.05, 0.15),
    #         "armature_distribution_params": (0.005, 0.01),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )

    randomize_leftcaster_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names="caster_back_left_link"),
          "static_friction_range": (0.005, 0.03),
          "dynamic_friction_range": (0.005, 0.02),
          "restitution_range": (0.0, 0.0),
          "num_buckets": 1,
      },
    )

    randomize_rightcaster_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names="caster_back_right_link"),
          "static_friction_range": (0.005, 0.03),
          "dynamic_friction_range": (0.005, 0.02),
          "restitution_range": (0.0, 0.0),
          "num_buckets": 1,
      },
    )

    randomize_leftwheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names="wheel_left_link"),
          "static_friction_range": (0.7, 0.9),
          "dynamic_friction_range": (0.6, 0.7),
          "restitution_range": (0.0, 0.0),
          "num_buckets": 1,
      },
    )
    
    randomize_rightwheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names="wheel_right_link"),
          "static_friction_range": (0.7, 0.9),
          "dynamic_friction_range": (0.6, 0.7),
          "restitution_range": (0.0, 0.0),
          "num_buckets": 1,
      },
    )

    randomize_leftfinger_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names="gripper_left_link"),
          "static_friction_range": (1.1, 1.3),
          "dynamic_friction_range": (0.85, 1.05),
          "restitution_range": (0.5, 0.5),
          "num_buckets": 1,
      },
    )

    randomize_rightfinger_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names="gripper_right_link"),
          "static_friction_range": (1.1, 1.3),
          "dynamic_friction_range": (0.85, 1.05),
          "restitution_range": (0.5, 0.5),
          "num_buckets": 1,
      },
    )


@configclass
class Turtlebot3TuningEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 25.01  # 160 timesteps
    decimation = 25
    state_space = 0
    seed = 42

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.01,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.8,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=100.0, replicate_physics=False)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.MultiUsdFileCfg(
            usd_path=[
                os.path.join(ASSET_ROOT, "isaaclab_assets/data/Robots/Turtlebot3_manipulation/turtlebot3_manipulation_nolidar_collision_test_.usd"),
                # os.path.join(ASSET_ROOT, "isaaclab_assets/data/Robots/Turtlebot3_manipulation/turtlebot3_manipulation_nolidar_collision_01.usd"),
                # os.path.join(ASSET_ROOT, "isaaclab_assets/data/Robots/Turtlebot3_manipulation/turtlebot3_manipulation_nolidar_collision_02.usd"),
                # os.path.join(ASSET_ROOT, "isaaclab_assets/data/Robots/Turtlebot3_manipulation/turtlebot3_manipulation_nolidar_collision_03.usd"),
            ],
            activate_contact_sensors=True,
            random_choice=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "joint1": 0.0,
                "joint2": 0.0,
                "joint3": 0.0,
                "joint4": 0.0,
                "gripper_left_joint": 0.019,
                "gripper_right_joint": 0.019,
                "wheel_left_joint": 0.0,
                "wheel_right_joint": 0.0,
            },
            pos=(0.0, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "turtlebot3_arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-4]"],
                effort_limit_sim=4.1,
                velocity_limit_sim=0.2,
                # stiffness={"joint1": 200, "joint2": 220, "joint3": 50, "joint4": 70},
                # stiffness={"joint1": 200, "joint2": 220, "joint3": 40, "joint4": 80},
                # stiffness={"joint1": 200, "joint2": 220, "joint3": 40, "joint4": 100},
                # damping=4.0,

                # stiffness=200,
                # damping=50.0,
                # friction=0.5,
                # armature=0.01,

                stiffness=100,
                damping=5.0,

                # ここからaction_noizeはFalse
                # stiffness=100,
                # damping=20.0,
                # friction=0.2,
                # armature=0.0075

                # stiffness={"joint1": 200, "joint2": 20, "joint3": 25, "joint4": 200},
                # damping={"joint1": 40, "joint2": 4, "joint3": 5, "joint4": 30},
                # friction=0.05,
                # armature=0.0075

                # stiffness={"joint1": 200, "joint2": 17.5, "joint3": 20, "joint4": 200},
                # damping={"joint1": 40, "joint2": 5, "joint3": 5, "joint4": 40},
                # friction=0.1,
                # armature=0.0075

                # stiffness={"joint1": 200, "joint2": 200, "joint3": 200, "joint4": 200},
                # damping={"joint1": 10, "joint2": 10, "joint3": 10, "joint4": 10},
                # friction=0.05,
                # armature=0.0075
            ),
            "turtlebot3_gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper_.*"],
                effort_limit_sim=4.1,
                velocity_limit_sim=0.02,
                stiffness=2000.0,
                damping=100.0,
                # friction=0.2
            ),
            "turtlebot3_wheel": ImplicitActuatorCfg(
                joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
                effort_limit_sim=4.1,
                velocity_limit_sim=0.8,
                stiffness=0.0,
                damping=300.0,
            ),
        },
    )

    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_footprint/front_cam",
        update_period=0.03,
        offset=TiledCameraCfg.OffsetCfg(pos=(0.076, 0.068, 0.041), rot=(0.99756405, 0.0, 0.06975647, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=0.304, focus_distance=0.927, horizontal_aperture=0.45, vertical_aperture=4.5, clipping_range=(0.01, 20.0)
        ),
        width=120,
        height=120,
    )

    contact_base: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot/link4",
            "/World/envs/env_.*/Robot/link5",
            "/World/envs/env_.*/Robot/gripper_left_link",
            "/World/envs/env_.*/Robot/gripper_right_link",
            ],
    )

    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         # restitution=0.5,
    #     ),
    # )

    ground_plane: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ground",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(10.0, 10.0, 0.01),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3), metallic=0.2),
                ),
            ],
            random_choice=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=1000.0,
                        max_linear_velocity=1000.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,
                    ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    contact_leftgripper_ground: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gripper_left_link",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=["/World/envs/env_.*/ground"],
    )

    contact_rightgripper_ground: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gripper_right_link",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=["/World/envs/env_.*/ground"],
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
                    # pos=[0.126, 0.0, 0.0],
                    pos=[0.146, 0.0, 0.0],
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
                    # pos=[0.045, 0.0, 0.0],
                    pos=[0.065, 0.0, 0.0],
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
                    # pos=[0.045, 0.0, 0.0],
                    pos=[0.065, 0.0, 0.0],
                ),
            ),
        ],
    )

    goal_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_footprint",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/base_link",
                name="goal",
                offset=OffsetCfg(
                    pos=[0.2, 0.05, 0.02],
                ),
            ),
        ],
    )

    events: EventCfg = EventCfg()

    action_space = 7
    observation_space = {"joint": 6, "actions": 7}
    # observation_space = {"joint": 6, "actions": 7, "rgb": [camera.height, camera.width, 3]}
    
    # observation noise
    observation_noise_model = False
    observation_noise_model_joint1: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0, operation="add"),
    )
    observation_noise_model_joint2: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.01, operation="add"),
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0, operation="add"),
    )
    observation_noise_model_joint3: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.01, operation="add"),
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0, operation="add"),
    )
    observation_noise_model_joint4: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.01, operation="add"),
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0, operation="add"),
    )
    observation_noise_model_rgb: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0, operation="add"),
    )    

    # action noise
    action_noise_model = True

    action_noise_model_joint1: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0, operation="add"),
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
    )
    action_noise_model_joint2: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.01, std=0.0025, operation="add"),
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0025, std=0.005, operation="add"),
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
    )
    action_noise_model_joint3: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0075, std=0.0025, operation="add"),
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0025, std=0.005, operation="add"),
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
    )
    action_noise_model_joint4: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0015, operation="add"),
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.005, std=0.0015, operation="add"),
    )
    action_noise_model_wheel: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0, operation="add"),
    )

    action_scale = 1.0
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.0
    action_penalty_scale = -0.15
    self_collision_penalty_scale = -0.25
    contact_ground_penalty_scale = -0.15


class Turtlebot3TuningEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: Turtlebot3TuningEnvCfg

    def __init__(self, cfg: Turtlebot3TuningEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_vel_limits_tensor = self._robot.data.joint_velocity_limits[0, :].to(device=self.device)

        self.joint_pos_names = ["joint1", "joint2", "joint3", "joint4", "gripper_left_joint.*", "gripper_right_joint"]
        self.arm_names = ["joint1", "joint2", "joint3", "joint4"]
        self.gripper_names = ["gripper_left_joint.*", "gripper_right_joint"]
        self.wheel_names = ["wheel_left_joint", "wheel_right_joint"]
        self.joint_pos_ids, _ = self._robot.find_joints(self.joint_pos_names, preserve_order=False)
        self.arm_ids, _ = self._robot.find_joints(self.arm_names, preserve_order=False)
        self.gripper_ids, _ = self._robot.find_joints(self.gripper_names, preserve_order=False)
        self.wheel_ids, _ = self._robot.find_joints(self.wheel_names, preserve_order=False)

        self.joint_1_ids, _ = self._robot.find_joints("joint1", preserve_order=False)
        self.default_joint_1_pos = self._robot.data.default_joint_pos[:, self.joint_1_ids]

        self.robot_arm_targets = torch.zeros((self.num_envs, len(self.arm_ids)), device=self.device)
        self.robot_gripper_targets = torch.zeros((self.num_envs, len(self.gripper_ids)), device=self.device)
        self.robot_wheel_targets = torch.zeros((self.num_envs, len(self.wheel_ids)), device=self.device)

        self.curr_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        self.prev_joint_pos = torch.zeros((self.num_envs, self.cfg.observation_space['joint']), device=self.device)

        # with open('data_log.csv', mode='w', newline='') as f:
        #     writer = csv.writer(f)
        #     # 列名（必要に応じて変更してください）
        #     header = ["ideal_joint1", "ideal_joint2", "ideal_joint3", "ideal_joint4", "real_joint1", "real_joint2", "real_joint3", "real_joint4"]
        #     writer.writerow(header)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        # self._camera = TiledCamera(self.cfg.camera)
        self._contact_base = ContactSensor(self.cfg.contact_base)
        self._contact_leftgripper_ground = ContactSensor(self.cfg.contact_leftgripper_ground)
        self._contact_rightgripper_ground = ContactSensor(self.cfg.contact_rightgripper_ground)
        self._ee_frame = FrameTransformer(self.cfg.ee_frame)
        self._lee_frame = FrameTransformer(self.cfg.lee_frame)
        self._ree_frame = FrameTransformer(self.cfg.ree_frame)
        self._goal_frame = FrameTransformer(self.cfg.goal_frame)
        self._ground_plane = RigidObject(self.cfg.ground_plane)

        self.scene.articulations["robot"] = self._robot
        # self.scene.sensors["camera"] = self._camera
        self.scene.sensors["contact_base"] = self._contact_base
        self.scene.sensors["contact_leftgripper_ground"] = self._contact_leftgripper_ground
        self.scene.sensors["contact_rightgripper_ground"] = self._contact_rightgripper_ground
        self.scene.sensors["ee_frame"] = self._ee_frame
        self.scene.sensors["lee_frame"] = self._lee_frame
        self.scene.sensors["ree_frame"] = self._ree_frame
        self.scene.sensors["goal_frame"] = self._goal_frame
        self.scene.rigid_objects["ground_plane"] = self._ground_plane
    
        # self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        # self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        # self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        # self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # omni.kit.commands.execute(
        #     "ToggleVisibilitySelectedPrims",
        #     selected_paths=["/World/ground"]
        # )

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = torch.nan_to_num(actions, nan=0.0)
        arm_actions = actions[:, :len(self.arm_ids)].clone().clamp(-1.0, 1.0)
        gripper_action = actions[:, len(self.arm_ids)].clone().clamp(-1.0, 1.0) 
        wheel_actions = actions[:, len(self.arm_ids)+1:].clone().clamp(-1.0, 1.0)
        # arm_actions[:, 1] = 0.0
        # arm_actions[:, 2] = -1.0
        # arm_actions[:, 3] = -1.0

        arm_targets = self._robot.data.joint_pos[:, self.arm_ids] + self.robot_dof_vel_limits_tensor[self.arm_ids] * self.dt * arm_actions
        # arm_targets[:, 0] = 0.0
        self.robot_arm_targets[:] = torch.clamp(arm_targets, self.robot_dof_lower_limits[self.arm_ids], self.robot_dof_upper_limits[self.arm_ids])
        gripper_actions = torch.zeros(self.num_envs, len(self.gripper_ids), device=self.device)
        gripper_actions[:, 0] = torch.where(gripper_action >= 0.0, self.robot_dof_upper_limits[self.gripper_ids[0]].item(),
                                      self.robot_dof_lower_limits[self.gripper_ids[0]].item())
        gripper_actions[:, 1] = torch.where(gripper_action >= 0.0, self.robot_dof_upper_limits[self.gripper_ids[1]].item(),
                                      self.robot_dof_lower_limits[self.gripper_ids[1]].item())
        self.robot_gripper_targets[:] = gripper_actions
        
        wheel_actions[:, 0] = torch.full_like(wheel_actions[:, 0], 0.0)
        wheel_actions[:, 1] = torch.full_like(wheel_actions[:, 1], 0.0)

        self.robot_wheel_targets[:] = wheel_actions * self.robot_dof_vel_limits_tensor[self.wheel_ids]

        # self.robot_arm_targets = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=self.device)
        # self.robot_arm_targets = torch.tensor([[0.0, 1.0, -0.5, -0.55]], device=self.device)
        # print(1, self.robot_arm_targets)
        # print(2, self.joint_pos)

        # joint_tensor = torch.cat((self.robot_arm_targets, self.joint_pos), dim=1)
        # data_list = joint_tensor.flatten().tolist()
        # with open('data_log.csv', mode='a', newline='') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(data_list)
        
        # self.robot_arm_targets = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=self.device)
        # self.robot_gripper_targets[:] = torch.tensor([[-0.01, -0.01]], device=self.device)
    
        self.curr_actions = actions

    def _apply_action(self):
        # 制御
        self._robot.set_joint_position_target(self.robot_arm_targets, self.arm_ids)
        self._robot.set_joint_position_target(self.robot_gripper_targets, self.gripper_ids)
        self._robot.set_joint_velocity_target(self.robot_wheel_targets, self.wheel_ids)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = self.episode_length_buf >= self.max_episode_length - 1
        
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        return self._compute_rewards(
            self.ee_pos,
            self.lee_pos,
            self.ree_pos,
            self.goal_pos,
            self.contact_base,
            self.contact_leftgripper_ground,
            self.contact_rightgripper_ground,
            self.cfg.dist_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.self_collision_penalty_scale,
            self.cfg.contact_ground_penalty_scale,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
        #     -0.3,
        #     0.3,
        #     (len(env_ids), self._robot.num_joints),
        #     self.device,
        # )
        # joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        # joint_pos[:, self.joint_1_ids] = 0.0
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = torch.zeros_like(joint_pos)
        default_robot_state = self._robot.data.default_root_state[env_ids].clone()
        default_robot_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_link_pose_to_sim(default_robot_state[:, :7], env_ids=env_ids)
        self._robot.write_root_com_velocity_to_sim(default_robot_state[:, 7:], env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        
        self.curr_actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.prev_joint_pos[env_ids] = 0.0
        
        self._compute_intermediate_values(env_ids)

    def reset_root_state_uniform(
        self,
        env_ids,
        pose_range,
        avoid_radius: float | None = None,
    ):
        keys = ["x", "y", "z", "roll", "pitch", "yaw"]
        range_list = [pose_range.get(k, (0.0, 0.0)) for k in keys]
        ranges = torch.tensor(range_list, device=self.device)

        num_envs = len(env_ids)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (num_envs, 6), device=self.device
        )

        if avoid_radius is not None and avoid_radius > 0.0:
            xy = rand_samples[:, 0:2]
            in_circle = (xy[:, 0] ** 2 + xy[:, 1] ** 2) <= avoid_radius ** 2
            while in_circle.any():
                n_resample = int(in_circle.sum().item())
                xy_new = math_utils.sample_uniform(
                    ranges[0:2, 0], ranges[0:2, 1], (n_resample, 2), device=self.device
                )
                xy[in_circle] = xy_new
                in_circle = (xy[:, 0] ** 2 + xy[:, 1] ** 2) <= avoid_radius ** 2

        positions_delta = rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )

        return positions_delta, orientations_delta

    def _get_observations(self) -> dict:
        # rgb = self._camera.data.output["rgb"] / 255.0

        obs = {
            "joint": self.joint_pos,
            "actions": self.curr_actions,
            # "rgb": rgb
            }

        return obs
    
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self.joint_pos = self._robot.data.joint_pos[:, self.joint_pos_ids]
        
        self.base_pos = self._robot.data.root_link_state_w[env_ids, :3]
        self.base_rot = self._robot.data.root_link_state_w[env_ids, 3:7]
        self.ee_pos = self._ee_frame.data.target_pos_w[env_ids, 0, :]
        self.ee_rot = self._ee_frame.data.target_quat_w[env_ids, 0, :]
        self.lee_pos = self._lee_frame.data.target_pos_w[env_ids, 0, :]
        self.ree_pos = self._ree_frame.data.target_pos_w[env_ids, 0, :]
        self.goal_pos = self._goal_frame.data.target_pos_w[env_ids, 0, :]
        
        self.contact_base = self._contact_base.data.force_matrix_w[env_ids, :]
        self.contact_leftgripper_ground = self._contact_leftgripper_ground.data.force_matrix_w[env_ids, :]
        self.contact_rightgripper_ground = self._contact_rightgripper_ground.data.force_matrix_w[env_ids, :]

    def _compute_rewards(
        self,
        end_effector_pos,
        left_tip_pos,
        right_tip_pos,
        goal_pos,
        contact_base,
        contact_leftgripper_ground,
        contact_rightgripper_ground,
        dist_reward_scale,
        action_penalty_scale,
        self_collision_penalty_scale,
        contact_ground_penalty_scale
    ):
        
        d_c = torch.norm(goal_pos - end_effector_pos, dim=-1)
        d_l = torch.norm(goal_pos - left_tip_pos, dim=-1)
        d_r = torch.norm(goal_pos - right_tip_pos, dim=-1)
        d = (d_c + d_l + d_r) / 3
        dis_reward = torch.exp(-20*d)

        # print(1, goal_pos)
        # print(2, end_effector_pos)

        actions_penalty = self.action_rate_l2_ratio()
       
        contact_base_penalty = torch.norm(contact_base, dim=-1).squeeze() > 1.0
        self_collision_penalty = contact_base_penalty.any(dim=-1)
        
        contact_left_ground_penalty = torch.norm(contact_leftgripper_ground, dim=-1).squeeze() > 0.5
        contact_right_ground_penalty = torch.norm(contact_rightgripper_ground, dim=-1).squeeze() > 0.5
        contact_ground_penalty = contact_left_ground_penalty | contact_right_ground_penalty

        # joint_diff_penalty = self.joint_rate_l2_ratio()
        
        reward = (
            dist_reward_scale * dis_reward
            + self_collision_penalty_scale * self_collision_penalty
            # + contact_ground_penalty_scale * contact_ground_penalty
            + action_penalty_scale * actions_penalty
            # + action_penalty_scale * joint_diff_penalty
        )

        return reward
    
    def action_rate_l2_ratio(self) -> torch.Tensor:
        rate_of_change = (self.curr_actions - self.prev_actions) / 2.0
        self.prev_actions = self.curr_actions        
        return torch.mean(torch.abs(rate_of_change), dim=1)
    
    def joint_rate_l2_ratio(self) -> torch.Tensor:
        rate_of_change = (self.joint_pos - self.prev_joint_pos) / (self.robot_dof_upper_limits[self.joint_pos_ids] - self.robot_dof_lower_limits[self.joint_pos_ids])
        self.prev_joint_pos = self.joint_pos        
        return torch.mean(torch.abs(rate_of_change), dim=1)
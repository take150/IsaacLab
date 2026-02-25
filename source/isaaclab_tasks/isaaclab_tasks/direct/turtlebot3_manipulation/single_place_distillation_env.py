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

    randomize_joints_gain_1 = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint1"]),
            "stiffness_distribution_params": (180.0, 220.0),
            "damping_distribution_params": (18.0, 22.0),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    randomize_joints_gain_2 = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint2"]),
            "stiffness_distribution_params": (270.0, 330.0),
            "damping_distribution_params": (27.0, 33.0),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    randomize_joints_gain_3 = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint3"]),
            "stiffness_distribution_params": (270.0, 330.0),
            "damping_distribution_params": (27.0, 33.0),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    randomize_joints_gain_4 = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=["joint4"]),
            "stiffness_distribution_params": (360.0, 440.0),
            "damping_distribution_params": (36.0, 44.0),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

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

    randomize_object_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("cube", body_names="object"),
          "static_friction_range": (0.7, 0.9),
          "dynamic_friction_range": (0.6, 0.8),
          "restitution_range": (0.5, 0.5),
          "num_buckets": 1,
      },
    )

    randomize_object_color = EventTerm(
        func=mdp.randomize_visual_color,
        mode="interval",
        interval_range_s=(2.5, 4.0),
        is_global_time=True,
        params={
            "event_name": "randomize_object_color",
            "asset_cfg": SceneEntityCfg("cube", body_names="object"),
            "colors":  {"r": (0.2, 0.4), "g": (0.0, 0.1), "b": (0.0, 0.1)},
        }
    )

    randomize_terrain_color = EventTerm(
        func=mdp.randomize_visual_color,
        mode="interval",
        interval_range_s=(2.5, 4.0),
        is_global_time=True,
        params={
            "event_name": "randomize_terrain_color",
            "asset_cfg": SceneEntityCfg("ground_plane", body_names="ground_color"),
            "colors":  {"r": (0.3, 0.6), "g": (0.3, 0.6), "b": (0.3, 0.6)},
        }
    )

    randomize_dome_light = EventTerm(
        func=mdp.randomize_dome_light,
        mode="interval",
        interval_range_s=(2.5, 4.0),
        is_global_time=True,
        params={
            "event_name": "randomize_dome_light",
            "light_paths": ["/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/abandoned_garage_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/blinds_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/blocky_photo_studio_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/boma_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/blue_photo_studio_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/brown_photostudio_01_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/brown_photostudio_02_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/carpentry_shop_02_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/cinema_hall_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/climbing_gym_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/creepy_bathroom_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/empty_play_room_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/gym_01_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/hall_of_finfish_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/hall_of_mammals_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/hangar_interior_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/mirrored_hall_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/metro_noord_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/modern_bathroom_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/photo_studio_loft_hall_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/poly_haven_studio_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/rostock_laage_airport_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/sepulchral_chapel_basement_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/small_empty_room_1_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/small_empty_room_3_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/studio_small_09_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/warm_restaurant_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/warm_restaurant_night_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/whale_skeleton_4k.hdr",
                            "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/zwartkops_pit_4k.hdr",
                            ],
            "light_rotation": [(0.0, 0.0, 0.0), (0.0, 0.0, math.pi * 2)]
        }
    )


@configclass
class Turtlebot3SinglePlaceEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 40.01  # 160 timesteps
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=100.0, replicate_physics=False)

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
                stiffness={"joint1": 200, "joint2": 300, "joint3": 300, "joint4": 400},
                damping={"joint1": 20, "joint2": 30, "joint3": 30, "joint4": 40},
                friction=0.05,
                armature=0.0075
            ),
            "turtlebot3_gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper_.*"],
                effort_limit_sim=4.1,
                velocity_limit_sim=0.02,
                stiffness=2000.0,
                damping=100.0,
            ),
            "turtlebot3_wheel": ImplicitActuatorCfg(
                joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
                effort_limit_sim=4.1,
                velocity_limit_sim=0.8,
                stiffness=0.0,
                damping=6.0,
            ),
        },
    )

    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_footprint/front_cam",
        update_period=0.03,
        offset=TiledCameraCfg.OffsetCfg(pos=(0.076, 0.068, 0.041), rot=(0.99756405,  0.0,  0.06975647,  0.0), convention="world"),
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

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                # sim_utils.CuboidCfg(
                #     size=(0.036, 0.036, 0.036),
                #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                # ),
                sim_utils.CuboidCfg(
                    size=(0.04, 0.04, 0.04),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.041, 0.041, 0.041),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.042, 0.042, 0.042),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                # sim_utils.CuboidCfg(
                #     size=(0.044, 0.044, 0.044),
                #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                # ),
            ],
            random_choice=False,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=1000.0,
                        max_linear_velocity=1000.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,
                    ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.11),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    contact_gripper_object: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/object",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot/gripper_left_link",
            "/World/envs/env_.*/Robot/gripper_right_link",
            ],
    )

    goal: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    # size=(0.1, 0.3, 0.015),
                    size=(0.08, 0.08, 0.034),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
                ),
            ],
            random_choice=False,
            activate_contact_sensors=True,
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
            # mass_props=sim_utils.MassPropertiesCfg(mass=0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.045), rot=(1.0, 0.0, 0.0, 0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.015), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    contact_robot_goal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/goal",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot/base_link",
            "/World/envs/env_.*/Robot/gripper_left_link",
            "/World/envs/env_.*/Robot/gripper_right_link",
            "/World/envs/env_.*/Robot/link1",
            "/World/envs/env_.*/Robot/link2",
            "/World/envs/env_.*/Robot/link3",
            "/World/envs/env_.*/Robot/link4",
            "/World/envs/env_.*/Robot/link5"],
    )

    contact_object_goal: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/object",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/goal"],
    )

    ground_plane: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ground_color",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(1.0, 1.0, 0.01),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(1.3, 1.3, 0.01),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(2.0, 2.0, 0.01),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(4.0, 4.0, 0.01),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(6.0, 6.0, 0.01),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(8.0, 8.0, 0.01),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(10.0, 10.0, 0.01),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
            ],
            random_choice=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    kinematic_enabled=True,
                ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(
                    collision_enabled=False
                ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.005), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            # restitution=0.5,
        ),
    )

    # contact_leftgripper_ground: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/gripper_left_link",
    #     update_period=0.0,
    #     history_length=0,
    #     filter_prim_paths_expr=["/World/ground"],
    # )

    # contact_rightgripper_ground: ContactSensorCfg = ContactSensorCfg(
    #     prim_path="/World/envs/env_.*/Robot/gripper_right_link",
    #     update_period=0.0,
    #     history_length=0,
    #     filter_prim_paths_expr=["/World/ground"],
    # )

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
                    pos=[0.0, 0.0, 0.047],
                ),
            ),
        ],
    )
    default_goal_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*",
                name="robot_goal_",
                offset=OffsetCfg(
                    pos=[0.0, 0.0, 0.08],
                ),
            ),
        ],
    )

    camera_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot/base_footprint",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot/base_footprint",
                name="camera_up",
                offset=OffsetCfg(
                    pos=(0.076, 0.068, 0.041),
                    rot=(0.99756405, 0.0, 0.06975647, 0.0)
                ),
            ),
        ],
    )

    events: EventCfg = EventCfg()

    action_space = 7
    observation_space = {"joint": [5, 6], "object": [5, 14], "actions": [5, 7], "goal": [5, 7], "task_id": 1}
    
    # # observation noise
    # observation_noise_model = True
    # observation_noise_model_joint1: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0025, operation="add"),
    # #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0, operation="add"),
    # )
    # observation_noise_model_joint2: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0025, operation="add"),
    # #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0, operation="add"),
    # )
    # observation_noise_model_joint3: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0025, operation="add"),
    # #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0, operation="add"),
    # )
    # observation_noise_model_joint4: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0025, operation="add"),
    # #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0, operation="add"),
    # )
    # observation_noise_model_rgb: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
    # #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0, operation="add"),
    # )    

    # # action noise
    # action_noise_model = True

    # action_noise_model_joint1: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0025, operation="add"),
    # )
    # action_noise_model_joint2: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0025, std=0.005, operation="add"),
    # )
    # action_noise_model_joint3: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0025, std=0.005, operation="add"),
    # )
    # action_noise_model_joint4: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0025, operation="add"),
    # )
    # action_noise_model_wheel: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0, operation="add"),
    # )

    # observation noise
    observation_noise_model = True
    observation_noise_model_joint1: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
    )
    observation_noise_model_joint2: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
    )
    observation_noise_model_joint3: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
    )
    observation_noise_model_joint4: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
    )
    observation_noise_model_rgb: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
    )    

    # action noise
    action_noise_model = True

    action_noise_model_joint1: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
    )
    action_noise_model_joint2: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.025, std=0.0025, operation="add"),
    )
    action_noise_model_joint3: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.01, std=0.0025, operation="add"),
    )
    action_noise_model_joint4: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.005, std=0.0015, operation="add"),
    )
    action_noise_model_wheel: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.1, operation="add"),
    )

    action_scale = 1.0
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.0
    lift_reward_scale = 1.0
    drop_penalty_scale = -1.0
    dist_g_reward_scale = 1.0
    task_reward_scale = 40.0
    goal_reward_scale = 1.0
    action_penalty_scale = -0.15
    joint_1_penalty_scale = -1.0
    self_collision_penalty_scale = -0.15
    contact_ground_penalty_scale = 0.0
    contact_goal_penalty_scale = -0.15
    touch_penalty_scale = -0.25
    dist_g_penalty_scale = -1.0

class Turtlebot3SinglePlaceEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: Turtlebot3SinglePlaceEnvCfg

    def __init__(self, cfg: Turtlebot3SinglePlaceEnvCfg, render_mode: str | None = None, **kwargs):
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

        self.joint_list = CircularBuffer(max_len=5, batch_size=self.num_envs, device=self.device)  
        self.action_list = CircularBuffer(max_len=5, batch_size=self.num_envs, device=self.device)
        self.object_list = CircularBuffer(max_len=5, batch_size=self.num_envs, device=self.device)  
        self.goal_list = CircularBuffer(max_len=5, batch_size=self.num_envs, device=self.device)  

        self.reward_delay = True
        self.reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.prev_dis_g = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)

        self.gripper_bias = -0.5

        self.lift = torch.full_like(self.episode_length_buf, -1)
        self.reach = torch.full_like(self.episode_length_buf, -1)
        self.drop = torch.full_like(self.episode_length_buf, -1)
        self.unreach = torch.full_like(self.episode_length_buf, -1)
        self.goal = torch.full_like(self.episode_length_buf, -1)
        self.goal_count = torch.zeros_like(self.episode_length_buf, dtype=torch.float32)
        self.task_id = torch.zeros(self.num_envs, device=self.device)
        self.prev_task_id = torch.zeros_like(self.task_id)
        self.total_changes = torch.zeros_like(self.task_id)
        self.log  = torch.ones_like(self.episode_length_buf, dtype=torch.bool)
        self.success_time = torch.full_like(self.episode_length_buf, -1)

        self.csv_path = "success_log.csv"
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "success_count"])
        
        self.episode_idx = 0

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self._camera = TiledCamera(self.cfg.camera)
        self._contact_base = ContactSensor(self.cfg.contact_base)
        self._contact_gripper_object = ContactSensor(self.cfg.contact_gripper_object)
        self._contact_robot_goal = ContactSensor(self.cfg.contact_robot_goal)
        # self._contact_leftgripper_ground = ContactSensor(self.cfg.contact_leftgripper_ground)
        # self._contact_rightgripper_ground = ContactSensor(self.cfg.contact_rightgripper_ground)
        self._ee_frame = FrameTransformer(self.cfg.ee_frame)
        self._lee_frame = FrameTransformer(self.cfg.lee_frame)
        self._ree_frame = FrameTransformer(self.cfg.ree_frame)
        self._camera_frame = FrameTransformer(self.cfg.camera_frame)
        self._cube = RigidObject(self.cfg.cube)
        self._ground_plane = RigidObject(self.cfg.ground_plane)

        self._goal = RigidObject(self.cfg.goal)
        self._contact_object_goal = ContactSensor(self.cfg.contact_object_goal)
        self._goal_frame = FrameTransformer(self.cfg.goal_frame)

        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["camera"] = self._camera
        self.scene.sensors["contact_base"] = self._contact_base
        self.scene.sensors["contact_gripper_object"] = self._contact_gripper_object
        self.scene.sensors["contact_robot_goal"] = self._contact_robot_goal
        # self.scene.sensors["contact_leftgripper_ground"] = self._contact_leftgripper_ground
        # self.scene.sensors["contact_rightgripper_ground"] = self._contact_rightgripper_ground
        self.scene.rigid_objects["cube"] = self._cube
        self.scene.sensors["ee_frame"] = self._ee_frame
        self.scene.sensors["lee_frame"] = self._lee_frame
        self.scene.sensors["ree_frame"] = self._ree_frame
        self.scene.sensors["camera_frame"] = self._camera_frame
        self.scene.rigid_objects["ground_plane"] = self._ground_plane
        
        self.scene.rigid_objects["goal"] = self._goal
        self.scene.sensors["contact_object_goal"] = self._contact_object_goal
        self.scene.sensors["goal_frame"] = self._goal_frame

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        omni.kit.commands.execute(
            "ToggleVisibilitySelectedPrims",
            selected_paths=["/World/ground"]
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = torch.nan_to_num(actions, nan=0.0)
        arm_actions = actions[:, :len(self.arm_ids)].clone().clamp(-1.0, 1.0)
        gripper_action = actions[:, len(self.arm_ids)].clone().clamp(-1.0, 1.0)
        # raw_gripper_action = actions[:, len(self.arm_ids)].clone()
        # bias_tensor = torch.zeros_like(raw_gripper_action)
        # bias_tensor[self.task_id==1] = self.gripper_bias
        # gripper_action = (raw_gripper_action + bias_tensor).clamp(-1.0, 1.0)
        gripper_action[self.task_id==1] = -1.0
        wheel_actions = actions[:, len(self.arm_ids)+1:].clone().clamp(-1.0, 1.0)

        arm_targets = self._robot.data.joint_pos[:, self.arm_ids] + self.robot_dof_vel_limits_tensor[self.arm_ids] * self.dt * arm_actions
        self.robot_arm_targets[:] = torch.clamp(arm_targets, self.robot_dof_lower_limits[self.arm_ids], self.robot_dof_upper_limits[self.arm_ids])

        gripper_actions = torch.zeros(self.num_envs, len(self.gripper_ids), device=self.device)
        gripper_actions[:, 0] = torch.where(gripper_action >= 0.0, self.robot_dof_upper_limits[self.gripper_ids[0]].item(),
                                      self.robot_dof_lower_limits[self.gripper_ids[0]].item())
        gripper_actions[:, 1] = torch.where(gripper_action >= 0.0, self.robot_dof_upper_limits[self.gripper_ids[1]].item(),
                                      self.robot_dof_lower_limits[self.gripper_ids[1]].item())
        self.robot_gripper_targets[:] = gripper_actions
        
        self.robot_wheel_targets[:] = wheel_actions * self.robot_dof_vel_limits_tensor[self.wheel_ids]

        # self.robot_arm_targets = torch.tensor([[0.0, 1.47, -0.83, -0.54]], device=self.device)
        # self.robot_arm_targets = torch.tensor([[0.0, 0.0, 0.0, 0.0]], device=self.device)
        # self.robot_gripper_targets = torch.tensor([[0.019, 0.019]], device=self.device)
        # wheel_actions[:, 0] = torch.full_like(wheel_actions[:, 0], 0.0)
        # wheel_actions[:, 1] = torch.full_like(wheel_actions[:, 1], 0.0)

        self.curr_actions = actions

    def _apply_action(self):
        # 制御
        self._robot.set_joint_position_target(self.robot_arm_targets, self.arm_ids)
        self._robot.set_joint_position_target(self.robot_gripper_targets, self.gripper_ids)
        self._robot.set_joint_velocity_target(self.robot_wheel_targets, self.wheel_ids)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # if self.reward_delay:
        #     truncated = self.episode_length_buf >= (self.max_episode_length - 1) / 2
        # else:
        #     truncated = self.episode_length_buf >= self.max_episode_length - 1

        truncated = self.episode_length_buf >= self.max_episode_length - 1
        # terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        terminated = (self.task_id != self.prev_task_id)

        self.prev_task_id[:] = self.task_id[:]
        self.total_changes += terminated.float()
        
        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        return self._compute_rewards(
            self.ee_pos,
            self.lee_pos,
            self.ree_pos,
            self.joint_1_pos,
            self.cube_pos,
            self.camera_to_object_pos,
            self.goal_pos,
            self.contact_base,
            self.contact_gripper_object,
            self.contact_object_goal,
            self.contact_robot_goal,
            self.cfg.dist_reward_scale,
            self.cfg.lift_reward_scale,
            self.cfg.drop_penalty_scale,
            self.cfg.dist_g_reward_scale,
            self.cfg.goal_reward_scale,
            self.cfg.task_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.joint_1_penalty_scale,
            self.cfg.self_collision_penalty_scale,
            self.cfg.contact_ground_penalty_scale,
            self.cfg.contact_goal_penalty_scale,
            self.cfg.touch_penalty_scale,
            self.cfg.dist_g_penalty_scale,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.5,
            0.5,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        # joint_pos[:, self.arm_ids] = torch.tensor([[0.0, -1.4, 1.4, 0.0]], device=self.device)
        joint_vel = torch.zeros_like(joint_pos)
        default_robot_state = self._robot.data.default_root_state[env_ids].clone()
        default_robot_state[env_ids, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_link_pose_to_sim(default_robot_state[:, :7], env_ids=env_ids)
        self._robot.write_root_com_velocity_to_sim(default_robot_state[:, 7:], env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        default_goal_state = self._goal.data.default_root_state[env_ids, :7].clone()
        default_goal_state[env_ids, :3] += self.scene.env_origins[env_ids]
        # default_goal_state[:, :2] = default_goal_base_state[:, :2]
        
        positions_goal, orientations_goal= self.reset_root_state_uniform(
            env_ids=env_ids,
            # pose_range = {"x": (0.265, 0.265), "y": (-0.0, 0.0), "z": (0.01, 0.01)},
            # pose_range={"x": (-0.25, 0.5), "y": (-0.15, 0.15), "z": (0.01, 0.01), "yaw": (0.0, math.pi/2)},
            # pose_range={"x": (0.5, 0.5), "y": (-0.0, 0.0), "z": (0.01, 0.01), "yaw": (0.0, 0.0)},
            pose_range={"x": (0.6, 0.8), "y": (-0.05, 0.05), "z": (0.001, 0.001), "yaw": (0.0, 0.0)},
            avoid_radius=0.2
        )
        default_goal_state[:, :3] += positions_goal
        default_goal_state[:, 3:7] = math_utils.quat_mul(default_goal_state[:, 3:7], orientations_goal)
        # self._goal_base.write_root_link_pose_to_sim(default_goal_base_state[:, :7], env_ids=env_ids)
        self._goal.write_root_link_pose_to_sim(default_goal_state[:, :7], env_ids=env_ids)

        default_cube_state = self._cube.data.default_root_state[env_ids].clone()
        default_cube_state[env_ids, :3] += self.scene.env_origins[env_ids]
        positions_delta, orientations_delta= self.reset_root_state_uniform(
            env_ids=env_ids,
            # pose_range = {"x": (0.265, 0.265), "y": (-0.0, 0.0), "z": (0.01, 0.01)},
            # pose_range={"x": (-0.25, 0.5), "y": (-0.15, 0.15), "z": (0.01, 0.01), "yaw": (0.0, math.pi/2)},
            # pose_range={"x": (0.5, 0.5), "y": (-0.0, 0.0), "z": (0.07, 0.07), "yaw": (0.0, 0.0)},
            pose_range={"x": (0.3, 0.5), "y": (-0.05, 0.05), "z": (0.01, 0.01), "yaw": (0.0, 0.0)},
            avoid_radius=0.2
        )
        default_cube_state[:, :3] += positions_delta
        default_cube_state[:, 3:7] = math_utils.quat_mul(default_cube_state[:, 3:7], orientations_delta)
        self._cube.write_root_link_pose_to_sim(default_cube_state[:, :7], env_ids=env_ids)
        

        self.joint_list.reset()
        self.action_list.reset()
        self.object_list.reset()
        self.goal_list.reset()

        success_mask = self.success_time[env_ids] >= 0
        if success_mask.any():
            succ_envs = env_ids[success_mask]
            times = self.success_time[succ_envs].float() * 0.25   # 例: 1step=0.25[s]

            mean_t = times.mean().item()
            std_t  = times.std(unbiased=False).item()  # N 分母

            # 1 行だけ出力
            tqdm.tqdm.write(f"[RESET] success (n={times.numel()}) :  {mean_t:.3f} ± {std_t:.3f}  [s]")

            with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([self.episode_idx, times.numel()])
        
        tqdm.tqdm.write(f"[RESET] total_changes (n={self.total_changes.mean().item()})")
        tqdm.tqdm.write(f"[RESET] goal_count (n={self.goal_count.mean().item()})")

        self.episode_idx += 1

        self.lift[env_ids] = -1
        self.reach[env_ids] = -1
        self.drop[env_ids] = -1
        self.unreach[env_ids] = -1
        self.goal[env_ids] = -1
        self.goal_count[env_ids] = 0
        self.task_id[env_ids] = 0
        self.prev_task_id[env_ids] = 0
        self.total_changes[env_ids] = 0
        self.log[env_ids] = True
        self.success_time[env_ids] = -1
        # self.lift = torch.full_like(self.episode_length_buf, -1)  # すべて -1
        # self.log  = torch.ones_like(self.episode_length_buf, dtype=torch.bool)
        # self.success_time = torch.full_like(self.episode_length_buf, -1)

        # if self.reward.mean().item() >= 250:
        #     self.reward_delay = True
        
        # self.gripper_bias = min(-0.8, self.gripper_bias+0.05)
        
        self.curr_actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.reward[env_ids] = 0.0
        self.prev_dis_g[env_ids] = 0.0

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

        rgb = self._camera.data.output["rgb"] / 255.0

        self.joint_list.append(self.joint_pos)
        self.action_list.append(self.curr_actions)
        self.object_list.append(self.object_be)
        self.goal_list.append(self.goal_b)

        joint_list = 2 * (self.joint_list.buffer - self.robot_dof_lower_limits[self.joint_pos_ids]) / (self.robot_dof_upper_limits[self.joint_pos_ids] - self.robot_dof_lower_limits[self.joint_pos_ids]) - 1

        obs = {
            "joint": joint_list,
            "object": self.object_list.buffer,
            "actions": self.action_list.buffer,
            "goal": self.goal_list.buffer,
            "task_id": self.task_id,
            "rgb": rgb,
            }
        
        return obs
    
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self.joint_pos = self._robot.data.joint_pos[:, self.joint_pos_ids]
        self.joint_1_pos = self._robot.data.joint_pos[:, self.joint_1_ids]
        self.joint_acc = self._robot.data.joint_acc[env_ids][:, self.joint_pos_ids]
        
        self.base_pos = self._robot.data.root_link_state_w[env_ids, :3]
        self.base_rot = self._robot.data.root_link_state_w[env_ids, 3:7]
        self.ee_pos = self._ee_frame.data.target_pos_w[env_ids, 0, :]
        self.ee_rot = self._ee_frame.data.target_quat_w[env_ids, 0, :]
        self.lee_pos = self._lee_frame.data.target_pos_w[env_ids, 0, :]
        self.ree_pos = self._ree_frame.data.target_pos_w[env_ids, 0, :]
        self.camera_pos = self._camera_frame.data.target_pos_w[env_ids, 0, :]
        self.camera_rot = self._camera_frame.data.target_quat_w[env_ids, 0, :]
        
        self.cube_pos = self._cube.data.root_link_pos_w[env_ids, :]
        self.cube_rot = self._cube.data.root_link_state_w[env_ids, 3:7]
        
        self.contact_base = self._contact_base.data.force_matrix_w[env_ids, :]
        self.contact_gripper_object = self._contact_gripper_object.data.force_matrix_w[env_ids, :]
        self.contact_robot_goal = self._contact_robot_goal.data.force_matrix_w[env_ids, :]
        self.contact_object_goal = self._contact_object_goal.data.force_matrix_w[env_ids, :]
        
        object_pos_e, object_rot_e = math_utils.subtract_frame_transforms(
            self.ee_pos, self.ee_rot, self.cube_pos, self.cube_rot
        )
        object_pos_b, object_rot_b = math_utils.subtract_frame_transforms(
            self.base_pos, self.base_rot, self.cube_pos, self.cube_rot
        )
        self.object_be = torch.cat((object_pos_b, object_rot_b, object_pos_e, object_rot_e), dim=1)        

        goal_pos = self._goal.data.root_link_state_w[env_ids, :3]
        goal_rot = self._goal.data.root_link_state_w[env_ids, 3:7]

        goal_pos_b, goal_rot_b = math_utils.subtract_frame_transforms(
            self.base_pos, self.base_rot, goal_pos, goal_rot
        )
        self.goal_b = torch.cat((goal_pos_b, goal_rot_b), dim=1)
        self.goal_pos = self._goal_frame.data.target_pos_w[env_ids, 0, :]

        self.camera_to_object_pos, _ = math_utils.subtract_frame_transforms(
            self.camera_pos, self.camera_rot, self.cube_pos, self.cube_rot
        )

    def _compute_rewards(
        self,
        end_effector_pos,
        left_tip_pos,
        right_tip_pos,
        joint_1_pos,
        cube_pos,
        camera_to_object_pos,
        goal_pos,
        contact_base,
        contact_gripper_object,
        contact_object_goal,
        contact_robot_goal,
        dist_reward_scale,
        lift_reward_scale,
        drop_penalty_scale,
        dist_g_reward_scale,
        goal_reward_scale,
        task_reward_scale,
        action_penalty_scale,
        joint_1_penalty_scale,
        self_collision_penalty_scale,
        contact_ground_penalty_scale,
        contact_goal_penalty_scale,
        touch_penalty_scale,
        dist_g_penalty_scale,
    ):
        
        d_c = torch.norm(cube_pos-end_effector_pos, dim=-1)
        d_l = torch.norm(cube_pos-left_tip_pos, dim=-1)
        d_r = torch.norm(cube_pos-right_tip_pos, dim=-1)
        d = (d_c*2 + d_l + d_r) / 4
        dis_reward = torch.exp(-10*d)

        d_g = torch.norm(cube_pos-goal_pos, dim=-1)
        # dis_g_reward = torch.exp(-10*d_g)
        dis_g_reward = torch.exp(-5*d_g) + torch.exp(-30*d_g)
        # dis_g = torch.exp(-5*d_g) + torch.exp(-30*d_g)
        # dis_g_reward = dis_g - self.prev_dis_g
        # self.prev_dis_g = dis_g
        dis_g_penalty = torch.tanh(10*d_g)

        contact_gripper_object_reward = torch.norm(contact_gripper_object, dim=-1).squeeze() > 0.1
        catch_object = contact_gripper_object_reward.any(dim=-1)
        lift_reward = torch.where(cube_pos[:, 2] > 0.03, 1.0, 0.0) * catch_object * torch.where(d_c < 0.025, 1.0, 0.0)

        is_lifted = lift_reward.bool()
        is_dropped = ~(catch_object * torch.where(d_c < 0.025, 1.0, 0.0)).bool()
        # is_dropped = ~is_lifted & ~is_goal

        self.lift[is_dropped] = -1
        new_lift_mask = is_lifted & (self.lift == -1)
        self.lift[new_lift_mask] = self.episode_length_buf[new_lift_mask]
        success_lift_mask = (
            is_lifted &
            (self.lift != -1) &
            ((self.episode_length_buf - self.lift) >= 1)
        )

        is_reached = (d_g < 0.02) & is_lifted
        is_unreached = ~is_reached
        self.reach[is_unreached] = -1
        new_reach_mask = is_reached & (self.reach == -1)
        self.reach[new_reach_mask] = self.episode_length_buf[new_reach_mask]
        success_reach_mask = (
            is_reached &
            (self.reach != -1) &
            ((self.episode_length_buf - self.reach) >= 1)
        )

        is_goal = (
            (contact_object_goal.select(dim=-1, index=2).squeeze() > 0.4) &
            ~catch_object
        )
        
        if is_goal.ndim == 0:
            is_goal = is_goal.unsqueeze(-1)
        
        self.drop[is_lifted | is_goal] = -1
        new_drop_mask = is_dropped & (self.drop == -1)
        self.drop[new_drop_mask] = self.episode_length_buf[new_drop_mask]
        failed_lift_mask = (
            is_dropped &
            (self.drop != -1) &
            ((self.episode_length_buf - self.drop) >= 12)
        )

        self.unreach[is_reached | is_goal] = -1
        new_unreach_mask = is_unreached & (self.unreach == -1)
        self.unreach[new_unreach_mask] = self.episode_length_buf[new_unreach_mask]
        failed_reach_mask = (
            is_unreached &
            (self.unreach != -1) &
            ((self.episode_length_buf - self.unreach) >= 12)
        )      

        goal_reward = is_goal

        new_goal_mask = is_goal & (self.goal == -1)
        self.goal[new_goal_mask] = self.episode_length_buf[new_goal_mask]
        success_goal_mask = (
            is_goal &
            (self.goal != -1) &
            ((self.episode_length_buf - self.goal) >= 8) &
            self.log
        )          

        # success_log = success_mask & self.log
        self.success_time[success_goal_mask] = self.episode_length_buf[success_goal_mask]
        self.log[success_goal_mask] = False

        task_id = self.task_id.squeeze(-1)
        task_0 = (task_id == 0)
        task_1 = (task_id == 1)   
        task_2 = (task_id == 2)

        # if success_mask.any():
        #     env_ids = torch.nonzero(success_mask, as_tuple=False).squeeze(-1)
        #     for eid in env_ids.tolist():
        #         print(f"env {eid}: success")
        #         print((self.episode_length_buf[eid] * 0.25).item())
        #     self.log[env_ids] = False

        actions_penalty = self.action_rate_l2_ratio()
        # joint_1_penalty = torch.abs(joint_1_pos - self.default_joint_1_pos).squeeze()
        joint_1_penalty = torch.abs(torch.atan2(camera_to_object_pos[:, 1], camera_to_object_pos[:, 0]))

        contact_base_penalty = torch.norm(contact_base, dim=-1).squeeze() > 1.0
        self_collision_penalty = contact_base_penalty.any(dim=-1)

        contact_goal = torch.norm(contact_robot_goal, dim=-1).squeeze() > 1.0
        contact_goal_penalty = contact_goal.any(dim=-1)

        on_goal = contact_object_goal.select(dim=-1, index=2).squeeze() > 0.4
        
        if on_goal.ndim == 0:
            on_goal = on_goal.unsqueeze(-1)

        # contact_left_ground_penalty = torch.norm(contact_leftgripper_ground, dim=-1).squeeze() > 1.0
        # contact_right_ground_penalty = torch.norm(contact_rightgripper_ground, dim=-1).squeeze() > 1.0
        # contact_ground_penalty = contact_left_ground_penalty | contact_right_ground_penalty

        reward = (
            self_collision_penalty_scale * self_collision_penalty
            + action_penalty_scale * actions_penalty
            + joint_1_penalty_scale * joint_1_penalty
            + contact_goal_penalty_scale * contact_goal_penalty
        )

        # reward[task_0] += (
        #     dist_reward_scale * dis_reward[task_0]
        #     + lift_reward_scale * lift_reward[task_0]
        #     + task_reward_scale * success_mask[task_0]
        # )
        reward[task_0] = 0

        # reward[task_1] += (
        #     dist_g_reward_scale * dis_g_reward[task_1] * is_lifted[task_1]
        #     + drop_penalty_scale * is_dropped[task_1].float()
        # )
        reward[task_1] = 0

        self.goal_count += goal_reward

        reward[task_2 & on_goal] += (
            touch_penalty_scale * catch_object[task_2 & on_goal]
            + goal_reward_scale * goal_reward[task_2 & on_goal]
        )

        reward[task_2 & ~on_goal] += (
            dist_g_penalty_scale * dis_g_penalty[task_2 & ~on_goal]
            + goal_reward_scale * goal_reward[task_2 & ~on_goal]
        )
        
        # reward = (
        #     dist_reward_scale * dis_reward
        #     # goal_reward
        #     + lift_reward_scale * lift_reward
        #     # + dist_g_reward_scale * dis_g_reward
        #     + self_collision_penalty_scale * self_collision_penalty
        #     # + contact_ground_penalty_scale * contact_ground_penalty
        #     + action_penalty_scale * actions_penalty
        #     # + contact_goal_penalty_scale * contact_goal_penalty
        # )

        self.task_id[success_lift_mask & task_0] = 1
        self.task_id[success_reach_mask] = 2
        self.task_id[failed_reach_mask & task_2] = 1
        self.task_id[failed_lift_mask] = 0

        self.reward += reward

        return reward
    
    def action_rate_l2_ratio(self) -> torch.Tensor:
        rate_of_change = (self.curr_actions - self.prev_actions) / 2.0
        self.prev_actions = self.curr_actions        
        return torch.mean(torch.abs(rate_of_change), dim=1)
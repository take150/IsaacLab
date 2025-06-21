# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torchvision.models as models
# from torch2trt import torch2trt
import numpy as np
import random
import cv2
import math
import os
from typing import Tuple

import omni.isaac.core.utils.prims as prims_utils
from omni.isaac.core.utils.stage import get_current_stage
from omni.isaac.core.utils.torch.transformations import tf_combine, tf_inverse, tf_vector
from pxr import UsdGeom

import omni.isaac.lab.sim as sim_utils
import omni.isaac.lab.envs.mdp as mdp
from omni.isaac.lab.managers import EventTermCfg as EventTerm
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.actuators.actuator_cfg import ImplicitActuatorCfg, IdealPDActuatorCfg, DelayedPDActuatorCfg
from omni.isaac.lab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from omni.isaac.lab.envs import DirectRLEnv, DirectRLEnvCfg
from omni.isaac.lab.scene import InteractiveSceneCfg
from omni.isaac.lab.sim import SimulationCfg
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import NUCLEUS_ASSET_ROOT_DIR
import omni.isaac.lab.utils.math as math_utils
import omni.isaac.lab.utils.noise as noise_utils
from omni.isaac.lab.utils.math import sample_uniform
from omni.isaac.lab.markers import VisualizationMarkersCfg, VisualizationMarkers
from omni.isaac.lab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.sensors import FrameTransformerCfg, FrameTransformer
from omni.isaac.lab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from omni.isaac.lab.sensors import TiledCamera, TiledCameraCfg
from omni.isaac.lab.sensors import ContactSensor, ContactSensorCfg
from omni.isaac.lab.utils.buffers import CircularBuffer

ASSET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../../"))

@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    randomize_leftcaster_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names="caster_back_left_link"),
          "static_friction_range": (0.005, 0.03),
          "dynamic_friction_range": (0.005, 0.02),
          "restitution_range": (0.5, 0.5),
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
          "restitution_range": (0.5, 0.5),
          "num_buckets": 1,
      },
    )

    randomize_leftwheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names="wheel_left_link"),
          "static_friction_range": (0.4, 0.6),
          "dynamic_friction_range": (0.3, 0.4),
          "restitution_range": (0.5, 0.5),
          "num_buckets": 1,
      },
    )
    
    randomize_rightwheel_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot", body_names="wheel_right_link"),
          "static_friction_range": (0.4, 0.6),
          "dynamic_friction_range": (0.3, 0.4),
          "restitution_range": (0.5, 0.5),
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

    randomize_terrain_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("ground_plane", body_names="ground"),
          "static_friction_range": (0.4, 0.6),
          "dynamic_friction_range": (0.25, 0.45),
          "restitution_range": (0.8, 0.8),
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
            # "colors": [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
            # "colors": [(1.0, 0.0, 0.0)]
        }
    )

    randomize_terrain_color = EventTerm(
        func=mdp.randomize_visual_color,
        mode="interval",
        interval_range_s=(2.5, 4.0),
        is_global_time=True,
        params={
            "event_name": "randomize_terrain_color",
            "asset_cfg": SceneEntityCfg("ground_plane", body_names="ground"),
            "colors":  {"r": (0.3, 0.6), "g": (0.3, 0.6), "b": (0.3, 0.6)},
            # "colors": [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
            # "colors": [(1.0, 0.0, 0.0)]
        }
    )

    randomize_dome_light = EventTerm(
        func=mdp.randomize_dome_light,
        mode="interval",
        interval_range_s=(2.5, 4.0),
        is_global_time=True,
        params={
            "event_name": "randomize_dome_light",
            "light_paths": ["/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/abandoned_garage_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/blinds_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/blocky_photo_studio_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/boma_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/blue_photo_studio_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/brown_photostudio_01_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/brown_photostudio_02_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/carpentry_shop_02_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/cinema_hall_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/climbing_gym_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/creepy_bathroom_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/empty_play_room_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/gym_01_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/hall_of_finfish_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/hall_of_mammals_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/hangar_interior_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/mirrored_hall_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/metro_noord_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/modern_bathroom_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/photo_studio_loft_hall_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/poly_haven_studio_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/rostock_laage_airport_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/sepulchral_chapel_basement_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/small_empty_room_1_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/small_empty_room_3_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/studio_small_09_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/warm_restaurant_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/warm_restaurant_night_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/whale_skeleton_4k.hdr",
                            "/home/takenami/IsaacLab/source/extensions/omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20/zwartkops_pit_4k.hdr",
                            ],
            "light_rotation": [(0.0, 0.0, 0.0), (0.0, 0.0, math.pi * 2)]
        }
    )


@configclass
class Turtlebot3ImageEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 50.01  # 160 timesteps
    decimation = 25
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
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=100.0, replicate_physics=True)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=100.0, replicate_physics=False)

    # robot
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            # usd_path=os.path.join(ASSET_ROOT, "omni.isaac.lab_assets/data/Robots/Turtlebot3_manipulation/turtlebot3_manipulation_merge_02.usd"),
            usd_path=os.path.join(ASSET_ROOT, "omni.isaac.lab_assets/data/Robots/Turtlebot3_manipulation/turtlebot3_manipulation_nolidar_collision_05.usd"),
            # usd_path=f"{ASSET_ROOT}/omni.isaac.lab_assets/data/Robots/Turtlebot3_manipulation/turtlebot3_manipulation.usd",
            activate_contact_sensors=True,
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
            # "turtlebot3_arm": ImplicitActuatorCfg(
            #     joint_names_expr=["joint[1-4]"],
            #     effort_limit=4.1,
            #     velocity_limit=0.2,
            #     stiffness={"joint1": 120, "joint2": 100, "joint3": 80, "joint4": 60},
            #     damping  ={"joint1": 6, "joint2": 5, "joint3": 4, "joint4": 3},
            # ),
            "turtlebot3_arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-4]"],
                effort_limit=4.1,
                velocity_limit=0.2,
                stiffness=40.0,
                damping=2.0,
            ),
            "turtlebot3_gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper_.*"],
                effort_limit=4.1,
                velocity_limit=0.02,
                stiffness=2000.0,
                damping=100.0,
            ),
            # "turtlebot3_wheel": ImplicitActuatorCfg(
            #     joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
            #     effort_limit=4.1,
            #     velocity_limit=1.8,
            #     stiffness=0.0,
            #     damping=6.0,
            # ),
            "turtlebot3_wheel": ImplicitActuatorCfg(
                joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
                effort_limit=4.1,
                velocity_limit=1.8,
                stiffness=100.0,
                damping=5.0,
            ),
            # "turtlebot3_arm": DelayedImplicitActuatorCfg(
            #     joint_names_expr=["joint[1-4]"],
            #     effort_limit=4.1,
            #     velocity_limit=2.0,
            #     stiffness={"joint1": 120, "joint2": 100, "joint3": 80, "joint4": 60},
            #     damping  ={"joint1": 6, "joint2": 5, "joint3": 4, "joint4": 3},
            #     min_delay=1,
            #     max_delay=10,
            # ),
            # "turtlebot3_gripper": DelayedImplicitActuatorCfg(
            #     joint_names_expr=["gripper_.*"],
            #     effort_limit=4.1,
            #     velocity_limit=0.2,
            #     stiffness=600.0,
            #     damping=30.0,
            #     min_delay=1,
            #     max_delay=10,
            # ),
            # "turtlebot3_wheel": DelayedImplicitActuatorCfg(
            #     joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
            #     effort_limit=4.1,
            #     velocity_limit=4.8,
            #     stiffness=0.0,
            #     damping=6.0,
            #     min_delay=1,
            #     max_delay=10,
            # ),
        },
    )

    # K = [69.6, 0, 42.0,   0, 92.6, 42.0,   0, 0, 1]
    # K = [42.27, 0, 26.15, 0, 56.15, 27.17, 0, 0, 1]
    # K = [132.767666, 0.000000, 60.718848, 0.000000, 132.922229, 61.248024, 0.00, 0.00, 1.0]

    # spawn_cfg = sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
    #                 intrinsic_matrix=K, width=128, height=128,
    #                 focal_length=0.304,            # 3.04 mm
    #                 clipping_range=(0.01, 20.0),
    #                 focus_distance=0.927,
    #             )

    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/base_footprint/front_cam",
        update_period=0.03,
        # offset=TiledCameraCfg.OffsetCfg(pos=(0.076, 0.067, 0.039), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        offset=TiledCameraCfg.OffsetCfg(pos=(0.076, 0.068, 0.041), rot=(0.99862953,  0.0,  0.05233596,  0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=0.304, focus_distance=0.927, horizontal_aperture=0.45, vertical_aperture=4.5, clipping_range=(0.01, 20.0)
        ),
        # spawn = spawn_cfg,
        width=120,
        height=120,
    )

    contact_link4_base: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/link4",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=["/World/envs/env_.*/Robot/base_link"],
    )

    contact_link5_base: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/link5",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=["/World/envs/env_.*/Robot/base_link"],
    )
    
    # cube: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/object",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=os.path.join(ASSET_ROOT, "omni.isaac.lab_assets/data/Objects/red_cube.usd"),
    #         scale=(0.02, 0.02, 0.02),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                 solver_position_iteration_count=16,
    #                 solver_velocity_iteration_count=1,
    #                 max_angular_velocity=1000.0,
    #                 max_linear_velocity=1000.0,
    #                 max_depenetration_velocity=5.0,
    #                 disable_gravity=False,
    #             ),
    #             mass_props=sim_utils.MassPropertiesCfg(mass=0.06),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    # )

    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(0.036, 0.036, 0.036),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.038, 0.038, 0.038),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.04, 0.04, 0.04),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.042, 0.042, 0.042),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.044, 0.044, 0.044),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
            ],
            random_choice=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=1000.0,
                        max_linear_velocity=1000.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,
                    ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.06),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    contact_leftgripper_object: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gripper_left_link",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=["/World/envs/env_.*/object"],
    )

    contact_rightgripper_object: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/gripper_right_link",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=["/World/envs/env_.*/object"],
    )
    
    # goal object
    # goal: VisualizationMarkersCfg = VisualizationMarkersCfg(
    #     prim_path="/Visuals/goal_marker",
    #     markers={
    #         "goal": sim_utils.SphereCfg(
    #             radius=0.025,
    #             visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.4, 0.3, 1.0)),
    #         ),
    #     },
    # )

    # ground plane
    # ground_plane: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/ground",
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=os.path.join(ASSET_ROOT, "omni.isaac.lab_assets/data/Objects/red_cube.usd"),
    #         scale=(4.0, 4.0, 0.005),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                 solver_position_iteration_count=16,
    #                 solver_velocity_iteration_count=1,
    #                 max_angular_velocity=1000.0,
    #                 max_linear_velocity=1000.0,
    #                 max_depenetration_velocity=5.0,
    #                 disable_gravity=False,
    #                 kinematic_enabled=True,
    #             ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.005), rot=(1.0, 0.0, 0.0, 0.0)),
    # )

    ground_plane: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ground",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(1.4, 1.4, 0.01),
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
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                    kinematic_enabled=True,
                ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.005), rot=(1.0, 0.0, 0.0, 0.0)),
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

    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.8,
    #     ),
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
    object_frame = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/object",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/object",
                name="object_up",
                offset=OffsetCfg(
                    # pos=[0.0, 0.0, 0.06],
                    pos=[0.0, 0.0, 0.0],
                ),
            ),
        ],
    )

    events: EventCfg = EventCfg()

    action_space = 2
    # observation_space = [camera.height, camera.width, 3]
    observation_space = {"rgb": [camera.height, camera.width, 3], "object": 7, "actions": 2}
    # observation_space = {"joint": 8, "rgb": [512, 7, 7]}
    # observation_space = {"joint": 8, "rgb": [512, 7, 7]}
    # observation_space = {"joint": 8, "rgb": 512}
    # observation_space = {"rgb": [camera.height, camera.width, 3]}

    # observation_noise_model: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
    #   noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.001, operation="add"),
    # )

    observation_noise_model = False

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

    action_noise_model = False

    action_noise_model_joint1: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"),
    )

    action_noise_model_joint2: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.03, std=0.0025, operation="add"),
    )

    action_noise_model_joint3: noise_utils.NoiseModelCfg = noise_utils.NoiseModelCfg(
      noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.02, std=0.0025, operation="add"),
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
    catch_reward_scale = 0.5
    lift_reward_scale = 1.0
    action_penalty_scale = -0.1
    dist_g_reward_scale = 1.0
    self_collision_penalty_scale = -0.3
    contact_ground_penalty_scale = -0.2


class Turtlebot3ImageEnv(DirectRLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: Turtlebot3ImageEnvCfg

    def __init__(self, cfg: Turtlebot3ImageEnvCfg, render_mode: str | None = None, **kwargs):
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

        # self.goal_rot = torch.zeros((self.num_envs, 4), dtype=torch.float, device=self.device)
        # self.goal_rot[:, 0] = 1.0
        # self.goal_pos_init = torch.zeros((self.num_envs, 3), dtype=torch.float, device=self.device)
        # self.goal_pos_init[:, :] = torch.tensor([0.0, 0.0, 0.08], device=self.device)

        self.x_unit_tensor = torch.tensor([1, 0, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.y_unit_tensor = torch.tensor([0, 1, 0], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))
        self.z_unit_tensor = torch.tensor([0, 0, 1], dtype=torch.float, device=self.device).repeat((self.num_envs, 1))

        self.joint_pos_names = ["joint1", "joint2", "joint3", "joint4", "gripper_left_joint.*", "gripper_right_joint"]
        self.arm_names = ["joint1", "joint2", "joint3", "joint4"]
        self.gripper_names = ["gripper_left_joint.*", "gripper_right_joint"]
        self.wheel_names = ["wheel_left_joint", "wheel_right_joint"]
        self.joint_pos_ids, _ = self._robot.find_joints(self.joint_pos_names, preserve_order=False)
        self.arm_ids, _ = self._robot.find_joints(self.arm_names, preserve_order=False)
        self.gripper_ids, _ = self._robot.find_joints(self.gripper_names, preserve_order=False)
        self.wheel_ids, _ = self._robot.find_joints(self.wheel_names, preserve_order=False)

        # 各関節の目標位置を初期化
        self.robot_arm_targets = torch.zeros((self.num_envs, len(self.arm_ids)), device=self.device)
        self.robot_gripper_targets = torch.zeros((self.num_envs, len(self.gripper_ids)), device=self.device)
        self.robot_wheel_targets = torch.zeros((self.num_envs, len(self.wheel_ids)), device=self.device)

        self.curr_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self.prev_actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        self.prev_cube_xy = torch.zeros((self.num_envs, 2), device=self.device)
        
        # self.img_features = torch.nn.Sequential(
        #     *list(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children())[:-2]
        # ).eval().to(self.device)

        # self.img_features = torch.nn.Sequential(
        #     *list(models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).children())[:-2]
        # ).eval().to(self.device)

        # dummy_input = torch.randn(1, 3, 224, 224, device=self.device)
        # self.img_features_trt = torch2trt(
        #     self.img_features, 
        #     [dummy_input],
        #     max_batch_size=512,
        # )

        # self.preprocess = models.ResNet18_Weights.DEFAULT.transforms()

        # self.preprocess = models.EfficientNet_B0_Weights.DEFAULT.transforms()

        # backgrounds_path = os.path.join(ASSET_ROOT, "omni.isaac.lab_assets/data/Backgrounds/4k_hdr_20")

        # file_names = [os.path.join(backgrounds_path, name) for name in os.listdir(backgrounds_path)]

        # self.dome_light_configs = [
        #     sim_utils.DomeLightCfg(
        #         intensity=2000.0,
        #         color=(0.75, 0.75, 0.75),
        #         texture_file=file_name  # 必要に応じてパスを調整する
        #     )
        #     for file_name in file_names
        # ]

        # self.episode = 0

    # シーン全体をセットアップ
    def _setup_scene(self):
        # ロボットを初期化
        self._robot = Articulation(self.cfg.robot)
        self._camera = TiledCamera(self.cfg.camera)
        self._contact_link4_base = ContactSensor(self.cfg.contact_link4_base)
        self._contact_link5_base = ContactSensor(self.cfg.contact_link5_base)
        self._contact_leftgripper_object = ContactSensor(self.cfg.contact_leftgripper_object)
        self._contact_rightgripper_object = ContactSensor(self.cfg.contact_rightgripper_object)
        self._contact_leftgripper_ground = ContactSensor(self.cfg.contact_leftgripper_ground)
        self._contact_rightgripper_ground = ContactSensor(self.cfg.contact_rightgripper_ground)
        self._ee_frame = FrameTransformer(self.cfg.ee_frame)
        self._lee_frame = FrameTransformer(self.cfg.lee_frame)
        self._ree_frame = FrameTransformer(self.cfg.ree_frame)
        self._cube = RigidObject(self.cfg.cube)
        self._object_frame = FrameTransformer(self.cfg.object_frame)
        self._ground_plane = RigidObject(self.cfg.ground_plane)
        # self.goal_markers = VisualizationMarkers(self.cfg.goal)
        # ロボットをシーンに追加
        self.scene.articulations["robot"] = self._robot
        self.scene.sensors["camera"] = self._camera
        self.scene.sensors["contact_link4_base"] = self._contact_link4_base
        self.scene.sensors["contact_link5_base"] = self._contact_link5_base
        self.scene.sensors["contact_leftgripper_object"] = self._contact_leftgripper_object
        self.scene.sensors["contact_rightgripper_object"] = self._contact_rightgripper_object
        self.scene.sensors["contact_leftgripper_ground"] = self._contact_leftgripper_ground
        self.scene.sensors["contact_rightgripper_ground"] = self._contact_rightgripper_ground
        self.scene.rigid_objects["cube"] = self._cube
        self.scene.sensors["ee_frame"] = self._ee_frame
        self.scene.sensors["lee_frame"] = self._lee_frame
        self.scene.sensors["ree_frame"] = self._ree_frame
        self.scene.sensors["object_frame"] = self._object_frame
        self.scene.rigid_objects["ground_plane"] = self._ground_plane
    
        # 地形の準備
        # self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        # self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        # self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # 元のシーンを複製
        self.scene.clone_environments(copy_from_source=False)
        # 特定のオブジェクト間での衝突を無効化
        # self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.scene.filter_collisions()

        # 証明を追加
        # light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        # light_cfg.func("/World/Light", light_cfg)
        
    # pre-physics step calls

    def _pre_physics_step(self, actions: torch.Tensor):
        actions = torch.nan_to_num(actions, nan=0.0)
        # arm_actions = actions[:, :len(self.arm_ids)].clone().clamp(-1.0, 1.0)
        # gripper_action = actions[:, len(self.arm_ids)].clone().clamp(-1.0, 1.0) 
        wheel_actions = actions.clone().clamp(-1.0, 1.0)

        # arm_targets = self._robot.data.joint_pos[:, self.arm_ids] + self.robot_dof_vel_limits_tensor[self.arm_ids] * self.dt * arm_actions
        # self.robot_arm_targets[:] = torch.clamp(arm_targets, self.robot_dof_lower_limits[self.arm_ids], self.robot_dof_upper_limits[self.arm_ids])

        # gripper_actions = torch.zeros(self.num_envs, len(self.gripper_ids), device=self.device)
        # gripper_actions[:, 0] = torch.where(gripper_action >= 0.0, self.robot_dof_upper_limits[self.gripper_ids[0]].item(),
        #                               self.robot_dof_lower_limits[self.gripper_ids[0]].item())
        # gripper_actions[:, 1] = torch.where(gripper_action >= 0.0, self.robot_dof_upper_limits[self.gripper_ids[1]].item(),
        #                               self.robot_dof_lower_limits[self.gripper_ids[1]].item())
        # self.robot_gripper_targets[:] = gripper_actions

        # wheel_actions[:, 0] = torch.full_like(wheel_actions[:, 0], 1.0)
        # wheel_actions[:, 1] = torch.full_like(wheel_actions[:, 1], 1.0)
        
        # self.robot_wheel_targets[:] = wheel_actions * self.robot_dof_vel_limits_tensor[self.wheel_ids]
        self.robot_wheel_targets[:] = self._robot.data.joint_pos[:, self.wheel_ids] + self.robot_dof_vel_limits_tensor[self.wheel_ids] * self.dt * wheel_actions

        # if self.episode_length_buf % 2 == 0:
        #     self.robot_arm_targets = torch.tensor([[0.0, -1.4, 1.4, 0.0]], device=self.device)
        # else:
        #     self.robot_arm_targets = torch.tensor([[0.0, -1.4, 1.35, 0.0]], device=self.device)
        # self.robot_gripper_targets = torch.tensor([[0.019, 0.019]], device=self.device)

        self.curr_actions = actions

    def _apply_action(self):
        # 制御
        # self._robot.set_joint_position_target(self.robot_arm_targets, self.arm_ids)
        # self._robot.set_joint_position_target(self.robot_gripper_targets, self.gripper_ids)
        # self._robot.set_joint_velocity_target(self.robot_wheel_targets, self.wheel_ids)
        self._robot.set_joint_position_target(self.robot_wheel_targets, self.wheel_ids)

    # post-physics step calls
    
    # 終了判定
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        terminated = torch.zeros(self.num_envs, dtype=torch.bool)
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        # if terminated.any():
        #     print("Terminated contains at least one 1 (True).", terminated.sum().item())

        # if truncated.any():
        #     print("Truncated contains at least one 1 (True).", truncated.sum().item())

        return terminated, truncated

    def _get_rewards(self) -> torch.Tensor:
        # Refresh the intermediate values after the physics steps
        self._compute_intermediate_values()

        return self._compute_rewards(
            self.base_pos,
            self.base_rot,
            self.cube_pos,
            self.object_pos,
            # self.goal_pos,
            self._robot.data.joint_pos,
            self.cfg.dist_reward_scale,
            self.cfg.catch_reward_scale,
            self.cfg.lift_reward_scale,
            self.cfg.dist_g_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.self_collision_penalty_scale,
            self.cfg.contact_ground_penalty_scale,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        # rand_floats = sample_uniform(-1.0, 1.0, (len(env_ids), 2), device=self.device)
        # new_rot = randomize_rotation(
        #     rand_floats[:, 0], rand_floats[:, 1], self.x_unit_tensor[env_ids], self.y_unit_tensor[env_ids]
        # )
        # self.goal_rot[env_ids] = new_rot
        # self.goal_pos = self.goal_pos_init + self.scene.env_origins
        # self.goal_pos[:, :3] += self.reset_root_state_uniform(
        #     env_ids=env_ids,
        #     pose_range = {"x": (0.3, 0.5), "y": (-0.1, 0.1), "z": (0.0, 0.0)},
        # )
        # self.goal_markers.visualize(self.goal_pos, self.goal_rot)

        # initialize robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids] + sample_uniform(
            -0.5,
            0.5,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )
        # joint_pos = torch.clamp(self._robot.data.default_joint_pos[env_ids], self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_pos = torch.clamp(joint_pos, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        # joint_pos[:, self.arm_ids] = torch.tensor([[0.0, -1.4, 1.4, 0.0]], device=self.device)
        joint_vel = torch.zeros_like(joint_pos)
        default_robot_state = self._robot.data.default_root_state[env_ids].clone()
        default_robot_state[:, :3] += self.scene.env_origins[env_ids]
        self._robot.write_root_link_pose_to_sim(default_robot_state[:, :7], env_ids=env_ids)
        self._robot.write_root_com_velocity_to_sim(default_robot_state[:, 7:], env_ids=env_ids)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        default_cube_state = self._cube.data.default_root_state[env_ids].clone()
        default_cube_state[:, :3] += self.scene.env_origins[env_ids]
        positions_delta, orientations_delta= self.reset_root_state_uniform(
            env_ids=env_ids,
            # pose_range = {"x": (0.265, 0.265), "y": (-0.0, 0.0), "z": (0.01, 0.01)},
            # pose_range = {"x": (0.4, 0.5), "y": (-0.15, 0.15), "z": (0.01, 0.01)},
            pose_range={"x": (0.25, 0.5), "y": (-0.1, 0.1), "z": (0.01, 0.01), "yaw": (0.0, math.pi/2)},
            avoid_radius=0.2
        )
        default_cube_state[:, :3] += positions_delta
        default_cube_state[:, 3:7] = math_utils.quat_mul(default_cube_state[:, 3:7], orientations_delta)
        self._cube.write_root_link_pose_to_sim(default_cube_state[:, :7], env_ids=env_ids)
        self.curr_actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.prev_cube_xy[env_ids] = default_cube_state[:, :2]
        # Need to refresh the intermediate values so that _get_observations() can use the latest values
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

        # ---- 円内チェック → 必要なら xy を引き直し ------------------------------
        if avoid_radius is not None and avoid_radius > 0.0:
            # pos_delta の view。あとでまとめて z / 回転を計算するので OK
            xy = rand_samples[:, 0:2]

            # 距離判定マスク
            in_circle = (xy[:, 0] ** 2 + xy[:, 1] ** 2) <= avoid_radius ** 2

            # 円に入ったぶんだけ resample（必要なくなるまで繰り返し）
            while in_circle.any():
                n_resample = int(in_circle.sum().item())
                # 0:x, 1:y だけ取り出して再サンプル
                xy_new = math_utils.sample_uniform(
                    ranges[0:2, 0], ranges[0:2, 1], (n_resample, 2), device=self.device
                )
                xy[in_circle] = xy_new
                # マスク更新
                in_circle = (xy[:, 0] ** 2 + xy[:, 1] ** 2) <= avoid_radius ** 2

            # xy を上書き済みなので rand_samples は更新済み

        # ---- tensor を分解して返却 ---------------------------------------------
        positions_delta = rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(
            rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
        )

        return positions_delta, orientations_delta

    def _get_observations(self) -> dict:

        rgb = self._camera.data.output["rgb"] / 255.0
        # with torch.no_grad():
        #     rgb = self.preprocess(self._camera.data.output["rgb"].permute(0, 3, 1, 2))
        #     rgb = self.img_features(rgb)

            # # move the image to the model device
            # image_proc = self._camera.data.output["rgb"].permute(0, 3, 1, 2).float() / 255.0
            # # normalize the image
            # mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
            # std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
            # image_proc = (image_proc - mean) / std
            # rgb_resnet = self.img_features_trt(image_proc)
            

        # # カメラデータからRGB画像を取り出し、CPU上に移動し、numpy配列に変換（uint8型）
        # image_np = self._camera.data.output["rgb"][0, ...].cpu().numpy().astype(np.uint8)  # RGB形式

        # # RGBからBGRに変換（OpenCVの表示用）
        # image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # # 画像を拡大。ここでは2倍に拡大する例。
        # scale_factor = 10.0
        # image_np = cv2.resize(image_np, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)

        # # ウィンドウを表示（WINDOW_NORMALでウィンドウサイズの変更を可能に）
        # cv2.namedWindow('Camera Feed', cv2.WINDOW_NORMAL)

        # # 画像をリアルタイムで表示
        # cv2.imshow('Camera Feed', image_np)

        # # 'q'キーが押されたらウィンドウを閉じる処理
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     cv2.destroyAllWindows()
        #     exit()

        joint_pos = self._robot.data.joint_pos[:, self.joint_pos_ids]
        joint_vel = self._robot.data.joint_vel[:, self.wheel_ids]
        # joint_pos = 2 * (self._robot.data.joint_pos[:, self.joint_pos_ids] - self.robot_dof_lower_limits[self.joint_pos_ids]) / (self.robot_dof_upper_limits[self.joint_pos_ids] - self.robot_dof_lower_limits[self.joint_pos_ids]) - 1.0
        # joint_vel = 2 * (self._robot.data.joint_vel[:, self.wheel_ids] + self.robot_dof_vel_limits_tensor[self.wheel_ids]) / (2*self.robot_dof_vel_limits_tensor[self.wheel_ids]) - 1.0
        # joint_pos = (self._robot.data.joint_pos[:, self.joint_pos_ids] - self.robot_dof_lower_limits[self.joint_pos_ids]) / (self.robot_dof_upper_limits[self.joint_pos_ids] - self.robot_dof_lower_limits[self.joint_pos_ids])
        # joint_vel = (self._robot.data.joint_vel[:, self.wheel_ids] + self.robot_dof_vel_limits_tensor[self.wheel_ids]) / (2*self.robot_dof_vel_limits_tensor[self.wheel_ids])
        # print(joint_pos)

        object_pos_b, object_rot_b = math_utils.subtract_frame_transforms(
            self.base_pos, self.base_rot, self.cube_pos, self.cube_rot
        )
        object_b = torch.cat((object_pos_b, object_rot_b), dim=1)

        obs = {
            # "joint": torch.cat(
            #     (
            #         joint_pos,
            #         # joint_vel,
            #     ),
            #     dim=-1,
            # ),
            "rgb": rgb,
            "object": object_b,
            "actions": self.curr_actions,
        }

        return obs
        # return {"rgb": rgb}
    
    # ロボットの把持位置、回転を最新状態に更新
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        self.base_pos = self._robot.data.root_link_state_w[env_ids, :3]
        self.base_rot = self._robot.data.root_link_state_w[env_ids, 3:7]
        self.cube_pos = self._cube.data.root_link_pos_w[env_ids, :]
        self.cube_rot = self._cube.data.root_link_state_w[env_ids, 3:7]
        self.ee_pos = self._ee_frame.data.target_pos_w[env_ids, 0, :]
        self.lee_pos = self._lee_frame.data.target_pos_w[env_ids, 0, :]
        self.ree_pos = self._ree_frame.data.target_pos_w[env_ids, 0, :]
        self.object_pos = self._object_frame.data.target_pos_w[env_ids, 0, :]
        self.contact_link4_base = self._contact_link4_base.data.force_matrix_w[env_ids, :]
        self.contact_link5_base = self._contact_link5_base.data.force_matrix_w[env_ids, :]
        self.contact_leftgripper_object = self._contact_leftgripper_object.data.force_matrix_w[env_ids, :]
        self.contact_rightgripper_object = self._contact_rightgripper_object.data.force_matrix_w[env_ids, :]
        self.contact_leftgripper_ground = self._contact_leftgripper_ground.data.force_matrix_w[env_ids, :]
        self.contact_rightgripper_ground = self._contact_rightgripper_ground.data.force_matrix_w[env_ids, :]

    def _compute_rewards(
        self,
        base_pos,
        base_rot,
        cube_pos,
        object_pos,
        # goal_pos,
        joint_positions,
        dist_reward_scale,
        catch_reward_scale,
        lift_reward_scale,
        dist_g_reward_scale,
        action_penalty_scale,
        self_collision_penalty_scale,
        contact_ground_penalty_scale
    ):
        
        # print(1111111111111111)
        # print(self.actions_buf.buffer.shape)
        # print(self.actions_buf.buffer[0])
        # print(self.actions_buf.buffer[-1])
        delta_xy = object_pos[:, :2] - base_pos[:, :2]
        d = torch.norm(delta_xy, dim=-1)
        d_reward = torch.exp(-10 * torch.abs(d - 0.2))
        
        yaw = 2.0 * torch.atan2(base_rot[:, 3], base_rot[:, 0])  # (N,)
        target_yaw = torch.atan2(delta_xy[:, 1], delta_xy[:, 0])
        yaw_err = torch.abs(math_utils.wrap_to_pi(target_yaw - yaw))
        h_reward = torch.exp(-yaw_err)

        dis_reward = d_reward * h_reward

        # d_c = torch.norm(object_pos-end_effector_pos, dim=-1)
        # d_l = torch.norm(object_pos-left_tip_pos, dim=-1)
        # d_r = torch.norm(object_pos-right_tip_pos, dim=-1)
        # d = (d_c*2 + d_l + d_r) / 4
        # # dis_reward = torch.exp(-10*d_c)
        # dis_reward = torch.where(d_c >= 0.02,
        #                  torch.exp(-10*d_c),   # 条件を満たす場合
        #                  1.25 * torch.exp(-10*d))     # それ以外の場合
        # dis_reward = 1 - torch.tanh(10*d)
        # dis_reward = torch.exp(-10*d)
        # dis_reward = torch.where(inside_radius, dis_reward, torch.zeros_like(dis_reward))
        # print(d_c)

        # catch_object = torch.norm(contact_leftgripper_object, dim=-1).squeeze() * torch.norm(contact_rightgripper_object, dim=-1).squeeze() > 1.0
        # lift_reward = torch.where(cube_pos[:, 2] > 0.03, 1.0, 0.0) * catch_object
        # lift_reward = torch.where(cube_pos[:, 2] > 0.10, 1.0, 0.0) * torch.where(dis < 0.03, 1.0, 0.0)
        # lift_reward = (cube_pos[:, 2] - 0.025/2) > 0.04

        # actions_penalty = self.action_rate_l2_ratio()
        # dis_g = torch.norm(goal_pos-cube_pos, dim=-1)
        # dis_g_reward = torch.exp(-10*dis_g)

        # contact_4_penalty = torch.norm(contact_link4_base, dim=-1).squeeze() > 1.0
        # contact_5_penalty = torch.norm(contact_link5_base, dim=-1).squeeze() > 1.0
        # self_collision_penalty = contact_4_penalty | contact_5_penalty

        # contact_left_ground_penalty = torch.norm(contact_leftgripper_ground, dim=-1).squeeze() > 1.0
        # contact_right_ground_penalty = torch.norm(contact_rightgripper_ground, dim=-1).squeeze() > 1.0
        # contact_ground_penalty = contact_left_ground_penalty | contact_right_ground_penalty

        rewards = (
            dist_reward_scale * dis_reward
            # + catch_reward_scale * catch_object
            # + lift_reward_scale * lift_reward
            # + self_collision_penalty_scale * self_collision_penalty
            # + contact_ground_penalty_scale * contact_ground_penalty
            # + action_penalty_scale * actions_penalty
            # + dist_g_reward_scale * dis_g_reward
        )

        # print(111111111111111111111111111111111)
        # print(dis_reward)
        # print(dis)
        # print(d_c)
        # print(d_l)
        # print(d_r)
        # print(lift_reward)
        # print(dis_g_reward)
        # print(rewards)

        # self.extras["log"] = {
        #     "action_penalty": (-action_penalty_scale * action_penalty).mean(),
        # }

        return rewards
    
    def action_rate_l2_ratio(self) -> torch.Tensor:
        rate_of_change = (self.curr_actions - self.prev_actions) / 2.0
        self.prev_actions = self.curr_actions        
        return torch.mean(torch.abs(rate_of_change), dim=1)

@torch.jit.script
def randomize_rotation(rand0, rand1, x_unit_tensor, y_unit_tensor):
    return quat_mul(
        quat_from_angle_axis(rand0 * np.pi, x_unit_tensor), quat_from_angle_axis(rand1 * np.pi, y_unit_tensor)
    )

@torch.jit.script
def compute_dis_to_(obj1, obj2):
    # type: (torch.Tensor, torch.Tensor) -> torch.Tensor

    dis = torch.norm(obj2-obj1, dim=-1).unsqueeze(-1)
    return dis

@torch.jit.script
def compute_angle_to_(obj1_rot, obj1_pos, obj2_pos):
    # type: (torch.Tensor, torch.Tensor, torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]

    obj1_to_obj2_pos = obj2_pos - obj1_pos
    obj1_to_obj2_angle = torch.atan2(obj1_to_obj2_pos[:, 1], obj1_to_obj2_pos[:, 0]).unsqueeze(-1)
    obj1_angle = 2 * torch.atan2(obj1_rot[:, 3], obj1_rot[:, 0]).unsqueeze(-1)

    sin_angle = torch.sin(obj1_to_obj2_angle - obj1_angle)
    cos_angle = torch.cos(obj1_to_obj2_angle - obj1_angle)

    return sin_angle, cos_angle
    

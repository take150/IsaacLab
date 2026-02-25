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
import tqdm
import csv

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
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
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
from isaaclab.utils.buffers import CircularBuffer

ASSET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))

@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # randomize_joints_gain_1_1 = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot_1", joint_names=["joint1"]),
    #         "stiffness_distribution_params": (180.0, 220.0),
    #         "damping_distribution_params": (18.0, 22.0),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )

    # randomize_joints_gain_2_1 = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot_1", joint_names=["joint2"]),
    #         "stiffness_distribution_params": (270.0, 330.0),
    #         "damping_distribution_params": (27.0, 33.0),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )

    # randomize_joints_gain_3_1 = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot_1", joint_names=["joint3"]),
    #         "stiffness_distribution_params": (270.0, 330.0),
    #         "damping_distribution_params": (27.0, 33.0),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )

    # randomize_joints_gain_4_1 = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot_1", joint_names=["joint4"]),
    #         "stiffness_distribution_params": (360.0, 440.0),
    #         "damping_distribution_params": (36.0, 44.0),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )

    # randomize_joints_gain_1_2= EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot_2", joint_names=["joint1"]),
    #         "stiffness_distribution_params": (180.0, 220.0),
    #         "damping_distribution_params": (18.0, 22.0),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )

    # randomize_joints_gain_2_2 = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot_2", joint_names=["joint2"]),
    #         "stiffness_distribution_params": (270.0, 330.0),
    #         "damping_distribution_params": (27.0, 33.0),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )

    # randomize_joints_gain_3_2 = EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot_2", joint_names=["joint3"]),
    #         "stiffness_distribution_params": (270.0, 330.0),
    #         "damping_distribution_params": (27.0, 33.0),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )

    # randomize_joints_gain_4_2= EventTerm(
    #     func=mdp.randomize_actuator_gains,
    #     mode="reset",
    #     params={
    #         "asset_cfg": SceneEntityCfg("robot_2", joint_names=["joint4"]),
    #         "stiffness_distribution_params": (360.0, 440.0),
    #         "damping_distribution_params": (36.0, 44.0),
    #         "operation": "abs",
    #         "distribution": "uniform",
    #     },
    # )

    randomize_gripper_velocity_limit_1 = EventTerm(
        func=mdp.randomize_actuator_velocity_limit,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_1", joint_names=["gripper_left_joint", "gripper_right_joint"]),
            "distribution_params": (0.0075, 0.01),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    randomize_gripper_velocity_limit_2 = EventTerm(
        func=mdp.randomize_actuator_velocity_limit,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_2", joint_names=["gripper_left_joint", "gripper_right_joint"]),
            "distribution_params": (0.0075, 0.01),
            "operation": "abs",
            "distribution": "uniform",
        },
    )

    reset_leftcaster_friction_1 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot_1", body_names="caster_back_left_link"),
          "static_friction_range": (0.005, 0.03),
          "dynamic_friction_range": (0.005, 0.02),
          "restitution_range": (0.0, 0.0),
          "num_buckets": 1,
      },
    )

    reset_leftcaster_friction_2 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot_2", body_names="caster_back_left_link"),
          "static_friction_range": (0.005, 0.03),
          "dynamic_friction_range": (0.005, 0.02),
          "restitution_range": (0.0, 0.0),
          "num_buckets": 1,
      },
    )

    reset_rightcaster_friction_1 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot_1", body_names="caster_back_right_link"),
          "static_friction_range": (0.005, 0.03),
          "dynamic_friction_range": (0.005, 0.02),
          "restitution_range": (0.0, 0.0),
          "num_buckets": 1,
      },
    )

    reset_rightcaster_friction_2 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot_2", body_names="caster_back_right_link"),
          "static_friction_range": (0.005, 0.03),
          "dynamic_friction_range": (0.005, 0.02),
          "restitution_range": (0.0, 0.0),
          "num_buckets": 1,
      },
    )

    reset_leftwheel_friction_1 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot_1", body_names="wheel_left_link"),
          "static_friction_range": (0.7, 0.9),
          "dynamic_friction_range": (0.6, 0.7),
          "restitution_range": (0.0, 0.0),
          "num_buckets": 1,
      },
    )
    reset_leftwheel_friction_2 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot_2", body_names="wheel_left_link"),
          "static_friction_range": (0.7, 0.9),
          "dynamic_friction_range": (0.6, 0.7),
          "restitution_range": (0.0, 0.0),
          "num_buckets": 1,
      },
    )
    
    reset_rightwheel_friction_1 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot_1", body_names="wheel_right_link"),
          "static_friction_range": (0.7, 0.9),
          "dynamic_friction_range": (0.6, 0.7),
          "restitution_range": (0.0, 0.0),
          "num_buckets": 1,
      },
    )

    reset_rightwheel_friction_2 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot_2", body_names="wheel_right_link"),
          "static_friction_range": (0.7, 0.9),
          "dynamic_friction_range": (0.6, 0.7),
          "restitution_range": (0.0, 0.0),
          "num_buckets": 1,
      },
    )

    reset_leftfinger_friction_1 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot_1", body_names="gripper_left_link"),
          "static_friction_range": (1.1, 1.3),
          "dynamic_friction_range": (0.85, 1.05),
          "restitution_range": (0.5, 0.5),
          "num_buckets": 1,
      },
    )
    
    reset_leftfinger_friction_2 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot_2", body_names="gripper_left_link"),
          "static_friction_range": (1.1, 1.3),
          "dynamic_friction_range": (0.85, 1.05),
          "restitution_range": (0.5, 0.5),
          "num_buckets": 1,
      },
    )

    reset_rightfinger_friction_1 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot_1", body_names="gripper_right_link"),
          "static_friction_range": (1.1, 1.3),
          "dynamic_friction_range": (0.85, 1.05),
          "restitution_range": (0.5, 0.5),
          "num_buckets": 1,
      },
    )

    reset_rightfinger_friction_2 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot_2", body_names="gripper_right_link"),
          "static_friction_range": (1.1, 1.3),
          "dynamic_friction_range": (0.85, 1.05),
          "restitution_range": (0.5, 0.5),
          "num_buckets": 1,
      },
    )

    randomize_object_friction_1 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("cube_1", body_names="object_1"),
          "static_friction_range": (0.4, 0.6),
          "dynamic_friction_range": (0.3, 0.5),
          "restitution_range": (0.5, 0.5),
          "num_buckets": 1,
      },
    )

    randomize_object_friction_2 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("cube_2", body_names="object_2"),
          "static_friction_range": (0.4, 0.6),
          "dynamic_friction_range": (0.3, 0.5),
          "restitution_range": (0.5, 0.5),
          "num_buckets": 1,
      },
    )

    # randomize_object_color_1 = EventTerm(
    #     func=mdp.randomize_visual_color,
    #     mode="interval",
    #     interval_range_s=(2.5, 4.0),
    #     is_global_time=True,
    #     params={
    #         "event_name": "randomize_object_color_1",
    #         "asset_cfg": SceneEntityCfg("cube_1", body_names="object_1"),
    #         "colors":  {"r": (0.3, 0.4), "g": (0.0, 0.1), "b": (0.0, 0.1)},
    #     }
    # )

    # randomize_object_color_2 = EventTerm(
    #     func=mdp.randomize_visual_color,
    #     mode="interval",
    #     interval_range_s=(2.5, 4.0),
    #     is_global_time=True,
    #     params={
    #         "event_name": "randomize_object_color_2",
    #         "asset_cfg": SceneEntityCfg("cube_2", body_names="object_2"),
    #         "colors":  {"r": (0.3, 0.4), "g": (0.0, 0.1), "b": (0.0, 0.1)},
    #         # "colors":  {"r": (0.0, 0.1), "g": (0.3, 0.4), "b": (0.0, 0.1)},
    #     }
    # )
    
    # randomize_goal_base_color = EventTerm(
    #     func=mdp.randomize_visual_color,
    #     mode="interval",
    #     interval_range_s=(2.5, 4.0),
    #     is_global_time=True,
    #     params={
    #         "event_name": "randomize_goal_base_color",
    #         "asset_cfg": SceneEntityCfg("goal_base", body_names="goal_base"),
    #         "colors":  {"r": (0.0, 0.1), "g": (0.0, 0.1), "b": (0.3, 0.4)},
    #         }
    # )

    # randomize_goal_color = EventTerm(
    #     func=mdp.randomize_visual_color,
    #     mode="interval",
    #     interval_range_s=(2.5, 4.0),
    #     is_global_time=True,
    #     params={
    #         "event_name": "randomize_goal_color",
    #         "asset_cfg": SceneEntityCfg("goal", body_names="goal"),
    #         "colors":  {"r": (0.0, 0.1), "g": (0.0, 0.1), "b": (0.3, 0.4)},
    #     }
    # )

    # randomize_terrain_color = EventTerm(
    #     func=mdp.randomize_visual_color,
    #     mode="interval",
    #     interval_range_s=(2.5, 4.0),
    #     is_global_time=True,
    #     params={
    #         "event_name": "randomize_terrain_color",
    #         "asset_cfg": SceneEntityCfg("ground_plane", body_names="ground_color"),
    #         "colors":  {"r": (0.3, 0.6), "g": (0.3, 0.6), "b": (0.3, 0.6)},
    #     }
    # )

    # randomize_dome_light = EventTerm(
    #     func=mdp.randomize_dome_light,
    #     mode="interval",
    #     interval_range_s=(2.5, 4.0),
    #     is_global_time=True,
    #     params={
    #         "event_name": "randomize_dome_light",
    #         "light_paths": ["/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/abandoned_garage_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/blinds_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/blocky_photo_studio_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/boma_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/blue_photo_studio_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/brown_photostudio_01_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/brown_photostudio_02_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/carpentry_shop_02_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/cinema_hall_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/climbing_gym_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/creepy_bathroom_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/empty_play_room_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/gym_01_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/hall_of_finfish_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/hall_of_mammals_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/hangar_interior_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/mirrored_hall_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/metro_noord_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/modern_bathroom_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/photo_studio_loft_hall_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/poly_haven_studio_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/rostock_laage_airport_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/sepulchral_chapel_basement_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/small_empty_room_1_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/small_empty_room_3_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/studio_small_09_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/warm_restaurant_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/warm_restaurant_night_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/whale_skeleton_4k.hdr",
    #                         "/home/takenami/IsaacLab/source/isaaclab_assets/data/Backgrounds/4k_hdr_20/zwartkops_pit_4k.hdr",
    #                         ],
    #         "light_rotation": [(0.0, 0.0, 0.0), (0.0, 0.0, math.pi * 2)]
    #     }
    # )


@configclass
class Turtlebot3MultiPlaceEnvCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 30.1  # 1500 timesteps
    decimation = 25
    seed = 0

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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=2048, env_spacing=3, replicate_physics=False)

    # robot
    robot_1 = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ASSET_ROOT, "isaaclab_assets/data/Robots/Turtlebot3_manipulation/turtlebot3_manipulation_nolidar_collision_test__.usd"),
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
            # rot=(0.70710678, 0.0, 0.0, 0.70710678),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        actuators={
            "turtlebot3_arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-4]"],
                effort_limit_sim=4.1,
                velocity_limit_sim=0.2,
                # stiffness={"joint1": 100, "joint2": 60, "joint3": 60, "joint4": 80},
                # damping={"joint1": 10, "joint2": 6, "joint3": 6, "joint4": 8},
                stiffness=200,
                damping=10,
                # friction=0.2,
                # armature=0.0075
            ),
            "turtlebot3_gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper_.*"],
                effort_limit_sim=4.1,
                velocity_limit_sim=0.0075,
                stiffness=2000.0,
                damping=100.0,
            ),
            "turtlebot3_wheel": ImplicitActuatorCfg(
                joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
                effort_limit_sim=4.1,
                velocity_limit_sim=0.8,
                stiffness=0.0,
                damping=10.0,
                friction=0.2,
                armature=0.0075
            ),
        },
    )

    # camera_1: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="/World/envs/env_.*/Robot_1/base_footprint/front_cam",
    #     update_period=0.03,
    #     offset=TiledCameraCfg.OffsetCfg(pos=(0.076, 0.068, 0.041), rot=(0.99756405, 0.0, 0.06975647, 0.0), convention="world"),
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=0.304, focus_distance=0.927, horizontal_aperture=0.45, vertical_aperture=4.5, clipping_range=(0.01, 20.0)
    #     ),
    #     width=120,
    #     height=120,
    # )

    contact_base_1: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_1/base_link",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot_1/link4",
            "/World/envs/env_.*/Robot_1/link5",
            "/World/envs/env_.*/Robot_1/gripper_left_link",
            "/World/envs/env_.*/Robot_1/gripper_right_link",
            ],
    )

    robot_2 = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot_2",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ASSET_ROOT, "isaaclab_assets/data/Robots/Turtlebot3_manipulation/turtlebot3_manipulation_nolidar_collision_test__.usd"),
            # usd_path=f"{ASSET_ROOT}/isaaclab_assets/data/Robots/Turtlebot3_manipulation/turtlebot3_manipulation.usd",
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
            # rot=(0.70710678, 0.0, 0.0, -0.70710678),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        actuators={
            "turtlebot3_arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-4]"],
                effort_limit_sim=4.1,
                velocity_limit_sim=0.2,
                # stiffness={"joint1": 100, "joint2": 60, "joint3": 60, "joint4": 80},
                # damping={"joint1": 10, "joint2": 6, "joint3": 6, "joint4": 8},
                stiffness=200,
                damping=10,
                # friction=0.2,
                # armature=0.0075
            ),
            "turtlebot3_gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper_.*"],
                effort_limit_sim=4.1,
                velocity_limit_sim=0.0075,
                stiffness=2000.0,
                damping=100.0,
            ),
            "turtlebot3_wheel": ImplicitActuatorCfg(
                joint_names_expr=["wheel_left_joint", "wheel_right_joint"],
                effort_limit_sim=4.1,
                velocity_limit_sim=0.8,
                stiffness=0.0,
                damping=10.0,
                friction=0.2,
                armature=0.0075
            ),
        },
    )

    # camera_2: TiledCameraCfg = TiledCameraCfg(
    #     prim_path="/World/envs/env_.*/Robot_2/base_footprint/front_cam",
    #     update_period=0.03,
    #     offset=TiledCameraCfg.OffsetCfg(pos=(0.076, 0.068, 0.041), rot=(0.99756405, 0.0, 0.06975647, 0.0), convention="world"),
    #     data_types=["rgb"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=0.304, focus_distance=0.927, horizontal_aperture=0.45, vertical_aperture=4.5, clipping_range=(0.01, 20.0)
    #     ),
    #     width=120,
    #     height=120,
    # )

    contact_base_2: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_2/base_link",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot_2/link4",
            "/World/envs/env_.*/Robot_2/link5",
            "/World/envs/env_.*/Robot_2/gripper_left_link",
            "/World/envs/env_.*/Robot_2/gripper_right_link",
            ],
    )
    
    cube_1: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object_1",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                # sim_utils.CuboidCfg(
                #     size=(0.036, 0.036, 0.036),
                #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                # ),
                sim_utils.CuboidCfg(
                    size=(0.041, 0.041, 0.041),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.042, 0.042, 0.042),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.043, 0.043, 0.043),
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

    contact_gripper_object_1: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/object_1",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot_1/gripper_left_tip_link",
            "/World/envs/env_.*/Robot_1/gripper_right_tip_link",
            ],
    )

    cube_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object_2",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                # sim_utils.CuboidCfg(
                #     size=(0.036, 0.036, 0.036),
                #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                # ),
                sim_utils.CuboidCfg(
                    size=(0.043, 0.043, 0.043),
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
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),
    )

    contact_gripper_object_2: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/object_2",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot_2/gripper_left_tip_link",
            "/World/envs/env_.*/Robot_2/gripper_right_tip_link",
            ],
    )

    # goal_base: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/goal_base",
    #     spawn=sim_utils.MultiAssetSpawnerCfg(
    #         assets_cfg=[
    #             sim_utils.CuboidCfg(
    #                 # size=(0.15, 0.15, 0.015),
    #                 size=(0.5, 0.35, 0.015),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
    #             ),
    #             # sim_utils.CuboidCfg(
    #             #     size=(0.08, 0.07, 0.03),
    #             #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #             # ),
    #             # sim_utils.CuboidCfg(
    #             #     size=(0.08, 0.06, 0.03),
    #             #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #             # ),
    #             # sim_utils.CuboidCfg(
    #             #     size=(0.08, 0.05, 0.03),
    #             #     visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #             # ),
    #         ],
    #         random_choice=False,
    #         activate_contact_sensors=True,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                     kinematic_enabled=True,
    #                     solver_position_iteration_count=16,
    #                     solver_velocity_iteration_count=1,
    #                     max_angular_velocity=1000.0,
    #                     max_linear_velocity=1000.0,
    #                     max_depenetration_velocity=5.0,
    #                     disable_gravity=False,
    #                 ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
    #         # mass_props=sim_utils.MassPropertiesCfg(mass=0.8),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    # )

    goal: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    # size=(0.1, 0.3, 0.015),
                    size=(0.3, 0.1, 0.036),
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
            mass_props=sim_utils.MassPropertiesCfg(mass=10.0),
            # mass_props=sim_utils.MassPropertiesCfg(mass=0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        # init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.045), rot=(1.0, 0.0, 0.0, 0.0)),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.018), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    contact_robot_goal_1: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/goal",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot_1/base_link",
            "/World/envs/env_.*/Robot_1/gripper_left_link",
            "/World/envs/env_.*/Robot_1/gripper_right_link",
            "/World/envs/env_.*/Robot_1/link1",
            "/World/envs/env_.*/Robot_1/link2",
            "/World/envs/env_.*/Robot_1/link3",
            "/World/envs/env_.*/Robot_1/link4",
            "/World/envs/env_.*/Robot_1/link5"],
    )

    contact_robot_goal_2: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/goal",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot_2/base_link",
            "/World/envs/env_.*/Robot_2/gripper_left_link",
            "/World/envs/env_.*/Robot_2/gripper_right_link",
            "/World/envs/env_.*/Robot_2/link1",
            "/World/envs/env_.*/Robot_2/link2",
            "/World/envs/env_.*/Robot_2/link3",
            "/World/envs/env_.*/Robot_2/link4",
            "/World/envs/env_.*/Robot_2/link5"],
    )

    contact_object_goal_1: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/object_1",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/goal"],
    )

    contact_object_goal_2: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/object_2",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/goal"],
    )

    # # ground plane
    # ground_plane: RigidObjectCfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/ground_color",
    #     spawn=sim_utils.AssetSpawnerCfg(
    #         assets_cfg=[
    #             sim_utils.CuboidCfg(
    #                 size=(3.5, 3.5, 0.01),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #             ),
    #             sim_utils.CuboidCfg(
    #                 size=(4.0, 4.0, 0.01),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #             ),
    #             sim_utils.CuboidCfg(
    #                 size=(6.0, 6.0, 0.01),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #             ),
    #             sim_utils.CuboidCfg(
    #                 size=(8.0, 8.0, 0.01),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #             ),
    #             sim_utils.CuboidCfg(
    #                 size=(10.0, 10.0, 0.01),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #             ),
    #             sim_utils.CuboidCfg(
    #                 size=(15.0, 15.0, 0.01),
    #                 visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
    #             ),
    #         ],
    #         # random_choice=True,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #                 disable_gravity=False,
    #                 kinematic_enabled=True,
    #             ),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(
    #             collision_enabled=False
    #         ),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -0.005), rot=(1.0, 0.0, 0.0, 0.0)),
    # )

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

    marker_cfg = FRAME_MARKER_CFG.copy()
    marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    marker_cfg.prim_path = "/Visuals/FrameTransformer"

    ee_frame_1 = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot_1/base_footprint",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot_1/link5",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.126, 0.0, 0.0],
                ),
            ),
        ],
    )
    lee_frame_1 = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot_1/base_footprint",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot_1/gripper_left_link",
                name="left_end_effector",
                offset=OffsetCfg(
                    pos=[0.045, -0.002, 0.0],
                ),
            ),
        ],
    )
    ree_frame_1 = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot_1/base_footprint",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot_1/gripper_right_link",
                name="right_end_effector",
                offset=OffsetCfg(
                    pos=[0.045, 0.002, 0.0],
                ),
            ),
        ],
    )

    ee_frame_2 = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot_2/base_footprint",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot_2/link5",
                name="end_effector",
                offset=OffsetCfg(
                    pos=[0.126, 0.0, 0.0],
                ),
            ),
        ],
    )
    lee_frame_2 = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot_2/base_footprint",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot_2/gripper_left_link",
                name="left_end_effector",
                offset=OffsetCfg(
                    pos=[0.045, -0.002, 0.0],
                ),
            ),
        ],
    )
    ree_frame_2 = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot_2/base_footprint",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot_2/gripper_right_link",
                name="right_end_effector",
                offset=OffsetCfg(
                    pos=[0.045, 0.002, 0.0],
                ),
            ),
        ],
    )

    goal_frame_1 = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/goal",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/goal",
                name="cube_1_set",
                offset=OffsetCfg(
                    pos=[-0.12, 0.0, 0.0],
                ),
            ),
        ],
    )

    goal_frame_2 = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/goal",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/goal",
                name="cube_2_set",
                offset=OffsetCfg(
                    pos=[0.12, 0.0, 0.0],
                    rot=[0.0, 0.0, 0.0, 1.0]
                ),
            ),
        ],
    )

    goal_above_frame_1 = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/goal",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/goal",
                name="cube_1_set",
                offset=OffsetCfg(
                    pos=[-0.12, 0.0, 0.048],
                ),
            ),
        ],
    )

    goal_above_frame_2 = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/goal",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/goal",
                name="cube_2_set",
                offset=OffsetCfg(
                    pos=[0.12, 0.0, 0.048],
                    rot=[0.0, 0.0, 0.0, 1.0]
                ),
            ),
        ],
    )

    camera_frame_1 = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot_1/base_footprint",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot_1/base_footprint",
                name="camera_up_1",
                offset=OffsetCfg(
                    pos=(0.076, 0.068, 0.041),
                    rot=(0.99756405, 0.0, 0.06975647, 0.0)
                ),
            ),
        ],
    )

    camera_frame_2 = FrameTransformerCfg(
        prim_path="/World/envs/env_.*/Robot_2/base_footprint",
        debug_vis=False,
        visualizer_cfg=marker_cfg,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="/World/envs/env_.*/Robot_2/base_footprint",
                name="camera_up_2",
                offset=OffsetCfg(
                    pos=(0.076, 0.068, 0.041),
                    rot=(0.99756405, 0.0, 0.06975647, 0.0)
                ),
            ),
        ],
    )

    events: EventCfg = EventCfg()

    possible_agents = ["robot_1", "robot_2"]
    action_spaces = {"robot_1": 7, "robot_2": 7}
    observation_spaces = {
        "robot_1": {"task_id": 1, "joint": [5, 6], "object": [5, 14], "goal": [5, 7], "actions": [5, 7], "joint_other": [5, 6], "object_other": [5, 14], "goal_other": [5, 7], "actions_other": [5, 7]},
        "robot_2": {"task_id": 1, "joint": [5, 6], "object": [5, 14], "goal": [5, 7], "actions": [5, 7], "joint_other": [5, 6], "object_other": [5, 14], "goal_other": [5, 7], "actions_other": [5, 7]},
        }
    
    state_space = 54

    # observation noise
    # observation noise
    observation_noise_model = True
    observation_noise_model_joint1: dict[AgentID, noise_utils.NoiseModelCfg] = {
        agent: noise_utils.NoiseModelCfg(noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0025, operation="add"))
        for agent in possible_agents
    }
    observation_noise_model_joint2: dict[AgentID, noise_utils.NoiseModelCfg] = {
        agent: noise_utils.NoiseModelCfg(noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0025, operation="add"))
        for agent in possible_agents
    }
    observation_noise_model_joint3: dict[AgentID, noise_utils.NoiseModelCfg] = {
        agent: noise_utils.NoiseModelCfg(noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0025, operation="add"))
        for agent in possible_agents
    }
    observation_noise_model_joint4: dict[AgentID, noise_utils.NoiseModelCfg] = {
        agent: noise_utils.NoiseModelCfg(noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0025, operation="add"))
        for agent in possible_agents
    }
    observation_noise_model_gripper: dict[AgentID, noise_utils.NoiseModelCfg] = {
        agent: noise_utils.NoiseModelCfg(noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0025, operation="add"))
        for agent in possible_agents
    }
    observation_noise_model_rgb: dict[AgentID, noise_utils.NoiseModelCfg] = {
        agent: noise_utils.NoiseModelCfg(noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.0025, operation="add"))
        for agent in possible_agents
    }    

    # action noise
    action_noise_model = True
    action_noise_model_joint1: dict[AgentID, noise_utils.NoiseModelCfg] = {
        agent: noise_utils.NoiseModelCfg(noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"))
        for agent in possible_agents
    }
    action_noise_model_joint2: dict[AgentID, noise_utils.NoiseModelCfg] = {
        agent: noise_utils.NoiseModelCfg(noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"))
        for agent in possible_agents
    }
    action_noise_model_joint3: dict[AgentID, noise_utils.NoiseModelCfg] = {
        agent: noise_utils.NoiseModelCfg(noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"))
        for agent in possible_agents
    }
    action_noise_model_joint4: dict[AgentID, noise_utils.NoiseModelCfg] = {
        agent: noise_utils.NoiseModelCfg(noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.005, operation="add"))
        for agent in possible_agents
    }
    action_noise_model_wheel: dict[AgentID, noise_utils.NoiseModelCfg] = {
        agent: noise_utils.NoiseModelCfg(noise_cfg=noise_utils.GaussianNoiseCfg(mean=0.0, std=0.025, operation="add"))
        for agent in possible_agents
    }

    action_scale = 1.0
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.0
    lift_reward_scale = 1.0
    drop_penalty_scale = -1.0
    dist_g_reward_scale = 1.0
    goal_reward_scale = 1.0
    sync_reward_scale = 10.0
    action_penalty_scale = -0.15
    joint_1_penalty_scale = -0.5
    self_collision_penalty_scale = -0.15
    contact_ground_penalty_scale = 0.0
    contact_goal_penalty_scale = -0.15
    touch_penalty_scale = -0.25
    dist_g_penalty_scale = -1.0
    # touch_penalty_scale = -0.25
    # dist_g_penalty_scale = -1.0


class Turtlebot3MultiPlaceEnv(DirectMARLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: Turtlebot3MultiPlaceEnvCfg

    def __init__(self, cfg: Turtlebot3MultiPlaceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_lower_limits_1 = self._robot_1.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits_1 = self._robot_1.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_vel_limits_tensor_1 = self._robot_1.data.joint_velocity_limits[0, :].to(device=self.device)

        self.robot_dof_lower_limits_2 = self._robot_1.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits_2 = self._robot_1.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_vel_limits_tensor_2 = self._robot_1.data.joint_velocity_limits[0, :].to(device=self.device)

        self.joint_pos_names = ["joint.*", "gripper_.*"]
        self.arm_names = ["joint.*"]
        self.gripper_names = ["gripper_.*"]
        self.wheel_names = ["wheel_.*"]
        self.joint_pos_ids_1, _ = self._robot_1.find_joints(self.joint_pos_names, preserve_order=False)
        self.arm_ids_1, _ = self._robot_1.find_joints(self.arm_names, preserve_order=False)
        self.gripper_ids_1, _ = self._robot_1.find_joints(self.gripper_names, preserve_order=False)
        self.wheel_ids_1, _ = self._robot_1.find_joints(self.wheel_names, preserve_order=False)
        self.joint_pos_ids_2, _ = self._robot_1.find_joints(self.joint_pos_names, preserve_order=False)
        self.arm_ids_2, _ = self._robot_1.find_joints(self.arm_names, preserve_order=False)
        self.gripper_ids_2, _ = self._robot_1.find_joints(self.gripper_names, preserve_order=False)
        self.wheel_ids_2, _ = self._robot_1.find_joints(self.wheel_names, preserve_order=False)

        self.joint_1_ids_1, _ = self._robot_1.find_joints("joint1", preserve_order=False)
        self.joint_1_ids_2, _ = self._robot_2.find_joints("joint1", preserve_order=False)

        self.default_joint_1_pos_1 = self._robot_1.data.default_joint_pos[:, self.joint_1_ids_1]
        self.default_joint_1_pos_2 = self._robot_2.data.default_joint_pos[:, self.joint_1_ids_2]

        # self.robot_dof_lower_limits_1[joint_1_ids_1] = -0.5235987
        # self.robot_dof_lower_limits_2[joint_1_ids_2] = -0.5235987
        # self.robot_dof_upper_limits_1[joint_1_ids_1] = 0.5235987
        # self.robot_dof_upper_limits_2[joint_1_ids_2] = 0.5235987

        self.robot_arm_targets_1 = torch.zeros((self.num_envs, len(self.arm_ids_1)), device=self.device)
        self.robot_gripper_targets_1 = torch.zeros((self.num_envs, len(self.gripper_ids_1)), device=self.device)
        self.robot_wheel_targets_1 = torch.zeros((self.num_envs, len(self.wheel_ids_1)), device=self.device)

        self.robot_arm_targets_2 = torch.zeros((self.num_envs, len(self.arm_ids_2)), device=self.device)
        self.robot_gripper_targets_2 = torch.zeros((self.num_envs, len(self.gripper_ids_2)), device=self.device)
        self.robot_wheel_targets_2 = torch.zeros((self.num_envs, len(self.wheel_ids_2)), device=self.device)

        self.curr_actions_1 = torch.zeros((self.num_envs, self.cfg.action_spaces["robot_1"]), device=self.device)
        self.prev_actions_1 = torch.zeros((self.num_envs, self.cfg.action_spaces["robot_1"]), device=self.device)

        self.curr_actions_2 = torch.zeros((self.num_envs, self.cfg.action_spaces["robot_2"]), device=self.device)
        self.prev_actions_2 = torch.zeros((self.num_envs, self.cfg.action_spaces["robot_2"]), device=self.device)

        self.joint_list_1 = CircularBuffer(max_len=5, batch_size=self.num_envs, device=self.device)  
        self.joint_list_2 = CircularBuffer(max_len=5, batch_size=self.num_envs, device=self.device)

        self.action_list_1 = CircularBuffer(max_len=5, batch_size=self.num_envs, device=self.device)  
        self.action_list_2 = CircularBuffer(max_len=5, batch_size=self.num_envs, device=self.device)

        self.object_list_1 = CircularBuffer(max_len=5, batch_size=self.num_envs, device=self.device)  
        self.object_list_2 = CircularBuffer(max_len=5, batch_size=self.num_envs, device=self.device)

        self.goal_list_1 = CircularBuffer(max_len=5, batch_size=self.num_envs, device=self.device)  
        self.goal_list_2 = CircularBuffer(max_len=5, batch_size=self.num_envs, device=self.device)

        self.actor = torch.zeros((self.num_envs, 256), device=self.device)

        self.lift_1 = torch.full_like(self.episode_length_buf, -1)
        self.reach_1 = torch.full_like(self.episode_length_buf, -1)
        self.drop_1 = torch.full_like(self.episode_length_buf, -1)
        self.unreach_1 = torch.full_like(self.episode_length_buf, -1)
        self.goal_1 = torch.full_like(self.episode_length_buf, -1)
        self.goal_count_1 = torch.zeros_like(self.episode_length_buf, dtype=torch.float32)
        self.task_id_1 = torch.zeros(self.num_envs, device=self.device)
        self.prev_task_id_1 = torch.zeros_like(self.task_id_1)
        self.total_changes_1 = torch.zeros_like(self.task_id_1)
        
        self.lift_2 = torch.full_like(self.episode_length_buf, -1)
        self.reach_2 = torch.full_like(self.episode_length_buf, -1)
        self.drop_2 = torch.full_like(self.episode_length_buf, -1)
        self.unreach_2 = torch.full_like(self.episode_length_buf, -1)
        self.goal_2 = torch.full_like(self.episode_length_buf, -1)
        self.goal_count_2 = torch.zeros_like(self.episode_length_buf, dtype=torch.float32)
        self.task_id_2 = torch.zeros(self.num_envs, device=self.device)
        self.prev_task_id_2 = torch.zeros_like(self.task_id_2)
        self.total_changes_2 = torch.zeros_like(self.task_id_2)
        
        self.success_time = torch.full_like(self.episode_length_buf, -1)
        self.success_time_1 = torch.full_like(self.episode_length_buf, -1)
        self.success_time_2 = torch.full_like(self.episode_length_buf, -1)

        self.log = torch.ones_like(self.episode_length_buf, dtype=torch.bool)
        self.log_1 = torch.ones_like(self.episode_length_buf, dtype=torch.bool)
        self.log_2 = torch.ones_like(self.episode_length_buf, dtype=torch.bool)
        
        self.sync_awarded = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.sync_award_step = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        self.sync_window_open = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.single_runlen_1 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.single_runlen_2 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.single_runlen_max_1 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.single_runlen_max_2 = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.single_skill_global_1 = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.single_skill_global_2 = torch.tensor(0.0, dtype=torch.float32, device=self.device)

        self.reward_1 = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.reward_2 = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.reward_delay = False

        self.dist_g_penalty_scale = self.cfg.dist_g_penalty_scale
        self.touch_penalty_scale = self.cfg.touch_penalty_scale

        self.over_episode = 0
        
        self.csv_path = "success_log.csv"
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "success_count", "delay_1", "delay_2"])

        self.episode_idx = 0

    def _setup_scene(self):
        self._robot_1 = Articulation(self.cfg.robot_1)
        # self._camera_1 = TiledCamera(self.cfg.camera_1)
        self._contact_base_1 = ContactSensor(self.cfg.contact_base_1)
        self._contact_gripper_object_1 = ContactSensor(self.cfg.contact_gripper_object_1)
        self._contact_robot_goal_1 = ContactSensor(self.cfg.contact_robot_goal_1)
        self._ee_frame_1 = FrameTransformer(self.cfg.ee_frame_1)
        self._lee_frame_1 = FrameTransformer(self.cfg.lee_frame_1)
        self._ree_frame_1 = FrameTransformer(self.cfg.ree_frame_1)
        self._camera_frame_1 = FrameTransformer(self.cfg.camera_frame_1)

        self._robot_2 = Articulation(self.cfg.robot_2)
        # self._camera_2 = TiledCamera(self.cfg.camera_2)
        self._contact_base_2 = ContactSensor(self.cfg.contact_base_2)
        self._contact_gripper_object_2 = ContactSensor(self.cfg.contact_gripper_object_2)
        self._contact_robot_goal_2 = ContactSensor(self.cfg.contact_robot_goal_2)
        self._ee_frame_2 = FrameTransformer(self.cfg.ee_frame_2)
        self._lee_frame_2 = FrameTransformer(self.cfg.lee_frame_2)
        self._ree_frame_2 = FrameTransformer(self.cfg.ree_frame_2)
        self._camera_frame_2 = FrameTransformer(self.cfg.camera_frame_2)

        self._cube_1 = RigidObject(self.cfg.cube_1)
        self._cube_2 = RigidObject(self.cfg.cube_2)

        # self._ground_plane = RigidObject(self.cfg.ground_plane)

        # self._goal_base = RigidObject(self.cfg.goal_base)
        self._goal = RigidObject(self.cfg.goal)
        self._goal_frame_1 = FrameTransformer(self.cfg.goal_frame_1)
        self._goal_frame_2 = FrameTransformer(self.cfg.goal_frame_2)
        self._goal_above_frame_1 = FrameTransformer(self.cfg.goal_above_frame_1)
        self._goal_above_frame_2 = FrameTransformer(self.cfg.goal_above_frame_2)
        self._contact_object_goal_1 = ContactSensor(self.cfg.contact_object_goal_1)
        self._contact_object_goal_2 = ContactSensor(self.cfg.contact_object_goal_2)

        # self.goal_markers = VisualizationMarkers(self.cfg.goal)

        # ロボットをシーンに追加
        self.scene.articulations["robot_1"] = self._robot_1
        # self.scene.sensors["camera_1"] = self._camera_1
        self.scene.sensors["contact_base_1"] = self._contact_base_1
        self.scene.sensors["contact_gripper_object_1"] = self._contact_gripper_object_1
        self.scene.sensors["contact_robot_goal_1"] = self._contact_robot_goal_1
        self.scene.sensors["ee_frame_1"] = self._ee_frame_1
        self.scene.sensors["lee_frame_1"] = self._lee_frame_1
        self.scene.sensors["ree_frame_1"] = self._ree_frame_1
        self.scene.sensors["camera_frame_1"] = self._camera_frame_1

        self.scene.articulations["robot_2"] = self._robot_2
        # self.scene.sensors["camera_2"] = self._camera_2
        self.scene.sensors["contact_base_2"] = self._contact_base_2
        self.scene.sensors["contact_gripper_object_2"] = self._contact_gripper_object_2
        self.scene.sensors["contact_robot_goal_2"] = self._contact_robot_goal_2
        self.scene.sensors["ee_frame_2"] = self._ee_frame_2
        self.scene.sensors["lee_frame_2"] = self._lee_frame_2
        self.scene.sensors["ree_frame_2"] = self._ree_frame_2
        self.scene.sensors["camera_frame_2"] = self._camera_frame_2

        self.scene.rigid_objects["cube_1"] = self._cube_1
        self.scene.rigid_objects["cube_2"] = self._cube_2

        # self.scene.rigid_objects["ground_plane"] = self._ground_plane

        # self.scene.rigid_objects["goal_base"] = self._goal_base
        self.scene.rigid_objects["goal"] = self._goal
        self.scene.sensors["goal_frame_1"] = self._goal_frame_1
        self.scene.sensors["goal_frame_2"] = self._goal_frame_2
        self.scene.sensors["goal_above_frame_1"] = self._goal_above_frame_1
        self.scene.sensors["goal_above_frame_2"] = self._goal_above_frame_2
        self.scene.sensors["contact_object_goal_1"] = self._contact_object_goal_1
        self.scene.sensors["contact_object_goal_2"] = self._contact_object_goal_2

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # omni.kit.commands.execute(
        #     "ToggleVisibilitySelectedPrims",
        #     selected_paths=["/World/ground"]
        # )

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        arm_actions_1 = actions["robot_1"][:, :len(self.arm_ids_1)].clone().clamp(-1.0, 1.0)
        gripper_action_1 = actions["robot_1"][:, len(self.arm_ids_1)].clone().clamp(-1.0, 1.0) 
        wheel_actions_1 = actions["robot_1"][:, len(self.arm_ids_1)+1:].clone().clamp(-1.0, 1.0)

        arm_actions_2 = actions["robot_2"][:, :len(self.arm_ids_2)].clone().clamp(-1.0, 1.0)
        gripper_action_2 = actions["robot_2"][:, len(self.arm_ids_2)].clone().clamp(-1.0, 1.0) 
        wheel_actions_2 = actions["robot_2"][:, len(self.arm_ids_2)+1:].clone().clamp(-1.0, 1.0)

        arm_targets_1 = self._robot_1.data.joint_pos[:, self.arm_ids_1] + self.robot_dof_vel_limits_tensor_1[self.arm_ids_1] * self.dt * arm_actions_1
        self.robot_arm_targets_1[:] = torch.clamp(arm_targets_1, self.robot_dof_lower_limits_1[self.arm_ids_1], self.robot_dof_upper_limits_1[self.arm_ids_1])

        arm_targets_2 = self._robot_2.data.joint_pos[:, self.arm_ids_2] + self.robot_dof_vel_limits_tensor_2[self.arm_ids_2] * self.dt * arm_actions_2
        self.robot_arm_targets_2[:] = torch.clamp(arm_targets_2, self.robot_dof_lower_limits_2[self.arm_ids_2], self.robot_dof_upper_limits_2[self.arm_ids_2])

        gripper_actions_1 = torch.zeros(self.num_envs, len(self.gripper_ids_1), device=self.device)
        gripper_actions_1[:, 0] = torch.where(gripper_action_1 >= 0.0, self.robot_dof_upper_limits_1[self.gripper_ids_1[0]].item(),
                                      self.robot_dof_lower_limits_1[self.gripper_ids_1[0]].item())
        gripper_actions_1[:, 1] = torch.where(gripper_action_1 >= 0.0, self.robot_dof_upper_limits_1[self.gripper_ids_1[1]].item(),
                                      self.robot_dof_lower_limits_1[self.gripper_ids_1[1]].item())
        self.robot_gripper_targets_1[:] = gripper_actions_1

        gripper_actions_2 = torch.zeros(self.num_envs, len(self.gripper_ids_2), device=self.device)
        gripper_actions_2[:, 0] = torch.where(gripper_action_2 >= 0.0, self.robot_dof_upper_limits_2[self.gripper_ids_2[0]].item(),
                                      self.robot_dof_lower_limits_2[self.gripper_ids_2[0]].item())
        gripper_actions_2[:, 1] = torch.where(gripper_action_2 >= 0.0, self.robot_dof_upper_limits_2[self.gripper_ids_2[1]].item(),
                                      self.robot_dof_lower_limits_2[self.gripper_ids_2[1]].item())
        self.robot_gripper_targets_2[:] = gripper_actions_2

        wheel_deadband = 0.4
        abs_w_1 = wheel_actions_1.abs()
        wheel_actions_1 = torch.where(
            abs_w_1 < wheel_deadband,
            torch.zeros_like(wheel_actions_1),
            wheel_actions_1
        )

        abs_w_2 = wheel_actions_2.abs()
        wheel_actions_2 = torch.where(
            abs_w_2 < wheel_deadband,
            torch.zeros_like(wheel_actions_2),
            wheel_actions_2
        )

        self.robot_wheel_targets_1[:] = wheel_actions_1 * self.robot_dof_vel_limits_tensor_1[self.wheel_ids_1]
        self.robot_wheel_targets_2[:] = wheel_actions_2 * self.robot_dof_vel_limits_tensor_2[self.wheel_ids_2]

        self.prev_actions_1 = self.curr_actions_1.clone()
        self.prev_actions_2 = self.curr_actions_2.clone()

        self.curr_actions_1 = actions["robot_1"]
        self.curr_actions_2 = actions["robot_2"]

    def _apply_action(self):
        # 制御
        self._robot_1.set_joint_position_target(self.robot_arm_targets_1, self.arm_ids_1)
        self._robot_1.set_joint_position_target(self.robot_gripper_targets_1, self.gripper_ids_1)
        self._robot_1.set_joint_velocity_target(self.robot_wheel_targets_1, self.wheel_ids_1)

        self._robot_2.set_joint_position_target(self.robot_arm_targets_2, self.arm_ids_2)
        self._robot_2.set_joint_position_target(self.robot_gripper_targets_2, self.gripper_ids_2)
        self._robot_2.set_joint_velocity_target(self.robot_wheel_targets_2, self.wheel_ids_2)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # break_goal_1 = self.goal_pos_1[:, 2] < 0.035
        # break_goal_2 = self.goal_pos_2[:, 2] < 0.035
        # out_of_ =  break_goal_1 | break_goal_2
        # out_of_ = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        out_of_1 = (self.task_id_1 != self.prev_task_id_1)
        out_of_2 = (self.task_id_2 != self.prev_task_id_2)
        self.prev_task_id_1[:] = self.task_id_1[:]
        self.prev_task_id_2[:] = self.task_id_2[:]
        terminated = {"robot_1": out_of_1, "robot_2": out_of_2}
        # terminated = {agent: out_of_ for agent in self.cfg.possible_agents}
        # if self.reward_delay:
        #     time_out = self.episode_length_buf >= (self.max_episode_length - 1) / 2
        # else:
        #     time_out = self.episode_length_buf >= self.max_episode_length - 1
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        
        return terminated, time_outs

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        return self._compute_rewards(
            self.ee_pos_1,
            self.lee_pos_1,
            self.ree_pos_1,
            self.ee_pos_2,
            self.lee_pos_2,
            self.ree_pos_2,
            self.joint_pos_1,
            self.joint_pos_2,
            self.cube_pos_1,
            self.cube_pos_2,
            self.camera_to_object_pos_1,
            self.camera_to_object_pos_2,
            self.goal_pos_1,
            self.goal_pos_2,
            self.contact_base_1,
            self.contact_gripper_object_1,
            self.contact_robot_goal_1,
            self.contact_base_2,
            self.contact_gripper_object_2,
            self.contact_robot_goal_2,
            self.contact_object_goal_1,
            self.contact_object_goal_2,
            self.cfg.dist_reward_scale,
            self.cfg.lift_reward_scale,
            self.cfg.dist_g_reward_scale,
            self.cfg.goal_reward_scale,
            self.cfg.sync_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.joint_1_penalty_scale,
            self.cfg.self_collision_penalty_scale,
            self.cfg.contact_ground_penalty_scale,
            self.cfg.contact_goal_penalty_scale,
            self.touch_penalty_scale,
            self.dist_g_penalty_scale,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        joint_pos_1 = self._robot_1.data.default_joint_pos[env_ids] + sample_uniform(
            -0.5,
            0.5,
            (len(env_ids), self._robot_1.num_joints),
            self.device,
        )
        joint_pos_2 = self._robot_2.data.default_joint_pos[env_ids] + sample_uniform(
            -0.5,
            0.5,
            (len(env_ids), self._robot_2.num_joints),
            self.device,
        )
        
        joint_pos_1 = torch.clamp(joint_pos_1, self.robot_dof_lower_limits_1, self.robot_dof_upper_limits_2)
        # joint_pos_1 = torch.clamp(self._robot_1.data.default_joint_pos[env_ids], self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel_1 = torch.zeros_like(joint_pos_1)
        default_robot_state_1 = self._robot_1.data.default_root_state[env_ids].clone()
        default_robot_state_1[:, :3] += self.scene.env_origins[env_ids]
        default_robot_state_1[:, :3] += self.reset_root_state_uniform(
            env_ids=env_ids,
            pose_range = {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            # pose_range = {"x": (-0.0, 0.0), "y": (-0.0, 0.0), "z": (0.01, 0.01)},
            # pose_range = {"x": (0.0, 0.0), "y": (5.0, 5.0), "z": (0.001, 0.001)},
        )
        self._robot_1.write_root_link_pose_to_sim(default_robot_state_1[:, :7], env_ids=env_ids)
        self._robot_1.write_root_com_velocity_to_sim(default_robot_state_1[:, 7:], env_ids=env_ids)
        self._robot_1.set_joint_position_target(joint_pos_1, env_ids=env_ids)
        self._robot_1.write_joint_state_to_sim(joint_pos_1, joint_vel_1, env_ids=env_ids)

        joint_pos_2 = torch.clamp(joint_pos_2, self.robot_dof_lower_limits_2, self.robot_dof_upper_limits_2)
        # joint_pos_2 = torch.clamp(self._robot_2.data.default_joint_pos[env_ids], self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel_2 = torch.zeros_like(joint_pos_2)
        default_robot_state_2 = self._robot_2.data.default_root_state[env_ids].clone()
        default_robot_state_2[:, :3] += self.scene.env_origins[env_ids]
        default_robot_state_2[:, :3] += self.reset_root_state_uniform(
            env_ids=env_ids,
            pose_range = {"x": (1.34, 1.34), "y": (0.0, 0.0), "z": (0.0, 0.0)},
            # pose_range = {"x": (0.0, 0.0), "y": (-5.0, -5.0), "z": (0.001, 0.001)},
        )
        self._robot_2.write_root_link_pose_to_sim(default_robot_state_2[:, :7], env_ids=env_ids)
        self._robot_2.write_root_com_velocity_to_sim(default_robot_state_2[:, 7:], env_ids=env_ids)
        self._robot_2.set_joint_position_target(joint_pos_2, env_ids=env_ids)
        self._robot_2.write_joint_state_to_sim(joint_pos_2, joint_vel_2, env_ids=env_ids)

        # default_goal_base_state = self._goal_base.data.default_root_state[env_ids, :7].clone()
        # default_goal_base_state[:, :3] += self.scene.env_origins[env_ids]
        # default_goal_base_state[:, :3] += self.reset_root_state_uniform(
        #     env_ids=env_ids,
        #     pose_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.001, 0.001)},
        # )
        default_goal_state = self._goal.data.default_root_state[env_ids, :7].clone()
        default_goal_state[:, :3] += self.scene.env_origins[env_ids]
        # default_goal_state[:, :2] = default_goal_base_state[:, :2]
        default_goal_state[:, :3] += self.reset_root_state_uniform(
            env_ids=env_ids,
            # pose_range = {"x": (1.5, 1.6), "y": (-0.05, 0.05), "z": (0.001, 0.001)},
            pose_range = {"x": (0.62, 0.72), "y": (-0.05, 0.05), "z": (0.001, 0.001)},
        )
        # self._goal_base.write_root_link_pose_to_sim(default_goal_base_state[:, :7], env_ids=env_ids)
        self._goal.write_root_link_pose_to_sim(default_goal_state[:, :7], env_ids=env_ids)

        # default_cube_state_1 = self._cube_1.data.default_root_state[env_ids].clone()
        # default_cube_state_1[:, :3] = self._set_frame_1.data.target_pos_w[env_ids, 0, :3]
        # self._cube_1.write_root_link_pose_to_sim(default_cube_state_1[:, :7], env_ids=env_ids)

        # default_cube_state_2 = self._cube_2.data.default_root_state[env_ids].clone()
        # default_cube_state_2[:, :3] = self._set_frame_2.data.target_pos_w[env_ids, 0, :3]
        # self._cube_2.write_root_link_pose_to_sim(default_cube_state_2[:, :7], env_ids=env_ids)

        default_cube_state_1 = self._cube_1.data.default_root_state[env_ids].clone()
        default_cube_state_1[:, :3] += self.scene.env_origins[env_ids]
        default_cube_state_1[:, :3] += self.reset_root_state_uniform(
            env_ids=env_ids,
            pose_range = {"x": (0.25, 0.4), "y": (-0.05, 0.05), "z": (0.01, 0.01)},
            # pose_range = {"x": (-0.05, 0.05), "y": (-0.75, -0.5), "z": (0.01, 0.01)},
            # pose_range = {"x": (-0.15, 0.15), "y": (1.0, 1.25), "z": (0.01, 0.01)},
        )
        self._cube_1.write_root_link_pose_to_sim(default_cube_state_1[:, :7], env_ids=env_ids)

        default_cube_state_2 = self._cube_2.data.default_root_state[env_ids].clone()
        default_cube_state_2[:, :3] += self.scene.env_origins[env_ids]
        default_cube_state_2[:, :3] += self.reset_root_state_uniform(
            env_ids=env_ids,
            pose_range = {"x": (0.94, 1.09), "y": (-0.05, 0.05), "z": (0.01, 0.01)},
            # pose_range = {"x": (-0.15, 0.15), "y": (-1.25, -1.0), "z": (0.01, 0.01)},
        )
        self._cube_2.write_root_link_pose_to_sim(default_cube_state_2[:, :7], env_ids=env_ids)

        self.joint_list_1.reset()
        self.joint_list_2.reset()
        self.action_list_1.reset()
        self.action_list_2.reset()
        self.object_list_1.reset()
        self.object_list_2.reset()
        self.goal_list_1.reset()
        self.goal_list_2.reset()

        success_mask = self.success_time[env_ids] >= 0
        success_mask_1 = self.success_time_1[env_ids] >= 0
        success_mask_2 = self.success_time_2[env_ids] >= 0
        # if success_mask.any():
        succ_envs = env_ids[success_mask]
        succ_envs_1 = env_ids[success_mask_1]
        succ_envs_2 = env_ids[success_mask_2]
        times = self.success_time[succ_envs].float() * 0.25   # 例: 1step=0.25[s]

        mean_t = times.mean().item()
        std_t  = times.std(unbiased=False).item()  # N 分母

        tqdm.tqdm.write(f"[RESET] success (n={times.numel()}) :  {mean_t:.3f} ± {std_t:.3f}  [s]")
        tqdm.tqdm.write(f"[RESET] success_1 n={succ_envs_1.numel()}")
        tqdm.tqdm.write(f"[RESET] success_2 n={succ_envs_2.numel()}")

        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([self.episode_idx, times.numel()])

        self.episode_idx += 1

        self.lift_1[env_ids] = -1
        self.reach_1[env_ids] = -1
        self.drop_1[env_ids] = -1
        self.unreach_1[env_ids] = -1
        self.goal_1[env_ids] = -1
        self.goal_count_1[env_ids] = 0
        self.task_id_1[env_ids] = 0
        self.prev_task_id_1[env_ids] = 0
        self.total_changes_1[env_ids] = 0
        
        self.lift_2[env_ids] = -1
        self.reach_2[env_ids] = -1
        self.drop_2[env_ids] = -1
        self.unreach_2[env_ids] = -1
        self.goal_2[env_ids] = -1
        self.goal_count_2[env_ids] = 0
        self.task_id_2[env_ids] = 0
        self.prev_task_id_2[env_ids] = 0
        self.total_changes_2[env_ids] = 0

        self.success_time[env_ids] = -1
        self.success_time_1[env_ids] = -1
        self.success_time_2[env_ids] = -1
        self.log[env_ids] = True
        self.log_1[env_ids] = True
        self.log_2[env_ids] = True
        self.sync_awarded[env_ids] = False
        self.sync_award_step[env_ids] = -1
        self.sync_window_open[env_ids] = False

        success_single_envs_1 = (self.single_runlen_max_1 > 4)
        success_single_envs_2 = (self.single_runlen_max_2 > 4)

        # if self.reward_1.mean().item() >= 100 and self.reward_2.mean().item() >= 100:
        # if self.reward_1.mean().item() + self.reward_2.mean().item() >= 40:
        #     self.reward_delay = True
    
        # if self.reward_delay:     
            # threshold = (self.num_envs + 1) // 4
            # # threshold = self.num_envs * 0.3
            # if int(success_single_envs_1.sum().item()) >= threshold:
            #     self.single_skill_global_1 += 1.0
            # if int(success_single_envs_2.sum().item()) >= threshold:
            #     self.single_skill_global_2 += 1.0
            
            # decay_by_skill_1 = torch.clamp(torch.exp(-0.002 * self.single_skill_global_1), min=0.05)
            # decay_by_skill_2 = torch.clamp(torch.exp(-0.002 * self.single_skill_global_2), min=0.05)
            # tqdm.tqdm.write(f"[success_single_num_1] {success_single_envs_1.sum().item()} [success_single_num_2] {success_single_envs_2.sum().item()}")
            # tqdm.tqdm.write(f"[reward_decay_1] {decay_by_skill_1} [reward_decay_2] {decay_by_skill_2}")
        # if times.numel() >= self.num_envs/2 and self.over_episode < 20:
        #     self.dist_g_penalty_scale -= 0.05
        #     self.touch_penalty_scale -= 0.0125
        #     self.over_episode += 1
        # tqdm.tqdm.write(f"[penalty_decay] {self.dist_g_penalty_scale}")


            
        self.single_runlen_1[env_ids] = 0.0
        self.single_runlen_2[env_ids] = 0.0
        self.single_runlen_max_1[env_ids] = 0.0
        self.single_runlen_max_2[env_ids] = 0.0

        self.curr_actions_1[env_ids] = 0.0
        self.prev_actions_1[env_ids] = 0.0
        self.curr_actions_2[env_ids] = 0.0
        self.prev_actions_2[env_ids] = 0.0
        
        self.reward_1[env_ids] = 0.0
        self.reward_2[env_ids] = 0.0

        self._compute_intermediate_values()
    
    def reset_root_state_uniform(self, env_ids, pose_range):
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)

        positions = rand_samples[:, 0:3]

        return positions

    def _get_observations(self) -> dict:
        # rgb_1 = self._camera_1.data.output["rgb"] / 255.0
        # rgb_2 = self._camera_2.data.output["rgb"] / 255.0

        self.joint_list_1.append(self.joint_pos_1)
        self.joint_list_2.append(self.joint_pos_2)
        self.action_list_1.append(self.curr_actions_1)
        self.action_list_2.append(self.curr_actions_2)
        self.object_list_1.append(self.object_be_1)
        self.object_list_2.append(self.object_be_2)
        # self.object_list_1.append(self.object_b_1)
        # self.object_list_2.append(self.object_b_2)
        self.goal_list_1.append(self.goal_b_1)
        self.goal_list_2.append(self.goal_b_2)
        # self.goal_list_1.append(self.goal_b_1)
        # self.goal_list_2.append(self.goal_b_2)

        joint_list_1 = 2 * (self.joint_list_1.buffer - self.robot_dof_lower_limits_1[self.joint_pos_ids_1]) / (self.robot_dof_upper_limits_1[self.joint_pos_ids_1] - self.robot_dof_lower_limits_1[self.joint_pos_ids_1]) - 1
        joint_list_2 = 2 * (self.joint_list_2.buffer - self.robot_dof_lower_limits_2[self.joint_pos_ids_2]) / (self.robot_dof_upper_limits_2[self.joint_pos_ids_2] - self.robot_dof_lower_limits_2[self.joint_pos_ids_2]) - 1
        
        mean_1 = joint_list_1[:, :, -2:].mean(dim=-1, keepdim=True)
        joint_list_1[:, :, -2:] = mean_1.expand(-1, -1, 2)

        mean_2 = joint_list_2[:, :, -2:].mean(dim=-1, keepdim=True)
        joint_list_2[:, :, -2:] = mean_2.expand(-1, -1, 2)

        obs = {
            "robot_1": {
                # "joint": self.joint_list_1.buffer,
                "joint": joint_list_1,
                # "joint": self.joint_pos_1,
                # "rgb": rgb_1,
                "object": self.object_list_1.buffer,
                # "object": self.object_b_1,
                "goal": self.goal_list_1.buffer,
                # "goal": self.goal_b_1,
                "actions": self.action_list_1.buffer,
                # "actions": self.curr_actions_1,
                # "joint_other": self.joint_list_2.buffer,
                "joint_other": joint_list_2,
                # "joint_other": self.joint_pos_2,
                "object_other": self.object_list_2.buffer,
                # "object_other": self.object_b_2,
                "goal_other": self.goal_list_2.buffer,
                # "goal_other": self.goal_b_2,
                "actions_other": self.action_list_2.buffer,
                # "actions_other": self.curr_actions_2,
                "task_id": self.task_id_1,
            },
            "robot_2": {
                # "joint": self.joint_list_2.buffer,
                "joint": joint_list_2,
                # "joint": self.joint_pos_,
                # "rgb": rgb_1,
                "object": self.object_list_2.buffer,
                # "object": self.object_b_1,
                "goal": self.goal_list_2.buffer,
                # "goal": self.goal_b_1,
                "actions": self.action_list_2.buffer,
                # "actions": self.curr_actions_1,
                # "joint_other": self.joint_list_1.buffer,
                "joint_other": joint_list_1,
                # "joint_other": self.joint_pos_2,
                "object_other": self.object_list_1.buffer,
                # "object_other": self.object_b_2,
                "goal_other": self.goal_list_1.buffer,
                # "goal_other": self.goal_b_2,
                "actions_other": self.action_list_1.buffer,
                # "actions_other": self.curr_actions_2,
                "task_id": self.task_id_2,
            },
        }
        
        return obs
    
    def _get_states(self) -> dict:

        state = torch.cat(
            (
               self.joint_pos_1,
               self.object_be_1,
            #    self.goal_b_1,
               self.curr_actions_1,
               self.joint_pos_2,
               self.object_be_2,
            #    self.goal_b_2,
               self.curr_actions_2 
            ),
            dim=-1
        )
        
        return state

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot_1._ALL_INDICES
    
        self.joint_pos_1 = self._robot_1.data.joint_pos[:, self.joint_pos_ids_1]
        self.joint_pos_2 = self._robot_2.data.joint_pos[:, self.joint_pos_ids_2]
        self.joint_1_pos_1 = self._robot_1.data.joint_pos[:, self.joint_1_ids_1]
        self.joint_1_pos_2 = self._robot_2.data.joint_pos[:, self.joint_1_ids_2]
        
        self.base_pos_1 = self._robot_1.data.root_link_state_w[env_ids, :3]
        self.base_rot_1 = self._robot_1.data.root_link_state_w[env_ids, 3:7]
        self.ee_pos_1 = self._ee_frame_1.data.target_pos_w[env_ids, 0, :]
        self.ee_rot_1 = self._ee_frame_1.data.target_quat_w[env_ids, 0, :]
        self.lee_pos_1 = self._lee_frame_1.data.target_pos_w[env_ids, 0, :]
        self.ree_pos_1 = self._ree_frame_1.data.target_pos_w[env_ids, 0, :]
        self.camera_pos_1 = self._camera_frame_1.data.target_pos_w[env_ids, 0, :]
        self.camera_rot_1 = self._camera_frame_1.data.target_quat_w[env_ids, 0, :]
        
        self.base_pos_2 = self._robot_2.data.root_link_state_w[env_ids, :3]
        self.base_rot_2 = self._robot_2.data.root_link_state_w[env_ids, 3:7]
        self.ee_pos_2 = self._ee_frame_2.data.target_pos_w[env_ids, 0, :]
        self.ee_rot_2 = self._ee_frame_2.data.target_quat_w[env_ids, 0, :]
        self.lee_pos_2 = self._lee_frame_2.data.target_pos_w[env_ids, 0, :]
        self.ree_pos_2 = self._ree_frame_2.data.target_pos_w[env_ids, 0, :]
        self.camera_pos_2 = self._camera_frame_2.data.target_pos_w[env_ids, 0, :]
        self.camera_rot_2 = self._camera_frame_2.data.target_quat_w[env_ids, 0, :]
        
        self.cube_pos_1 = self._cube_1.data.root_link_state_w[env_ids, :3]
        self.cube_rot_1 = self._cube_1.data.root_link_state_w[env_ids, 3:7]

        self.cube_pos_2 = self._cube_2.data.root_link_state_w[env_ids, :3]
        self.cube_rot_2 = self._cube_2.data.root_link_state_w[env_ids, 3:7]

        object_pos_e_1, object_rot_e_1 = math_utils.subtract_frame_transforms(
            self.ee_pos_1, self.ee_rot_1, self.cube_pos_1, self.cube_rot_1
        )

        object_pos_e_2, object_rot_e_2 = math_utils.subtract_frame_transforms(
            self.ee_pos_2, self.ee_rot_2, self.cube_pos_2, self.cube_rot_2
        )

        object_pos_b_1, object_rot_b_1 = math_utils.subtract_frame_transforms(
            self.base_pos_1, self.base_rot_1, self.cube_pos_1, self.cube_rot_1
        )

        object_pos_b_2, object_rot_b_2 = math_utils.subtract_frame_transforms(
            self.base_pos_2, self.base_rot_2, self.cube_pos_2, self.cube_rot_2
        )

        # print(1, object_rot_b_1)
        # print(2, object_rot_b_2)

        self.object_be_1 = torch.cat((object_pos_b_1, object_rot_b_1, object_pos_e_1, object_rot_e_1), dim=1)
        self.object_be_2 = torch.cat((object_pos_b_2, object_rot_b_2, object_pos_e_2, object_rot_e_2), dim=1)

        goal_pos_1 = self._goal_frame_1.data.target_pos_w[env_ids, 0, :]
        goal_rot_1 = self._goal_frame_1.data.target_quat_w[env_ids, 0, :]
        goal_pos_2 = self._goal_frame_2.data.target_pos_w[env_ids, 0, :]
        goal_rot_2 = self._goal_frame_2.data.target_quat_w[env_ids, 0, :]

        # goal_pos_e_1, goal_rot_e_1 = math_utils.subtract_frame_transforms(
        #     self.ee_pos_1, self.ee_rot_1, goal_pos_1, goal_rot_1
        # )
        
        # goal_pos_e_2, goal_rot_e_2 = math_utils.subtract_frame_transforms(
        #     self.ee_pos_2, self.ee_rot_2, goal_pos_2, goal_rot_2
        # )
        
        goal_pos_b_1, goal_rot_b_1 = math_utils.subtract_frame_transforms(
            self.base_pos_1, self.base_rot_1, goal_pos_1, goal_rot_1
        )
        # self.goal_b_1 = torch.cat((goal_pos_b_1, goal_rot_b_1), dim=1)

        goal_pos_b_2, goal_rot_b_2 = math_utils.subtract_frame_transforms(
            self.base_pos_2, self.base_rot_2, goal_pos_2, goal_rot_2
        )
        # self.goal_b_2 = torch.cat((goal_pos_b_2, goal_rot_b_2), dim=1)

        self.goal_b_1= torch.cat((goal_pos_b_1, goal_rot_b_1), dim=1)
        self.goal_b_2= torch.cat((goal_pos_b_2, goal_rot_b_2), dim=1)
        # self.goal_be_1= torch.cat((goal_pos_b_1, goal_rot_b_1, goal_pos_e_1, goal_rot_e_1), dim=1)
        # self.goal_be_2= torch.cat((goal_pos_b_2, goal_rot_b_2, goal_pos_e_2, goal_rot_e_2), dim=1)
        
        self.goal_pos_1 = self._goal_above_frame_1.data.target_pos_w[env_ids, 0, :]
        self.goal_pos_2 = self._goal_above_frame_2.data.target_pos_w[env_ids, 0, :]

        self.contact_base_1 = self._contact_base_1.data.force_matrix_w[env_ids, :]
        self.contact_gripper_object_1 = self._contact_gripper_object_1.data.force_matrix_w[env_ids, :]
        self.contact_robot_goal_1 = self._contact_robot_goal_1.data.force_matrix_w[env_ids, :]
        
        self.contact_base_2 = self._contact_base_2.data.force_matrix_w[env_ids, :]
        self.contact_gripper_object_2 = self._contact_gripper_object_2.data.force_matrix_w[env_ids, :]
        self.contact_robot_goal_2 = self._contact_robot_goal_2.data.force_matrix_w[env_ids, :]

        self.contact_object_goal_1 = self._contact_object_goal_1.data.force_matrix_w[env_ids, :]
        self.contact_object_goal_2 = self._contact_object_goal_2.data.force_matrix_w[env_ids, :]

        self.camera_to_object_pos_1, _ = math_utils.subtract_frame_transforms(
            self.camera_pos_1, self.camera_rot_1, self.cube_pos_1, self.cube_rot_1
        )

        self.camera_to_object_pos_2, _ = math_utils.subtract_frame_transforms(
            self.camera_pos_2, self.camera_rot_2, self.cube_pos_2, self.cube_rot_2
        )

    def _compute_rewards(
        self,
        ee_pos_1,
        lee_pos_1,
        ree_pos_1,
        ee_pos_2,
        lee_pos_2,
        ree_pos_2,
        joint_pos_1,
        joint_pos_2,    
        cube_pos_1,
        cube_pos_2,
        camera_to_object_pos_1,
        camera_to_object_pos_2,
        goal_pos_1,
        goal_pos_2,
        contact_base_1,
        contact_gripper_object_1,
        contact_robot_goal_1,
        contact_base_2,
        contact_gripper_object_2,
        contact_robot_goal_2,
        contact_object_goal_1,
        contact_object_goal_2,
        dist_reward_scale,
        lift_reward_scale,
        dist_g_reward_scale,
        goal_reward_scale,
        sync_reward_scale,
        action_penalty_scale,
        joint_1_penalty_scale,
        self_collision_penalty_scale,
        contact_ground_penalty_scale,
        contact_goal_penalty_scale,
        touch_penalty_scale,
        dist_g_penalty_scale,
    ):
        d_c_1 = torch.norm(cube_pos_1-ee_pos_1, dim=-1)
        d_l_1 = torch.norm(cube_pos_1-lee_pos_1, dim=-1)
        d_r_1 = torch.norm(cube_pos_1-ree_pos_1, dim=-1)
        dis_1 = (d_c_1*2 + d_l_1 + d_r_1) / 4
        # dis_reward_1 = torch.exp(-30*dis_1) + torch.exp(-5*dis_1) * 0.3
        # dis_reward_1 = torch.exp(-10*dis_1)

        d_c_2 = torch.norm(cube_pos_2-ee_pos_2, dim=-1)
        d_l_2 = torch.norm(cube_pos_2-lee_pos_2, dim=-1)
        d_r_2 = torch.norm(cube_pos_2-ree_pos_2, dim=-1)
        dis_2 = (d_c_2*2 + d_l_2 + d_r_2) / 4
        # dis_reward_2 = torch.exp(-30*dis_2) + torch.exp(-5*dis_2) * 0.3
        # dis_reward_2 = torch.exp(-10*dis_2)

        d_g_1 = torch.norm(cube_pos_1-goal_pos_1, dim=-1)
        # goal_pos_1[:, 2] -=0.028
        # d_g_1_ = torch.norm(cube_pos_1-goal_pos_1, dim=-1)
        # dis_g_reward_1 = torch.exp(-5*d_g_1) + torch.exp(-30*d_g_1)
        dis_g_penalty_1 = torch.tanh(10*d_g_1)

        d_g_2 = torch.norm(cube_pos_2-goal_pos_2, dim=-1)
        # goal_pos_2[:, 2] -=0.028
        # d_g_2_ = torch.norm(cube_pos_2-goal_pos_2, dim=-1)
        # dis_g_reward_2 = torch.exp(-5*d_g_2) + torch.exp(-30*d_g_2)
        dis_g_penalty_2 = torch.tanh(10*d_g_2)

        contact_gripper_object_reward_1 = torch.norm(contact_gripper_object_1, dim=-1).squeeze() > 0.1
        catch_object_1 = contact_gripper_object_reward_1.any(dim=-1)
        lift_reward_1 = torch.where(cube_pos_1[:, 2] > 0.03, 1.0, 0.0) * catch_object_1 * torch.where(d_c_1 < 0.025, 1.0, 0.0)
        
        contact_gripper_object_reward_2 = torch.norm(contact_gripper_object_2, dim=-1).squeeze() > 0.1
        catch_object_2 = contact_gripper_object_reward_2.any(dim=-1)
        lift_reward_2 = torch.where(cube_pos_2[:, 2] > 0.03, 1.0, 0.0) * catch_object_2 * torch.where(d_c_2 < 0.025, 1.0, 0.0)

        is_lifted_1 = lift_reward_1.bool()
        is_dropped_1 = ~(catch_object_1 * torch.where(d_c_1 < 0.025, 1.0, 0.0)).bool()
        is_lifted_2 = lift_reward_2.bool()
        is_dropped_2 = ~(catch_object_2 * torch.where(d_c_2 < 0.025, 1.0, 0.0)).bool()
        
        self.lift_1[is_dropped_1] = -1
        new_lift_mask_1 = is_lifted_1 & (self.lift_1 == -1)
        self.lift_1[new_lift_mask_1] = self.episode_length_buf[new_lift_mask_1]
        success_lift_mask_1 = (
            is_lifted_1 &
            (self.lift_1 != -1) &
            ((self.episode_length_buf - self.lift_1) >= 1)
        )

        self.lift_2[is_dropped_2] = -1
        new_lift_mask_2 = is_lifted_2 & (self.lift_2 == -1)
        self.lift_2[new_lift_mask_2] = self.episode_length_buf[new_lift_mask_2]
        success_lift_mask_2 = (
            is_lifted_2 &
            (self.lift_2 != -1) &
            ((self.episode_length_buf - self.lift_2) >= 1)
        )

        is_reached_1 = (d_g_1 < 0.01) & is_lifted_1
        is_unreached_1 = ~is_reached_1
        is_reached_2 = (d_g_2 < 0.01) & is_lifted_2
        is_unreached_2 = ~is_reached_2
        
        self.reach_1[is_unreached_1] = -1
        new_reach_mask_1 = is_reached_1 & (self.reach_1 == -1)
        self.reach_1[new_reach_mask_1] = self.episode_length_buf[new_reach_mask_1]
        success_reach_mask_1 = (
            is_reached_1 &
            (self.reach_1 != -1) &
            ((self.episode_length_buf - self.reach_1) >= 1)
        )

        self.reach_2[is_unreached_2] = -1
        new_reach_mask_2 = is_reached_2 & (self.reach_2 == -1)
        self.reach_2[new_reach_mask_2] = self.episode_length_buf[new_reach_mask_2]
        success_reach_mask_2 = (
            is_reached_2 &
            (self.reach_2 != -1) &
            ((self.episode_length_buf - self.reach_2) >= 1)
        )

        is_goal_1 = (
            (contact_object_goal_1.select(dim=-1, index=2).squeeze() > 0.4) &
            ~catch_object_1
        )

        is_goal_2 = (
            (contact_object_goal_2.select(dim=-1, index=2).squeeze() > 0.4) &
            ~catch_object_2
        )

        if is_goal_1.ndim == 0:
            is_goal_1 = is_goal_1.unsqueeze(-1)

        if is_goal_2.ndim == 0:
            is_goal_2 = is_goal_2.unsqueeze(-1)

        self.drop_1[is_lifted_1 | is_goal_1] = -1
        new_drop_mask_1 = is_dropped_1 & (self.drop_1 == -1)
        self.drop_1[new_drop_mask_1] = self.episode_length_buf[new_drop_mask_1]
        failed_lift_mask_1 = (
            is_dropped_1 &
            (self.drop_1 != -1) &
            ((self.episode_length_buf - self.drop_1) >= 12)
        )

        self.drop_2[is_lifted_2 | is_goal_2] = -1
        new_drop_mask_2 = is_dropped_2 & (self.drop_2 == -1)
        self.drop_2[new_drop_mask_2] = self.episode_length_buf[new_drop_mask_2]
        failed_lift_mask_2 = (
            is_dropped_2 &
            (self.drop_2 != -1) &
            ((self.episode_length_buf - self.drop_2) >= 12)
        )

        self.unreach_1[is_reached_1 | is_goal_1] = -1
        new_unreach_mask_1 = is_unreached_1 & (self.unreach_1 == -1)
        self.unreach_1[new_unreach_mask_1] = self.episode_length_buf[new_unreach_mask_1]
        failed_reach_mask_1 = (
            is_unreached_1 &
            (self.unreach_1 != -1) &
            ((self.episode_length_buf - self.unreach_1) >= 12)
        )

        self.unreach_2[is_reached_2 | is_goal_2] = -1
        new_unreach_mask_2 = is_unreached_2 & (self.unreach_2 == -1)
        self.unreach_2[new_unreach_mask_2] = self.episode_length_buf[new_unreach_mask_2]
        failed_reach_mask_2 = (
            is_unreached_2 &
            (self.unreach_2 != -1) &
            ((self.episode_length_buf - self.unreach_2) >= 12)
        )
        single_only_1 =  is_goal_1 & ~is_goal_2
        single_only_2 = ~is_goal_1 & is_goal_2

        self.single_runlen_1 = torch.where(single_only_1, self.single_runlen_1 + 1, torch.zeros_like(self.single_runlen_1))
        self.single_runlen_2 = torch.where(single_only_2, self.single_runlen_2 + 1, torch.zeros_like(self.single_runlen_2))

        self.single_runlen_max_1 = torch.maximum(self.single_runlen_max_1, self.single_runlen_1)
        self.single_runlen_max_2 = torch.maximum(self.single_runlen_max_2, self.single_runlen_2)

        # decay_by_skill_1 = torch.clamp(torch.exp(-0.002 * self.single_skill_global_1), min=0.05)
        # decay_by_skill_2 = torch.clamp(torch.exp(-0.002 * self.single_skill_global_2), min=0.05)
        decay_by_skill_1 = torch.exp(0.002 * self.single_skill_global_1) - 1
        decay_by_skill_2 = torch.exp(0.002 * self.single_skill_global_2) - 1

        goal_reward_1 = is_goal_1 * decay_by_skill_1
        goal_reward_2 = is_goal_2 * decay_by_skill_2
        
        new_goal_mask_1 = is_goal_1 & (self.goal_1 == -1)
        self.goal_1[new_goal_mask_1] = self.episode_length_buf[new_goal_mask_1]
        success_goal_mask_1 = (
            is_goal_1 &
            (self.goal_1 != -1) &
            ((self.episode_length_buf - self.goal_1) >= 8) &
            self.log_1
        )

        new_goal_mask_2 = is_goal_2 & (self.goal_2 == -1)
        self.goal_2[new_goal_mask_2] = self.episode_length_buf[new_goal_mask_2]
        success_goal_mask_2 = (
            is_goal_2 &
            (self.goal_2 != -1) &
            ((self.episode_length_buf - self.goal_2) >= 8) &
            self.log_2
        )

        both_goal = is_goal_1 & is_goal_2
        dt = torch.abs(self.goal_1 - self.goal_2)
        latest_goal_step = torch.max(self.goal_1, self.goal_2)
        just_synced = both_goal & (dt <= 4) & (latest_goal_step == self.episode_length_buf)

        pay_mask = just_synced & (~self.sync_awarded)
        self.sync_awarded[pay_mask] = True
        self.sync_award_step[pay_mask] = self.episode_length_buf[pay_mask]
        self.sync_window_open[pay_mask] = True

        within_window = self.sync_awarded & self.sync_window_open & ((self.episode_length_buf - self.sync_award_step) < 4)
        broken = within_window & (~both_goal)
        self.sync_window_open[broken] = False
        active_mask = within_window & both_goal
        sync_reward = active_mask.float()
        
        success_mask = (
            (latest_goal_step != -1) &
            ((self.episode_length_buf - latest_goal_step) >= 4) &
            (dt <= 4) &
            (both_goal) &
            self.log
        )

        self.success_time_1[success_goal_mask_1] = self.episode_length_buf[success_goal_mask_1]
        self.success_time_2[success_goal_mask_2] = self.episode_length_buf[success_goal_mask_2]
        self.success_time[success_mask] = self.episode_length_buf[success_mask]
        
        self.log_1[success_goal_mask_1] = False
        self.log_2[success_goal_mask_2] = False
        self.log[success_mask] = False

        task_id_1 = self.task_id_1.squeeze(-1)
        task_0_1 = (task_id_1 == 0)
        task_1_1 = (task_id_1 == 1)   
        task_2_1 = (task_id_1 == 2)

        task_id_2 = self.task_id_2.squeeze(-1)
        task_0_2 = (task_id_2 == 0)
        task_1_2 = (task_id_2 == 1)   
        task_2_2 = (task_id_2 == 2)

        actions_penalty_1 = self.action_rate_l2_ratio(self.curr_actions_1, self.prev_actions_1)
        actions_penalty_2 = self.action_rate_l2_ratio(self.curr_actions_2, self.prev_actions_2)
        joint_penalty_1 = torch.norm(joint_pos_1 - self._robot_1.data.default_joint_pos[:, self.joint_pos_ids_1], dim=-1).squeeze()
        joint_penalty_2 = torch.norm(joint_pos_2 - self._robot_2.data.default_joint_pos[:, self.joint_pos_ids_2], dim=-1).squeeze()
        joint_1_penalty_1 = torch.abs(torch.atan2(camera_to_object_pos_1[:, 1], camera_to_object_pos_1[:, 0]))
        joint_1_penalty_2 = torch.abs(torch.atan2(camera_to_object_pos_2[:, 1], camera_to_object_pos_2[:, 0]))

        contact_base_penalty_1 = torch.norm(contact_base_1, dim=-1).squeeze() > 1.0
        self_collision_penalty_1 = contact_base_penalty_1.any(dim=-1)
        
        contact_base_penalty_2 = torch.norm(contact_base_2, dim=-1).squeeze() > 1.0
        self_collision_penalty_2 = contact_base_penalty_2.any(dim=-1)
        
        contact_goal_1 = torch.norm(contact_robot_goal_1, dim=-1).squeeze() > 1.0
        contact_goal_penalty_1 = contact_goal_1.any(dim=-1)
        
        contact_goal_2 = torch.norm(contact_robot_goal_2, dim=-1).squeeze() > 1.0
        contact_goal_penalty_2 = contact_goal_2.any(dim=-1)

        on_goal_1 = contact_object_goal_1.select(dim=-1, index=2).squeeze() > 0.4
        on_goal_2 = contact_object_goal_2.select(dim=-1, index=2).squeeze() > 0.4

        if on_goal_1.ndim == 0:
            on_goal_1 = on_goal_1.unsqueeze(-1)

        if on_goal_2.ndim == 0:
            on_goal_2 = on_goal_2.unsqueeze(-1)

        reward_1 = (
            self_collision_penalty_scale * self_collision_penalty_1
            + contact_goal_penalty_scale * contact_goal_penalty_1
            + action_penalty_scale * actions_penalty_1
            + joint_1_penalty_scale * joint_1_penalty_1
        )
        reward_1[task_0_1] = 0
        reward_1[task_1_1] = 0

        gripper_open_penalty_1 = ((self.curr_actions_1[:, len(self.arm_ids_1)] >= 0) * (cube_pos_1[:, 2] >= 0.068)).float()
        gripper_open_penalty_2 = ((self.curr_actions_2[:, len(self.arm_ids_2)] >= 0) * (cube_pos_2[:, 2] >= 0.068)).float()

        if self.num_envs != 1:
            reward_1[task_2_1 & on_goal_1] += (
                touch_penalty_scale * catch_object_1[task_2_1 & on_goal_1]
                # touch_penalty_scale * catch_object_1 * task_2_1 * on_goal_1
                # + touch_penalty_scale * joint_penalty_1[task_2_1 & on_goal_1]
            )

        reward_1[task_2_1 & ~on_goal_1] += (
            # dist_g_penalty_scale * torch.ones(self.num_envs, device=self.device)[task_2_1 & ~on_goal_1]
            dist_g_penalty_scale * dis_g_penalty_1[task_2_1 & ~on_goal_1]
            + dist_g_penalty_scale * gripper_open_penalty_1[task_2_1 & ~on_goal_1]
            # + 0.1 * dist_g_penalty_scale * (contact_object_goal_1.select(dim=-1, index=2).squeeze() > 0.4)[task_2_1 & ~on_goal_1]        
        )
        reward_1[task_2_1] += (
            # + goal_reward_scale * goal_reward_1[task_2_1]
            + sync_reward_scale * sync_reward[task_2_1]
        )

        reward_2 = (
            self_collision_penalty_scale * self_collision_penalty_2
            + contact_goal_penalty_scale * contact_goal_penalty_2
            + action_penalty_scale * actions_penalty_2
            + joint_1_penalty_scale * joint_1_penalty_2
        )
        reward_2[task_0_2] = 0
        reward_2[task_1_2] = 0

        if self.num_envs != 1:
            reward_2[task_2_2 & on_goal_2] += (
                touch_penalty_scale * catch_object_2[task_2_2 & on_goal_2]
                # touch_penalty_scale * catch_object_2 * task_2_2 * on_goal_2            
                # + touch_penalty_scale * joint_penalty_2[task_2_2 & on_goal_2]
            )
            
        reward_2[task_2_2 & ~on_goal_2] += (
            # dist_g_penalty_scale * torch.ones(self.num_envs, device=self.device)[task_2_2 & ~on_goal_2]
            dist_g_penalty_scale * dis_g_penalty_2[task_2_2 & ~on_goal_2]
            + dist_g_penalty_scale * gripper_open_penalty_2[task_2_2 & ~on_goal_2]
            # + 0.1 * dist_g_penalty_scale * (contact_object_goal_2.select(dim=-1, index=2).squeeze() > 0.4)[task_2_2 & ~on_goal_2]
        )
        reward_2[task_2_2] += (
            # + goal_reward_scale * goal_reward_2[task_2_2]
            + sync_reward_scale * sync_reward[task_2_2]
        )
        
        # print(joint_penalty_1, joint_penalty_2)

        self.task_id_1[success_lift_mask_1 & task_0_1] = 1
        self.task_id_1[success_reach_mask_1] = 2
        self.task_id_1[failed_reach_mask_1 & task_2_1] = 1
        self.task_id_1[failed_lift_mask_1] = 0

        self.task_id_2[success_lift_mask_2 & task_0_2] = 1
        self.task_id_2[success_reach_mask_2] = 2
        self.task_id_2[failed_reach_mask_2 & task_2_2] = 1
        self.task_id_2[failed_lift_mask_2] = 0

        self.reward_1 += reward_1
        self.reward_2 += reward_2

        rewards = {
            "robot_1": reward_1,
            "robot_2": reward_2,
        }

        return rewards

    def action_rate_l2_ratio(self, curr_actions, prev_actions) -> torch.Tensor:
        rate_of_change = (curr_actions - prev_actions) / 2.0       
        return torch.mean(torch.abs(rate_of_change), dim=1)
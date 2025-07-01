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
from isaaclab.utils.math import sample_uniform
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.utils.math import quat_conjugate, quat_from_angle_axis, quat_mul, sample_uniform, saturate
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg, FrameTransformer
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sensors import TiledCamera, TiledCameraCfg
from isaaclab.sensors import ContactSensor, ContactSensorCfg


ASSET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))

@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

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

    reset_leftwheel_friction_1 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot_1", body_names="wheel_left_link"),
          "static_friction_range": (0.4, 0.6),
          "dynamic_friction_range": (0.3, 0.4),
          "restitution_range": (0.0, 0.0),
          "num_buckets": 1,
      },
    )
    
    reset_rightwheel_friction_1 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot_1", body_names="wheel_right_link"),
          "static_friction_range": (0.4, 0.6),
          "dynamic_friction_range": (0.3, 0.4),
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

    reset_leftwheel_friction_2 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot_2", body_names="wheel_left_link"),
          "static_friction_range": (0.4, 0.6),
          "dynamic_friction_range": (0.3, 0.4),
          "restitution_range": (0.0, 0.0),
          "num_buckets": 1,
      },
    )
    
    reset_rightwheel_friction_2 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("robot_2", body_names="wheel_right_link"),
          "static_friction_range": (0.4, 0.6),
          "dynamic_friction_range": (0.3, 0.4),
          "restitution_range": (0.0, 0.0),
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
          "static_friction_range": (0.7, 0.9),
          "dynamic_friction_range": (0.6, 0.8),
          "restitution_range": (0.5, 0.5),
          "num_buckets": 1,
      },
    )

    randomize_object_friction_2 = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        params={
          "asset_cfg": SceneEntityCfg("cube_2", body_names="object_2"),
          "static_friction_range": (0.7, 0.9),
          "dynamic_friction_range": (0.6, 0.8),
          "restitution_range": (0.5, 0.5),
          "num_buckets": 1,
      },
    )

    randomize_object_color_1 = EventTerm(
        func=mdp.randomize_visual_color,
        mode="interval",
        interval_range_s=(2.5, 4.0),
        is_global_time=True,
        params={
            "event_name": "randomize_object_color_1",
            "asset_cfg": SceneEntityCfg("cube_1", body_names="object_1"),
            "colors":  {"r": (0.3, 0.4), "g": (0.0, 0.1), "b": (0.0, 0.1)},
        }
    )

    randomize_object_color_2 = EventTerm(
        func=mdp.randomize_visual_color,
        mode="interval",
        interval_range_s=(2.5, 4.0),
        is_global_time=True,
        params={
            "event_name": "randomize_object_color_2",
            "asset_cfg": SceneEntityCfg("cube_2", body_names="object_2"),
            "colors":  {"r": (0.3, 0.4), "g": (0.0, 0.1), "b": (0.0, 0.1)},
        }
    )
    
    randomize_goal_base_color = EventTerm(
        func=mdp.randomize_visual_color,
        mode="interval",
        interval_range_s=(2.5, 4.0),
        is_global_time=True,
        params={
            "event_name": "randomize_goal_base_color",
            "asset_cfg": SceneEntityCfg("goal_base", body_names="goal_base"),
            "colors":  {"r": (0.0, 0.1), "g": (0.0, 0.1), "b": (0.3, 0.4)},
            }
    )

    randomize_goal_color = EventTerm(
        func=mdp.randomize_visual_color,
        mode="interval",
        interval_range_s=(2.5, 4.0),
        is_global_time=True,
        params={
            "event_name": "randomize_goal_color",
            "asset_cfg": SceneEntityCfg("goal", body_names="goal"),
            "colors":  {"r": (0.0, 0.1), "g": (0.0, 0.1), "b": (0.3, 0.4)},
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
class Turtlebot3MultiImageEnvCfg(DirectMARLEnvCfg):
    # env
    episode_length_s = 50.1  # 1500 timesteps
    decimation = 25
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
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=256, env_spacing=100, replicate_physics=False)

    # robot
    robot_1 = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot_1",
        spawn=sim_utils.UsdFileCfg(
            usd_path=os.path.join(ASSET_ROOT, "isaaclab_assets/data/Robots/Turtlebot3_manipulation/turtlebot3_manipulation_nolidar_collision_00.usd"),
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
            pos=(0.0, 0.7, 0.0),
            rot=(0.70710678, 0.0, 0.0, -0.70710678),
        ),
        actuators={
            "turtlebot3_arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-4]"],
                effort_limit_sim=4.1,
                velocity_limit_sim=0.2,
                stiffness=80.0,
                damping=4.0,
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

    camera_1: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot_1/base_footprint/front_cam",
        update_period=0.03,
        offset=TiledCameraCfg.OffsetCfg(pos=(0.076, 0.068, 0.041), rot=(0.99756405, 0.0, 0.06975647, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=0.304, focus_distance=0.927, horizontal_aperture=0.45, vertical_aperture=4.5, clipping_range=(0.01, 20.0)
        ),
        width=120,
        height=120,
    )

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
            usd_path=os.path.join(ASSET_ROOT, "isaaclab_assets/data/Robots/Turtlebot3_manipulation/turtlebot3_manipulation_nolidar_collision_00.usd"),
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
            pos=(0.0, -0.7, 0.0),
            rot=(0.70710678, 0.0, 0.0, 0.70710678),
        ),
        actuators={
            "turtlebot3_arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-4]"],
                effort_limit_sim=4.1,
                velocity_limit_sim=0.2,
                stiffness=80.0,
                damping=4.0,
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

    camera_2: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot_2/base_footprint/front_cam",
        update_period=0.03,
        offset=TiledCameraCfg.OffsetCfg(pos=(0.076, 0.068, 0.041), rot=(0.99756405, 0.0, 0.06975647, 0.0), convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=0.304, focus_distance=0.927, horizontal_aperture=0.45, vertical_aperture=4.5, clipping_range=(0.01, 20.0)
        ),
        width=120,
        height=120,
    )

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
            random_choice=True,
            activate_contact_sensors=True,
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

    contact_gripper_object_1: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/object_1",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot_1/gripper_left_link",
            "/World/envs/env_.*/Robot_1/gripper_right_link",
            ],
    )

    cube_2: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object_2",
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
            random_choice=True,
            activate_contact_sensors=True,
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

    contact_gripper_object_2: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/object_2",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=[
            "/World/envs/env_.*/Robot_2/gripper_left_link",
            "/World/envs/env_.*/Robot_2/gripper_right_link",
            ],
    )

    goal_base: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal_base",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(0.04, 0.04, 0.03),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.05, 0.05, 0.03),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.06, 0.06, 0.03),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.07, 0.07, 0.03),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
                sim_utils.CuboidCfg(
                    size=(0.08, 0.08, 0.03),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
            ],
            random_choice=True,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        solver_position_iteration_count=16,
                        solver_velocity_iteration_count=1,
                        max_angular_velocity=1000.0,
                        max_linear_velocity=1000.0,
                        max_depenetration_velocity=5.0,
                        disable_gravity=False,
                    ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
    )

    goal: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/goal",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(0.03, 0.3, 0.03),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
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
            mass_props=sim_utils.MassPropertiesCfg(mass=0.8),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.045), rot=(1.0, 0.0, 0.0, 0.0)),
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

    # ground plane
    ground_plane: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ground_color",
        spawn=sim_utils.MultiAssetSpawnerCfg(
            assets_cfg=[
                sim_utils.CuboidCfg(
                    size=(3.5, 3.5, 0.01),
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
                sim_utils.CuboidCfg(
                    size=(15.0, 15.0, 0.01),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0), metallic=0.2),
                ),
            ],
            random_choice=True,
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

    contact_leftgripper_ground_1: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_1/gripper_left_link",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=["/World/ground"],
    )

    contact_rightgripper_ground_1: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_1/gripper_right_link",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=["/World/ground"],
    )

    contact_leftgripper_ground_2: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_2/gripper_left_link",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=["/World/ground"],
    )

    contact_rightgripper_ground_2: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot_2/gripper_right_link",
        update_period=0.0,
        history_length=0,
        filter_prim_paths_expr=["/World/ground"],
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
                    pos=[0.045, 0.0, 0.0],
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
                    pos=[0.045, 0.0, 0.0],
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
                    pos=[0.045, 0.0, 0.0],
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
                    pos=[0.045, 0.0, 0.0],
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
                name="robot_1_goal",
                offset=OffsetCfg(
                    pos=[0.0, 0.13, 0.045],
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
                name="robot_2_goal",
                offset=OffsetCfg(
                    pos=[0.0, -0.13, 0.045],
                ),
            ),
        ],
    )

    events: EventCfg = EventCfg()

    possible_agents = ["robot_1", "robot_2"]
    action_spaces = {"robot_1": 7, "robot_2": 7}
    observation_spaces = {
        "robot_1": {"joint": 6, "rgb": [camera_1.height, camera_1.width, 3], "object": 7, "actions": 7},
        "robot_2": {"joint": 6, "rgb": [camera_2.height, camera_2.width, 3], "object": 7, "actions": 7},
        }
    state_space = -1

    action_scale = 1.0
    dof_velocity_scale = 0.1

    # reward scales
    dist_reward_scale = 1.0
    lift_reward_scale = 1.0
    dist_g_reward_scale = 2.0
    action_penalty_scale = -0.15
    self_collision_penalty_scale = -0.5
    contact_ground_penalty_scale = -0.05
    contact_goal_penalty_scale = -0.25

class Turtlebot3MultiImageEnv(DirectMARLEnv):
    # pre-physics step calls
    #   |-- _pre_physics_step(action)
    #   |-- _apply_action()
    # post-physics step calls
    #   |-- _get_dones()
    #   |-- _get_rewards()
    #   |-- _reset_idx(env_ids)
    #   |-- _get_observations()

    cfg: Turtlebot3MultiImageEnvCfg

    def __init__(self, cfg: Turtlebot3MultiImageEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.dt = self.cfg.sim.dt * self.cfg.decimation

        self.robot_dof_lower_limits = self._robot_1.data.soft_joint_pos_limits[0, :, 0].to(device=self.device)
        self.robot_dof_upper_limits = self._robot_1.data.soft_joint_pos_limits[0, :, 1].to(device=self.device)
        self.robot_dof_vel_limits_tensor = self._robot_1.data.joint_velocity_limits[0, :].to(device=self.device)

        self.joint_pos_names = ["joint.*", "gripper_.*"]
        self.arm_names = ["joint.*"]
        self.gripper_names = ["gripper_.*"]
        self.wheel_names = ["wheel_.*"]
        self.joint_pos_ids, _ = self._robot_1.find_joints(self.joint_pos_names, preserve_order=False)
        self.arm_ids, _ = self._robot_1.find_joints(self.arm_names, preserve_order=False)
        self.gripper_ids, _ = self._robot_1.find_joints(self.gripper_names, preserve_order=False)
        self.wheel_ids, _ = self._robot_1.find_joints(self.wheel_names, preserve_order=False)

        self.robot_arm_targets_1 = torch.zeros((self.num_envs, len(self.arm_ids)), device=self.device)
        self.robot_gripper_targets_1 = torch.zeros((self.num_envs, len(self.gripper_ids)), device=self.device)
        self.robot_wheel_targets_1 = torch.zeros((self.num_envs, len(self.wheel_ids)), device=self.device)

        self.robot_arm_targets_2 = torch.zeros((self.num_envs, len(self.arm_ids)), device=self.device)
        self.robot_gripper_targets_2 = torch.zeros((self.num_envs, len(self.gripper_ids)), device=self.device)
        self.robot_wheel_targets_2 = torch.zeros((self.num_envs, len(self.wheel_ids)), device=self.device)

        self.curr_actions_1 = torch.zeros((self.num_envs, self.cfg.action_spaces["robot_1"]), device=self.device)
        self.prev_actions_1 = torch.zeros((self.num_envs, self.cfg.action_spaces["robot_1"]), device=self.device)

        self.curr_actions_2 = torch.zeros((self.num_envs, self.cfg.action_spaces["robot_2"]), device=self.device)
        self.prev_actions_2 = torch.zeros((self.num_envs, self.cfg.action_spaces["robot_2"]), device=self.device)

    def _setup_scene(self):
        self._robot_1 = Articulation(self.cfg.robot_1)
        self._camera_1 = TiledCamera(self.cfg.camera_1)
        self._contact_base_1 = ContactSensor(self.cfg.contact_base_1)
        self._contact_gripper_object_1 = ContactSensor(self.cfg.contact_gripper_object_1)
        self._contact_robot_goal_1 = ContactSensor(self.cfg.contact_robot_goal_1)
        self._contact_leftgripper_ground_1 = ContactSensor(self.cfg.contact_leftgripper_ground_1)
        self._contact_rightgripper_ground_1 = ContactSensor(self.cfg.contact_rightgripper_ground_1)
        self._ee_frame_1 = FrameTransformer(self.cfg.ee_frame_1)
        self._lee_frame_1 = FrameTransformer(self.cfg.lee_frame_1)
        self._ree_frame_1 = FrameTransformer(self.cfg.ree_frame_1)

        self._robot_2 = Articulation(self.cfg.robot_2)
        self._camera_2 = TiledCamera(self.cfg.camera_2)
        self._contact_base_2 = ContactSensor(self.cfg.contact_base_2)
        self._contact_gripper_object_2 = ContactSensor(self.cfg.contact_gripper_object_2)
        self._contact_robot_goal_2 = ContactSensor(self.cfg.contact_robot_goal_2)
        self._contact_leftgripper_ground_2 = ContactSensor(self.cfg.contact_leftgripper_ground_2)
        self._contact_rightgripper_ground_2 = ContactSensor(self.cfg.contact_rightgripper_ground_2)
        self._ee_frame_2 = FrameTransformer(self.cfg.ee_frame_2)
        self._lee_frame_2 = FrameTransformer(self.cfg.lee_frame_2)
        self._ree_frame_2 = FrameTransformer(self.cfg.ree_frame_2)

        self._cube_1 = RigidObject(self.cfg.cube_1)
        self._cube_2 = RigidObject(self.cfg.cube_2)

        self._ground_plane = RigidObject(self.cfg.ground_plane)

        self._goal_base = RigidObject(self.cfg.goal_base)
        self._goal = RigidObject(self.cfg.goal)
        self._goal_frame_1 = FrameTransformer(self.cfg.goal_frame_1)
        self._goal_frame_2 = FrameTransformer(self.cfg.goal_frame_2)

        # self.goal_markers = VisualizationMarkers(self.cfg.goal)

        # ロボットをシーンに追加
        self.scene.articulations["robot_1"] = self._robot_1
        self.scene.sensors["camera_1"] = self._camera_1
        self.scene.sensors["contact_base_1"] = self._contact_base_1
        self.scene.sensors["contact_gripper_object_1"] = self._contact_gripper_object_1
        self.scene.sensors["contact_robot_goal_1"] = self._contact_robot_goal_1
        self.scene.sensors["contact_leftgripper_ground_1"] = self._contact_leftgripper_ground_1
        self.scene.sensors["contact_rightgripper_ground_1"] = self._contact_rightgripper_ground_1
        self.scene.sensors["ee_frame_1"] = self._ee_frame_1
        self.scene.sensors["lee_frame_1"] = self._lee_frame_1
        self.scene.sensors["ree_frame_1"] = self._ree_frame_1

        self.scene.articulations["robot_2"] = self._robot_2
        self.scene.sensors["camera_2"] = self._camera_2
        self.scene.sensors["contact_base_2"] = self._contact_base_2
        self.scene.sensors["contact_gripper_object_2"] = self._contact_gripper_object_2
        self.scene.sensors["contact_robot_goal_2"] = self._contact_robot_goal_2        
        self.scene.sensors["contact_leftgripper_ground_2"] = self._contact_leftgripper_ground_2
        self.scene.sensors["contact_rightgripper_ground_2"] = self._contact_rightgripper_ground_2
        self.scene.sensors["ee_frame_2"] = self._ee_frame_2
        self.scene.sensors["lee_frame_2"] = self._lee_frame_2
        self.scene.sensors["ree_frame_2"] = self._ree_frame_2

        self.scene.rigid_objects["cube_1"] = self._cube_1
        self.scene.rigid_objects["cube_2"] = self._cube_2

        self.scene.rigid_objects["ground_plane"] = self._ground_plane

        self.scene.rigid_objects["goal_base"] = self._goal_base
        self.scene.rigid_objects["goal"] = self._goal
        self.scene.sensors["goal_frame_1"] = self._goal_frame_1
        self.scene.sensors["goal_frame_2"] = self._goal_frame_2
    
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        omni.kit.commands.execute(
            "ToggleVisibilitySelectedPrims",
            selected_paths=["/World/ground"]   # ←ステージ上の床 Prim のパス
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        arm_actions_1 = actions["robot_1"][:, :len(self.arm_ids)].clone().clamp(-1.0, 1.0)
        gripper_action_1 = actions["robot_1"][:, len(self.arm_ids)].clone().clamp(-1.0, 1.0) 
        wheel_actions_1 = actions["robot_1"][:, len(self.arm_ids)+1:].clone().clamp(-1.0, 1.0)

        arm_actions_2 = actions["robot_2"][:, :len(self.arm_ids)].clone().clamp(-1.0, 1.0)
        gripper_action_2 = actions["robot_2"][:, len(self.arm_ids)].clone().clamp(-1.0, 1.0) 
        wheel_actions_2 = actions["robot_2"][:, len(self.arm_ids)+1:].clone().clamp(-1.0, 1.0)

        arm_targets_1 = self._robot_1.data.joint_pos[:, self.arm_ids] + self.robot_dof_vel_limits_tensor[self.arm_ids] * self.dt * arm_actions_1
        self.robot_arm_targets_1[:] = torch.clamp(arm_targets_1, self.robot_dof_lower_limits[self.arm_ids], self.robot_dof_upper_limits[self.arm_ids])

        arm_targets_2 = self._robot_2.data.joint_pos[:, self.arm_ids] + self.robot_dof_vel_limits_tensor[self.arm_ids] * self.dt * arm_actions_2
        self.robot_arm_targets_2[:] = torch.clamp(arm_targets_2, self.robot_dof_lower_limits[self.arm_ids], self.robot_dof_upper_limits[self.arm_ids])

        gripper_actions_1 = torch.zeros(self.num_envs, len(self.gripper_ids), device=self.device)
        gripper_actions_1[:, 0] = torch.where(gripper_action_1 >= 0.0, self.robot_dof_upper_limits[self.gripper_ids[0]].item(),
                                      self.robot_dof_lower_limits[self.gripper_ids[0]].item())
        gripper_actions_1[:, 1] = torch.where(gripper_action_1 >= 0.0, self.robot_dof_upper_limits[self.gripper_ids[1]].item(),
                                      self.robot_dof_lower_limits[self.gripper_ids[1]].item())
        self.robot_gripper_targets_1[:] = gripper_actions_1

        gripper_actions_2 = torch.zeros(self.num_envs, len(self.gripper_ids), device=self.device)
        gripper_actions_2[:, 0] = torch.where(gripper_action_2 >= 0.0, self.robot_dof_upper_limits[self.gripper_ids[0]].item(),
                                      self.robot_dof_lower_limits[self.gripper_ids[0]].item())
        gripper_actions_2[:, 1] = torch.where(gripper_action_2 >= 0.0, self.robot_dof_upper_limits[self.gripper_ids[1]].item(),
                                      self.robot_dof_lower_limits[self.gripper_ids[1]].item())
        self.robot_gripper_targets_2[:] = gripper_actions_2

        # wheel_actions_1[:, 0] = torch.full_like(wheel_actions_1[:, 0], -1.0)
        # wheel_actions_1[:, 1] = torch.full_like(wheel_actions_1[:, 1], 1.0)
        
        self.robot_wheel_targets_1[:] = wheel_actions_1 * self.robot_dof_vel_limits_tensor[self.wheel_ids]
        self.robot_wheel_targets_2[:] = wheel_actions_2 * self.robot_dof_vel_limits_tensor[self.wheel_ids]

        self.prev_actions_1 = self.curr_actions_1
        self.prev_actions_2 = self.curr_actions_2

        self.curr_actions_1 = actions["robot_1"]
        self.curr_actions_2 = actions["robot_2"]

    def _apply_action(self):
        # 制御
        self._robot_1.set_joint_position_target(self.robot_arm_targets_1, self.arm_ids)
        self._robot_1.set_joint_position_target(self.robot_gripper_targets_1, self.gripper_ids)
        self._robot_1.set_joint_velocity_target(self.robot_wheel_targets_1, self.wheel_ids)

        self._robot_2.set_joint_position_target(self.robot_arm_targets_2, self.arm_ids)
        self._robot_2.set_joint_position_target(self.robot_gripper_targets_2, self.gripper_ids)
        self._robot_2.set_joint_velocity_target(self.robot_wheel_targets_2, self.wheel_ids)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        out_of_ = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = {agent: out_of_ for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        return self._compute_rewards(
            self.joint_acc_1,
            self.joint_acc_2,
            self.ee_pos_1,
            self.lee_pos_1,
            self.ree_pos_1,
            self.ee_pos_2,
            self.lee_pos_2,
            self.ree_pos_2,
            self.cube_pos_1,
            self.cube_pos_2,
            self.goal_pos_1,
            self.goal_pos_2,
            self.contact_base_1,
            self.contact_gripper_object_1,
            self.contact_robot_goal_1,
            self.contact_leftgripper_ground_1,
            self.contact_rightgripper_ground_1,
            self.contact_base_2,
            self.contact_gripper_object_2,
            self.contact_robot_goal_2,
            self.contact_leftgripper_ground_2,
            self.contact_rightgripper_ground_2,
            self.cfg.dist_reward_scale,
            self.cfg.lift_reward_scale,
            self.cfg.dist_g_reward_scale,
            self.cfg.action_penalty_scale,
            self.cfg.self_collision_penalty_scale,
            self.cfg.contact_ground_penalty_scale,
            self.cfg.contact_goal_penalty_scale,
        )

    def _reset_idx(self, env_ids: torch.Tensor | None):
        super()._reset_idx(env_ids)

        joint_pos_1 = self._robot_1.data.default_joint_pos[env_ids] + sample_uniform(
            -0.5,
            0.5,
            (len(env_ids), self._robot_1.num_joints),
            self.device,
        )
        joint_pos_1 = torch.clamp(joint_pos_1, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        # joint_pos_1 = torch.clamp(self._robot_1.data.default_joint_pos[env_ids], self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel_1 = torch.zeros_like(joint_pos_1)
        default_robot_state_1 = self._robot_1.data.default_root_state[env_ids].clone()
        default_robot_state_1[:, :3] += self.scene.env_origins[env_ids]
        self._robot_1.write_root_link_pose_to_sim(default_robot_state_1[:, :7], env_ids=env_ids)
        self._robot_1.write_root_com_velocity_to_sim(default_robot_state_1[:, 7:], env_ids=env_ids)
        self._robot_1.set_joint_position_target(joint_pos_1, env_ids=env_ids)
        self._robot_1.write_joint_state_to_sim(joint_pos_1, joint_vel_1, env_ids=env_ids)

        joint_pos_2 = self._robot_2.data.default_joint_pos[env_ids] + sample_uniform(
            -0.5,
            0.5,
            (len(env_ids), self._robot_2.num_joints),
            self.device,
        )
        joint_pos_2 = torch.clamp(joint_pos_2, self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        # joint_pos_2 = torch.clamp(self._robot_2.data.default_joint_pos[env_ids], self.robot_dof_lower_limits, self.robot_dof_upper_limits)
        joint_vel_2 = torch.zeros_like(joint_pos_2)
        default_robot_state_2 = self._robot_2.data.default_root_state[env_ids].clone()
        default_robot_state_2[:, :3] += self.scene.env_origins[env_ids]
        self._robot_2.write_root_link_pose_to_sim(default_robot_state_2[:, :7], env_ids=env_ids)
        self._robot_2.write_root_com_velocity_to_sim(default_robot_state_2[:, 7:], env_ids=env_ids)
        self._robot_2.set_joint_position_target(joint_pos_2, env_ids=env_ids)
        self._robot_2.write_joint_state_to_sim(joint_pos_2, joint_vel_2, env_ids=env_ids)

        default_goal_base_state = self._goal_base.data.default_root_state[env_ids, :7].clone()
        default_goal_base_state[:, :3] += self.scene.env_origins[env_ids]
        default_goal_base_state[:, :3] += self.reset_root_state_uniform(
            env_ids=env_ids,
            pose_range = {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (0.001, 0.001)},
        )
        default_goal_state = self._goal.data.default_root_state[env_ids, :7].clone()
        default_goal_state[:, :2] = default_goal_base_state[:, :2]
        self._goal_base.write_root_link_pose_to_sim(default_goal_base_state)
        self._goal.write_root_link_pose_to_sim(default_goal_state)

        default_cube_state_1 = self._cube_1.data.default_root_state[env_ids].clone()
        default_cube_state_1[:, :3] = self._goal_frame_1.data.target_pos_w[env_ids, 0, :]
        self._cube_1.write_root_link_pose_to_sim(default_cube_state_1[:, :7], env_ids=env_ids)

        default_cube_state_2 = self._cube_2.data.default_root_state[env_ids].clone()
        default_cube_state_2[:, :3] = self._goal_frame_2.data.target_pos_w[env_ids, 0, :]
        self._cube_2.write_root_link_pose_to_sim(default_cube_state_2[:, :7], env_ids=env_ids)

        self._compute_intermediate_values(env_ids)
    
    def reset_root_state_uniform(self, env_ids, pose_range):
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)

        positions = rand_samples[:, 0:3]

        return positions

    def _get_observations(self) -> dict:
        rgb_1 = self._camera_1.data.output["rgb"] / 255.0
        rgb_2 = self._camera_2.data.output["rgb"] / 255.0

        obs = {
            "robot_1": {
                "joint": self.joint_pos_1,
                "rgb": rgb_1,
                "object": self.object_b_1,
                "actions": self.curr_actions_1,
            },
            "robot_2": {
                "joint": self.joint_pos_2,
                "rgb": rgb_2,
                "object": self.object_b_2,
                "actions": self.curr_actions_2,
            },
        }

        return obs
    
    def _get_states(self) -> dict:

        state = torch.cat(
            (
               self.joint_pos_1,
               self.object_b_1,
               self.curr_actions_1,
               self.joint_pos_2,
               self.object_b_2,
               self.curr_actions_2 
            ),
            dim=-1
        )
        
        return state

    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot_1._ALL_INDICES
    
        self.joint_pos_1 = self._robot_1.data.joint_pos[:, self.joint_pos_ids]
        self.joint_pos_2 = self._robot_2.data.joint_pos[:, self.joint_pos_ids]
        self.joint_acc_1 = self._robot_1.data.joint_acc[env_ids][:, self.joint_pos_ids]
        self.joint_acc_2 = self._robot_2.data.joint_acc[env_ids][:, self.joint_pos_ids]

        self.base_pos_1 = self._robot_1.data.root_link_state_w[env_ids, :3]
        self.base_rot_1 = self._robot_1.data.root_link_state_w[env_ids, 3:7]
        self.ee_pos_1 = self._ee_frame_1.data.target_pos_w[env_ids, 0, :]
        self.lee_pos_1 = self._lee_frame_1.data.target_pos_w[env_ids, 0, :]
        self.ree_pos_1 = self._ree_frame_1.data.target_pos_w[env_ids, 0, :]

        self.base_pos_2 = self._robot_2.data.root_link_state_w[env_ids, :3]
        self.base_rot_2 = self._robot_2.data.root_link_state_w[env_ids, 3:7]
        self.ee_pos_2 = self._ee_frame_2.data.target_pos_w[env_ids, 0, :]
        self.lee_pos_2 = self._lee_frame_2.data.target_pos_w[env_ids, 0, :]
        self.ree_pos_2 = self._ree_frame_2.data.target_pos_w[env_ids, 0, :]

        self.cube_pos_1 = self._cube_1.data.root_link_state_w[env_ids, :3]
        self.cube_rot_1 = self._cube_1.data.root_link_state_w[env_ids, 3:7]

        self.cube_pos_2 = self._cube_2.data.root_link_state_w[env_ids, :3]
        self.cube_rot_2 = self._cube_2.data.root_link_state_w[env_ids, 3:7]

        object_pos_b_1, object_rot_b_1 = math_utils.subtract_frame_transforms(
            self.base_pos_1, self.base_rot_1, self.cube_pos_1, self.cube_rot_1
        )
        self.object_b_1 = torch.cat((object_pos_b_1, object_rot_b_1), dim=1)

        object_pos_b_2, object_rot_b_2 = math_utils.subtract_frame_transforms(
            self.base_pos_2, self.base_rot_2, self.cube_pos_2, self.cube_rot_2
        )
        self.object_b_2 = torch.cat((object_pos_b_2, object_rot_b_2), dim=1)

        self.goal_pos_1 = self._goal_frame_1.data.target_pos_w[env_ids, 0, :]
        self.goal_rot_1 = self._goal_frame_1.data.target_quat_w[env_ids, 0, :]

        self.goal_pos_2 = self._goal_frame_2.data.target_pos_w[env_ids, 0, :]
        self.goal_rot_2 = self._goal_frame_2.data.target_quat_w[env_ids, 0, :]

        self.contact_base_1 = self._contact_base_1.data.force_matrix_w[env_ids, :]
        self.contact_gripper_object_1 = self._contact_gripper_object_1.data.force_matrix_w[env_ids, :]
        self.contact_robot_goal_1 = self._contact_robot_goal_1.data.force_matrix_w[env_ids, :]
        self.contact_leftgripper_ground_1 = self._contact_leftgripper_ground_1.data.force_matrix_w[env_ids, :]
        self.contact_rightgripper_ground_1 = self._contact_rightgripper_ground_1.data.force_matrix_w[env_ids, :]

        self.contact_base_2 = self._contact_base_2.data.force_matrix_w[env_ids, :]
        self.contact_gripper_object_2 = self._contact_gripper_object_2.data.force_matrix_w[env_ids, :]
        self.contact_robot_goal_2 = self._contact_robot_goal_2.data.force_matrix_w[env_ids, :]
        self.contact_leftgripper_ground_2 = self._contact_leftgripper_ground_2.data.force_matrix_w[env_ids, :]
        self.contact_rightgripper_ground_2 = self._contact_rightgripper_ground_2.data.force_matrix_w[env_ids, :]

    def _compute_rewards(
        self,
        joint_acc_1,
        joint_acc_2,
        ee_pos_1,
        lee_pos_1,
        ree_pos_1,
        ee_pos_2,
        lee_pos_2,
        ree_pos_2,
        cube_pos_1,
        cube_pos_2,
        goal_pos_1,
        goal_pos_2,
        contact_base_1,
        contact_gripper_object_1,
        contact_robot_goal_1,
        contact_leftgripper_ground_1,
        contact_rightgripper_ground_1,
        contact_base_2,
        contact_gripper_object_2,
        contact_robot_goal_2,
        contact_leftgripper_ground_2,
        contact_rightgripper_ground_2,
        dist_reward_scale,
        lift_reward_scale,
        dist_g_reward_scale,
        action_penalty_scale,
        self_collision_penalty_scale,
        contact_ground_penalty_scale,
        contact_goal_penalty_scale,
    ):
        d_c_1 = torch.norm(cube_pos_1-ee_pos_1, dim=-1)
        d_l_1 = torch.norm(cube_pos_1-lee_pos_1, dim=-1)
        d_r_1 = torch.norm(cube_pos_1-ree_pos_1, dim=-1)
        dis_1 = (d_c_1*2 + d_l_1 + d_r_1) / 4
        dis_reward_1 = torch.exp(-10*dis_1)

        d_c_2 = torch.norm(cube_pos_2-ee_pos_2, dim=-1)
        d_l_2 = torch.norm(cube_pos_2-lee_pos_2, dim=-1)
        d_r_2 = torch.norm(cube_pos_2-ree_pos_2, dim=-1)
        dis_2 = (d_c_2*2 + d_l_2 + d_r_2) / 4
        dis_reward_2 = torch.exp(-10*dis_2)

        contact_gripper_object_reward_1 = torch.norm(contact_gripper_object_1, dim=-1).squeeze() > 1.0
        catch_object_1 = contact_gripper_object_reward_1.all(dim=-1)
        lift_reward_1 = torch.where(cube_pos_1[:, 2] > 0.09, 1.0, 0.0) * catch_object_1 * torch.where(d_c_1 < 0.015, 1.0, 0.0)
        contact_gripper_object_reward_2 = torch.norm(contact_gripper_object_2, dim=-1).squeeze() > 1.0
        catch_object_2 = contact_gripper_object_reward_2.all(dim=-1)
        lift_reward_2 = torch.where(cube_pos_2[:, 2] > 0.09, 1.0, 0.0) * catch_object_2 * torch.where(d_c_2 < 0.015, 1.0, 0.0)
        
        # dis_g = torch.norm(goal_pos-cube_pos, dim=-1)
        # dis_g_reward = torch.exp(-10*dis_g)
        # dis_g_1 = torch.norm(goal_pos_1 - cube_pos_1, dim=-1)
        # dis_g_reward_1 = (1 - torch.tanh(10*dis_g_1)) * lift_reward_1
        # dis_g_reward_1 = torch.exp(-10*dis_g_1) * lift_reward_1
        # dis_g_reward_11 = (1 - torch.tanh(3*dis_g_1)) * lift_reward_1

        # dis_g_2 = torch.norm(goal_pos_2 - cube_pos_2, dim=-1)
        # dis_g_reward_2 = (1 - torch.tanh(10*dis_g_2)) * lift_reward_2
        # dis_g_reward_2 = torch.exp(-10*dis_g_2) * lift_reward_2
        # dis_g_reward_12 = (1 - torch.tanh(3*dis_g_2)) * lift_reward_2

        actions_penalty_1 = self.action_rate_l2_ratio(self.curr_actions_1, self.prev_actions_1)
        actions_penalty_2 = self.action_rate_l2_ratio(self.curr_actions_2, self.prev_actions_2)
        # actions_penalty_1 = torch.mean(torch.abs(joint_acc_1), dim=-1)
        # actions_penalty_2 = torch.mean(torch.abs(joint_acc_2), dim=-1)

        contact_base_penalty_1 = torch.norm(contact_base_1, dim=-1).squeeze() > 1.0
        self_collision_penalty_1 = contact_base_penalty_1.any(dim=-1)
        
        contact_base_penalty_2 = torch.norm(contact_base_2, dim=-1).squeeze() > 1.0
        self_collision_penalty_2 = contact_base_penalty_2.any(dim=-1)
        
        contact_goal_1 = torch.norm(contact_robot_goal_1, dim=-1).squeeze() > 1.0
        contact_goal_penalty_1 = contact_goal_1.any(dim=-1)
        
        contact_goal_2 = torch.norm(contact_robot_goal_2, dim=-1).squeeze() > 1.0
        contact_goal_penalty_2 = contact_goal_2.any(dim=-1)

        contact_left_ground_penalty_1 = torch.norm(contact_leftgripper_ground_1, dim=-1).squeeze() > 1.0
        contact_right_ground_penalty_1 = torch.norm(contact_rightgripper_ground_1, dim=-1).squeeze() > 1.0
        contact_ground_penalty_1 = contact_left_ground_penalty_1 | contact_right_ground_penalty_1

        contact_left_ground_penalty_2 = torch.norm(contact_leftgripper_ground_2, dim=-1).squeeze() > 1.0
        contact_right_ground_penalty_2 = torch.norm(contact_rightgripper_ground_2, dim=-1).squeeze() > 1.0
        contact_ground_penalty_2 = contact_left_ground_penalty_2 | contact_right_ground_penalty_2
        
        rewards = {
            "robot_1": dist_reward_scale * dis_reward_1 + lift_reward_scale * lift_reward_1 + self_collision_penalty_scale * self_collision_penalty_1 + action_penalty_scale * actions_penalty_1 + contact_ground_penalty_scale * contact_ground_penalty_1 + contact_goal_penalty_scale * contact_goal_penalty_1,
            "robot_2": dist_reward_scale * dis_reward_2 + lift_reward_scale * lift_reward_2 + self_collision_penalty_scale * self_collision_penalty_2 + action_penalty_scale * actions_penalty_2 + contact_ground_penalty_scale * contact_ground_penalty_2 + contact_goal_penalty_scale * contact_goal_penalty_2,
        }

        return rewards

    def action_rate_l2_ratio(self, curr_actions, prev_actions) -> torch.Tensor:
        rate_of_change = (curr_actions - prev_actions) / 2.0       
        return torch.mean(torch.abs(rate_of_change), dim=1)
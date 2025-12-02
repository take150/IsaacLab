# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""
Franka-Cabinet environment.
"""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Turtlebot3-Reach-Direct-v0",
    entry_point=f"{__name__}.reach_env_v0:Turtlebot3ReachEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.reach_env_v0:Turtlebot3ReachEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Turtlebot3ManipulationPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_reach_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Turtlebot3-Image-Direct-v6",
    entry_point=f"{__name__}.image_env:Turtlebot3ImageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.image_env:Turtlebot3ImageEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_image_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Turtlebot3ManipulationPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_image_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Turtlebot3-Multi-Image-Direct-v0",
    entry_point=f"{__name__}.multi_image_env:Turtlebot3MultiImageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multi_image_env:Turtlebot3MultiImageEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Turtlebot3ManipulationPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_image_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_image_mappo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Turtlebot3-Multi-Place-Direct-v0",
    entry_point=f"{__name__}.multi_place_env:Turtlebot3MultiPlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multi_place_env:Turtlebot3MultiPlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Turtlebot3ManipulationPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_image_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_image_mappo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Turtlebot3-Single-Direct-v0",
    entry_point=f"{__name__}.single_env:Turtlebot3SingleEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.single_env:Turtlebot3SingleEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Turtlebot3ManipulationPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Turtlebot3-Single-Distillation-Direct-v0",
    entry_point=f"{__name__}.single_distillation_env:Turtlebot3SingleDistillationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.single_distillation_env:Turtlebot3SingleDistillationEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Turtlebot3ManipulationPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Turtlebot3-Multi-Direct-v0",
    entry_point=f"{__name__}.multi_env:Turtlebot3MultiEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multi_env:Turtlebot3MultiEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Turtlebot3ManipulationPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Turtlebot3-Multi-Distillation-Direct-v0",
    entry_point=f"{__name__}.multi_distillation_env:Turtlebot3MultiDistillationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multi_distillation_env:Turtlebot3MultiDistillationEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Turtlebot3ManipulationPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Turtlebot3-Single-Place-Direct-v0",
    entry_point=f"{__name__}.single_place_env:Turtlebot3SinglePlaceEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.single_place_env:Turtlebot3SinglePlaceEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Turtlebot3ManipulationPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

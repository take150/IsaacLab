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
    id="Isaac-Turtlebot3-Move-Direct-v0",
    entry_point=f"{__name__}.move_env_v0:Turtlebot3MoveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.move_env_v0:Turtlebot3MoveEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Turtlebot3ManipulationPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Turtlebot3-Move-Direct-v1",
    entry_point=f"{__name__}.move_env_v1:Turtlebot3MoveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.move_env_v1:Turtlebot3MoveEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Turtlebot3ManipulationPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Turtlebot3-Multi-Move-Direct-v0",
    entry_point=f"{__name__}.multi_move_env:Turtlebot3MultiMoveEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.multi_move_env:Turtlebot3MultiMoveEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Turtlebot3ManipulationPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_mappo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Turtlebot3-Image-Direct-v0",
    entry_point=f"{__name__}.image_env_v0:Turtlebot3ImageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.image_env_v0:Turtlebot3ImageEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_image_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Turtlebot3ManipulationPPORunnerCfg",
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_image_ppo_cfg.yaml",
    },
)

gym.register(
    id="Isaac-Turtlebot3-Image-Direct-v1",
    entry_point=f"{__name__}.image_env_v1:Turtlebot3ImageEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.image_env_v1:Turtlebot3ImageEnvCfg",
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
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "skrl_ippo_cfg_entry_point": f"{agents.__name__}:skrl_ippo_cfg.yaml",
        "skrl_mappo_cfg_entry_point": f"{agents.__name__}:skrl_image_mappo_cfg.yaml",
    },
)

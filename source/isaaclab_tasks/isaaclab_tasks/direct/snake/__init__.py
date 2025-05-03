import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Snake-Direct-v3_cmd-l-1-5-10_localpose",
    entry_point=f"{__name__}.snake_env:SnakeEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.snake_env:SnakeEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:SnakePPORunnerCfg", #TODO: check if this is appropriate base
        "skrl_cfg_entry_point": f"{agents.__name__}:skrl_ppo_cfg.yaml",
        "sb3_cfg_entry_point" : f"{agents.__name__}:sb3_ppo_cfg.yaml",
    },
)

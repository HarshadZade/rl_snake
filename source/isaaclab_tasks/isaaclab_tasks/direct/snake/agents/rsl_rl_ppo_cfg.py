from isaaclab.envs.configs.omni.rsl_rl.runner_config import PPORunnerCfg, ActorCriticCfg

class LocomotionPPORunnerCfg(PPORunnerCfg):
    actor_critic: ActorCriticCfg = ActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )
    num_steps_per_env = 24
    max_epochs = 1000
    num_mini_batches = 4

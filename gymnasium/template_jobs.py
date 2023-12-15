import gymnasium as gym
# from gymnasium.vector.vector_env import VectorEnvWrapper

import torch
from torch import nn

from dac.A2C_PPO_agent import ASquaredCPPOAgent, ASquaredCPPOAgentConfig, run_steps, eval
from dac.network import FCBody, OptionGaussianActorCriticNet
from dac.component.normalizer import MeanStdNormalizer
from dac.util.misc import random_seed


# DAC+PPO
def a_squared_c_ppo_continuous():
    vec_env = gym.vector.make("Reacher-v4", render_mode=None)
    env = gym.make("Reacher-v4", render_mode="human")
    # vec_env = gym.vector.make("HalfCheetah-v4", render_mode=None)
    # env = gym.make("HalfCheetah-v4", render_mode="human")
    observation_space = vec_env.observation_space.shape
    action_space = env.action_space.shape

    num_o = 4
    hidden_units = (64, 64)

    state_dim = 0 if observation_space is None else observation_space[-1]
    action_dim = 0 if action_space is None else action_space[-1]

    gate = nn.ReLU()

    opt_ep = 5

    config = ASquaredCPPOAgentConfig(
        num_o=4,
        state_dim=state_dim,
        action_dim=action_dim,

        opt_ep=opt_ep,
        # save_interval=int(1e4 / 2048) * 2048,
        # log_interval=2048,
        save_interval=int(1e4 / 256) * 256,
        log_interval=256,
        eval_interval=4096,

        gate=gate,
        entropy_weight=0.01,
        beta_weight=0,
        ppo_ratio_clip=0.2,

        max_steps=int(2e5),
        discount=0.99,
        rollout_length=2048,
        # rollout_length=256,
        optimization_epochs=opt_ep,
        mini_batch_size=64,

        env=vec_env,
        eval_env=env,
        num_workers=vec_env.num_envs,

        network=OptionGaussianActorCriticNet(
            state_dim, action_dim,
            num_options=num_o,
            actor_body=FCBody(
                state_dim,
                hidden_units=hidden_units, gate=gate),
            critic_body=FCBody(
                state_dim,
                hidden_units=hidden_units,
                gate=gate),
            option_body_fn=lambda: FCBody(
                state_dim,
                hidden_units=hidden_units,
                gate=gate),
        ),
        optimizer_fn=lambda params: torch.optim.Adam(
            params, 3e-4, eps=1e-5),

        use_gae=True,
        gae_tau=0.95,
        gradient_clip=0.5,

        # learning="all",
        # log_level=1,
        freeze_v=False,
        tasks=True,

        state_normalizer=MeanStdNormalizer(),
    )

    run_steps(ASquaredCPPOAgent(config))
    # eval(ASquaredCPPOAgent(config))
    # ASquaredCPPOAgent(config).eval_episode()


if __name__ == '__main__':
    random_seed()
    # set_one_thread()
    # select_device(-1)

    a_squared_c_ppo_continuous()

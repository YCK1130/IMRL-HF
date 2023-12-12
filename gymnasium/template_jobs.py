from dac import *
import subprocess
import gymnasium as gym
from gymnasium.vector.vector_env import VectorEnvWrapper


# DAC+PPO
def a_squared_c_ppo_continuous():
    config["learning"] = 'all'
    config["log_level"] = 1
    config["num_o"] = 4
    config["opt_ep"] = 5
    config["freeze_v"] = False
    config["tasks"] = True
    config["save_interval"] = int(1e6 / 2048) * 2048

    # gate=nn.Tanh(),

    config["log_level"] = 0
    config["num_o"] = 6
    config["learning"] = 'all'
    config["gate"] = nn.ReLU()
    config["freeze_v"] = False
    config["opt_ep"] = 5
    config["entropy_weight"] = 0.01
    config["tasks"] = False
    config["max_steps"] = 2e6
    config["beta_weight"] = 0

    # config["env"] = gym.vector.make("Reacher", render_mode=None)
    config["env"] = gym.vector.make("Reacher", render_mode="human")
    config["eval_env"] = gym.make("Reacher", render_mode=None)

    observation_space = config["env"].observation_space.shape
    action_space = config["env"].action_space.shape

    config["state_dim"] = 0 if observation_space is None else observation_space[-1]
    config["action_dim"] = 0 if action_space is None else action_space[-1]

    hidden_units = (64, 64)

    config["network"] = OptionGaussianActorCriticNet(
        config["state_dim"], config["action_dim"],
        num_options=config["num_o"],
        actor_body=FCBody(
            config["state_dim"],
            hidden_units=hidden_units, gate=config["gate"]),
        critic_body=FCBody(
            config["state_dim"],
            hidden_units=hidden_units,
            gate=config["gate"]),
        option_body_fn=lambda: FCBody(
            config["state_dim"],
            hidden_units=hidden_units,
            gate=config["gate"]),
    )
    config["optimizer_fn"] = lambda params: torch.optim.Adam(
        params, 3e-4, eps=1e-5)

    config["discount"] = 0.99
    config["use_gae"] = True
    config["gae_tau"] = 0.95
    config["gradient_clip"] = 0.5
    config["rollout_length"] = 2048
    config["optimization_epochs"] = config["opt_ep"]
    config["mini_batch_size"] = 64
    config["ppo_ratio_clip"] = 0.2
    config["log_interval"] = 2048
    config["state_normalizer"] = MeanStdNormalizer()
    run_steps(ASquaredCPPOAgent(config))


if __name__ == '__main__':
    random_seed()
    # set_one_thread()
    # select_device(-1)

    a_squared_c_ppo_continuous()

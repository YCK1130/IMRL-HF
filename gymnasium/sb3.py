import gymnasium as gym
# from Gymnasium.gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv
from stable_baselines3 import SAC, TD3, A2C, PPO
import os
# import wandb
import argparse
from gymnasium.wrappers import TimeLimit, RecordVideo, RecordEpisodeStatistics
import wandb
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from wandb.integration.sb3 import WandbCallback
import torch
from torch import nn
import spaces

# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

run_num = 3
date = '1211'
my_config = {
    "run_id": f"{date}_{run_num}",
    "policy_network": "MlpPolicy",
    "save_path": "models/",
    "saving_timesteps": 1e5,
    "device": "cuda",
    "first_stage_steps": 5e5,
    "second_stage_alternating_steps": 1e5,

    "testing_first_stage_steps": 0,
    "testing_second_stage_alternating_steps": 1e6,
    "comment": '''
    no foul, no random reset
    {
        "win": 5,
        "lose": -5,
        "draw": 0,
        "foul": -5,
    }
    ''',
}

class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 16,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 16, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(
                torch.as_tensor(observation_space.sample()[
                                None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))



def train(env, sb3_algo):
    run = wandb.init(
        project="Fencer",
        config=my_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        id=my_config["run_id"]
    )
    device = my_config['device']
    policy_network = my_config['policy_network']
    match sb3_algo:
        case 'SAC':
            model = SAC(policy_network, env, verbose=1,
                        device=device, tensorboard_log=log_dir)
        case 'TD3':
            model = TD3(policy_network, env, verbose=1,
                        device=device, tensorboard_log=log_dir)
        case 'A2C':
            model = A2C(policy_network, env, verbose=1,
                        device=device, tensorboard_log=log_dir)
        case 'PPO':
            model = PPO(policy_network, env, verbose=1,
                        device=device, tensorboard_log=log_dir,
                        )
        case _:
            print('Algorithm not found')
            return

    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=my_config['saving_timesteps'], reset_num_timesteps=False, callback=WandbCallback(
                            gradient_save_freq=100,
                            verbose=2,
                        ))
        model.save(f"{model_dir}/{date}_{run_num}_{sb3_algo}_{int(my_config['saving_timesteps']*iters)}")
        if my_config['saving_timesteps']*iters > 2e6:
            break


def test(env, sb3_algo, path_to_model):
    # run = wandb.init(
    #     project="assignment_3",
    #     config=my_config,
    #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     # id=my_config["run_id"]
    # )

    match sb3_algo:
        case 'SAC':
            model = SAC.load(path_to_model, env=env)
        case 'TD3':
            model = TD3.load(path_to_model, env=env)
        case 'A2C':
            model = A2C.load(path_to_model, env=env)
        case 'PPO':
            model = PPO.load(path_to_model, env=env)
        case _:
            print('Algorithm not found')
            return

    obs = env.reset()[0]
    done = False
    extra_steps = 60
    while True:
        action, _ = model.predict(obs)
        obs, _, done, _, _ = env.step(action)

        if done or extra_steps < 60:
            extra_steps -= 1
            if extra_steps < 0:
                obs = env.reset()[0]
                extra_steps = 60
                # break


if __name__ == '__main__':
    
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    # parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
    parser.add_argument(
        'sb3_algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()
    
    if args.train:
        gymenv = gym.make("Fencer",
                          render_mode=None,
                          first_state_step=my_config['first_stage_steps'],
                          alter_state_step=my_config['second_stage_alternating_steps'],
                          wandb_log=True)
        print(gymenv.action_space, gymenv.observation_space)
        if my_config['comment']: print(my_config['comment'])
        my_config['run_id'] = f"{date}_{run_num}_{args.sb3_algo}"
        rep = input(f"You are about to train '{my_config['run_id']}'. Press Y/y to continue... : ")
        if rep.lower() != 'y':
            exit(0)
        try:
            train(gymenv, args.sb3_algo)
            wandb.finish()
        except KeyboardInterrupt:
            wandb.finish()


    if (args.test):
        if os.path.isfile(args.test):
            gymenv = gym.make("Fencer", render_mode='human',
                              first_state_step=my_config['testing_first_stage_steps'],
                              alter_state_step=my_config['testing_second_stage_alternating_steps'],
                              wandb_log=False)
            test(gymenv, args.sb3_algo, path_to_model=args.test)
            # wandb.finish()
        else:
            print(f'{args.test} not found.')


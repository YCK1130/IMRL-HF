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
from deep_rl.agent.ASquaredC_PPO_agent import ASquaredCPPOAgent
from deep_rl.component.envs import TaskSB3
from deep_rl.utils import Config, MeanStdNormalizer, generate_tag, run_steps, tensor
from deep_rl.network import FCBody, OptionGaussianActorCriticNet
import torch
from torch import nn
import spaces
import tqdm
# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

run_num = 5
date = '1218'
my_config = {
    "run_id": f"{date}_{run_num}",
    "policy_network": "MlpPolicy",
    "save_path": f"models/{date}_{run_num}",
    "saving_timesteps": 1e5,
    "device": "cpu",
    "eval_episode_num": 100,
    "first_stage_steps": 1e4,
    "second_stage_alternating_steps": 1e5,
    "second_stage_model": "models/1215_1/PPO_1000000.zip",
    "max_steps": 2e6,

    "testing_first_stage_steps": 0,
    "testing_second_stage_alternating_steps": 1e6,
    "comment": '''1 goal reward, no random reset, small control penalty, train with trained model''',
}
os.makedirs(my_config['save_path'], exist_ok=True)
model_dir = my_config['save_path']
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


def dacConfigSetup(**kwargs):
    generate_tag(kwargs)
    kwargs.setdefault('game', 'Fencer')
    kwargs.setdefault('log_level', 0)
    kwargs.setdefault('num_o', 4)
    kwargs.setdefault('learning', 'all')
    kwargs.setdefault('gate', torch.nn.ReLU())
    kwargs.setdefault('freeze_v', False)
    kwargs.setdefault('opt_ep', 5)
    kwargs.setdefault('entropy_weight', 0.01)
    kwargs.setdefault('tasks', False)
    kwargs.setdefault('max_steps', 1e5)
    kwargs.setdefault('beta_weight', 0)
    config = Config()
    config.merge(kwargs)

    # if config.tasks:
    #     set_tasks(config)

    if 'dm-humanoid' in config.game:
        hidden_units = (128, 128)
    else:
        hidden_units = (64, 64)

    config.task_fn = lambda: TaskSB3(config.game,
                                  my_config=my_config,
                                  method=ASquaredCPPOAgent)
    config.eval_env = config.task_fn()

    config.network_fn = lambda: OptionGaussianActorCriticNet(
        config.state_dim, config.action_dim,
        num_options=config.num_o,
        actor_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        critic_body=FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
        option_body_fn=lambda: FCBody(config.state_dim, hidden_units=hidden_units, gate=config.gate),
    )
    config.optimizer_fn = lambda params: torch.optim.Adam(params, 3e-4, eps=1e-5)
    config.discount = 0.99
    config.use_gae = True
    config.gae_tau = 0.95
    config.gradient_clip = 0.5
    config.rollout_length = 2048
    config.optimization_epochs = config.opt_ep
    config.mini_batch_size = 64
    config.ppo_ratio_clip = 0.2
    config.log_interval = 2048
    config.state_normalizer = MeanStdNormalizer()
    return config



def train(env, sb3_algo):
    run = wandb.init(
        project="Fencer",
        entity="fencer",
        config=my_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        id=my_config["run_id"]
    )
    dac_config = dacConfigSetup(game='Fencer')
    device = my_config['device']
    # policy_network = my_config['policy_network']
    # match sb3_algo:
    #     case 'SAC':
    #         model = SAC(policy_network, env, verbose=1,
    #                     device=device, tensorboard_log=log_dir)
    #     case 'TD3':
    #         model = TD3(policy_network, env, verbose=1,
    #                     device=device, tensorboard_log=log_dir)
    #     case 'A2C':
    #         model = A2C(policy_network, env, verbose=1,
    #                     device=device, tensorboard_log=log_dir)
    #     case 'PPO':
    #         model = PPO(policy_network, env, verbose=1,
    #                     device=device, tensorboard_log=log_dir)
    #     case _:
    #         print('Algorithm not found')
    #         return
    model = ASquaredCPPOAgent(dac_config)
    iters = 0
    current_best_reward = -1e5
    current_best_win = 0
    while True:
        iters += 1
        # model.learn(total_timesteps=my_config['saving_timesteps'], reset_num_timesteps=False, callback=WandbCallback(
        #                     gradient_save_freq=100,
        #                     verbose=2,
        #                 ))
        run_steps(model)
        save_path = f"{model_dir}/{sb3_algo}_{int(my_config['saving_timesteps']*iters)}"
        dac_config.max_steps += my_config['saving_timesteps']
        if iters < 2:
            model.save(save_path)
            continue

        # Evaluation after 3 iterations
        avg_reward = 0
        avg_win = 0
        avg_steps = 0
        print("---------------")
        print("Evaluating...")
        for seed in tqdm.tqdm(range(my_config["eval_episode_num"])):
            done = False

            # Set seed using old Gym API
            env.seed(seed)
            obs = env.reset()[0]

            # Interact with env using old Gym API
            steps = 0
            while not done:
                action = model.record_step(obs)
                obs, _, done,_, info = env.step(action)
                steps += 1
                if steps > 2500:
                    break

            if done:
                avg_reward += info['reward']/my_config["eval_episode_num"]
                avg_win += info['win']/my_config["eval_episode_num"]
            avg_steps += steps/my_config["eval_episode_num"]
        print("Avg reward:", avg_reward)
        print(f"Avg steps: {avg_steps}")
        print(f"Win Prob.: {avg_win*100}%")
        print()
        # Save best model
        if my_config['saving_timesteps']*iters < my_config['first_stage_steps'] + my_config['second_stage_alternating_steps'] \
            and current_best_reward < avg_reward:
            print("Saving Model -- avg_reward")
            current_best_reward = avg_reward
            model.save(save_path)
        if my_config['saving_timesteps']*iters >= my_config['first_stage_steps'] + my_config['second_stage_alternating_steps']:
            if current_best_win < 0 :
                current_best_win = 0
            if current_best_win <= avg_win:
                print("Saving Model -- avg_win")
                current_best_win = avg_win
                model.save(save_path)
        print("---------------")

        if my_config['saving_timesteps']*iters > my_config['max_steps']:
            break


def test(env, sb3_algo, path_to_model):
    # run = wandb.init(
    #     project="assignment_3",
    #     config=my_config,
    #     sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    #     # id=my_config["run_id"]
    # )

    dac_config = dacConfigSetup(game='Fencer')
    model = ASquaredCPPOAgent(dac_config)
    model.load(path_to_model)
    obs = env.reset()[0]
    done = False
    eps_steps = 0
    while True:
        eps_steps += 1
        action = model.record_step(obs)
        obs, _, done, _, _ = env.step(action)

        if done or eps_steps > 500: # 500 steps is about 2500 frames
            if not done:print(f'eps_steps: {eps_steps}')
            eps_steps = 0
            obs = env.reset()[0]


if __name__ == '__main__':
    
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    # parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
    parser.add_argument(
        'sb3_algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    parser.add_argument('-s2', '--second_model', metavar='second_state_model')
    args = parser.parse_args()
    
    if args.train:
        if args.second_model:
            print(f'Specifing second model: {args.second_model}')
            gymenv = gym.make("Fencer",
                            render_mode=None,
                            first_state_step=my_config['first_stage_steps'],
                            wandb_log=True,
                            save_model_dir=my_config['save_path'],
                            second_state_method='manual',
                            second_state_model=args.second_model,
                            method=ASquaredCPPOAgent)
        else:
            gymenv = gym.make("Fencer",
                            render_mode=None,
                            first_state_step=my_config['first_stage_steps'],
                            alter_state_step=my_config['second_stage_alternating_steps'],
                            wandb_log=True,
                            save_model_dir=my_config['save_path'],
                            method=ASquaredCPPOAgent)
        print(gymenv.action_space, gymenv.observation_space)
        if my_config['comment']: print("comment: \n\t", my_config['comment'])
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
        # if os.path.isfile(args.test):
        if args.second_model:
            print(f'Specifing second model: {args.second_model}')
            gymenv = gym.make("Fencer", render_mode='human',
                                first_state_step=my_config['testing_first_stage_steps'],
                                wandb_log=False,
                                enable_random=True,
                                save_model_dir=my_config['save_path'],
                                second_state_method='manual',
                                second_state_model=args.second_model,
                                method=ASquaredCPPOAgent)
                                
        else:
            gymenv = gym.make("Fencer", render_mode='human',
                            first_state_step=my_config['testing_first_stage_steps'],
                            alter_state_step=my_config['testing_second_stage_alternating_steps'],
                            wandb_log=False,
                            enable_random=True,
                            save_model_dir=my_config['save_path'],
                            method=ASquaredCPPOAgent)
        test(gymenv, args.sb3_algo, path_to_model=args.test)
            # wandb.finish()
        # else:
            # print(f'{args.test} not found.')


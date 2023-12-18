import torch
import gymnasium as gym
# from Gymnasium.gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv
from stable_baselines3 import SAC, TD3, A2C, PPO
import os
# import wandb
import argparse
from gymnasium.wrappers import TimeLimit, RecordVideo, RecordEpisodeStatistics
import wandb
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
from deep_rl.agent.ASquaredC_PPO_agent import ASquaredCPPOAgent
from deep_rl.component.envs import TaskSB3
from deep_rl.utils import Config, MeanStdNormalizer, generate_tag, run_steps, tensor
from deep_rl.network import FCBody, OptionGaussianActorCriticNet
# from .deep_rl.agent import ASquaredC_A2C_agent 
# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
my_config = {
    "run_id": f"test",
    "policy_network": "MlpPolicy",
    "save_path": f"models/{1217}_{2}",
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

    config.task_fn = lambda: TaskSB3(config.game, my_config=my_config, method=ASquaredCPPOAgent)
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


def train(env, dac_config):
    run = wandb.init(
        project="assignment_3",
        config=my_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # id=my_config["run_id"]
    )
    
    agent = ASquaredCPPOAgent(dac_config)

    TIMESTEPS = 1e5
    iters = 0
    while True:
        iters += 1
        run_steps(agent)
        agent.save(f"{model_dir}/new_DAC_{int(TIMESTEPS*iters)}")
        if(iters >= 5):
            break
        dac_config.max_steps += TIMESTEPS
        


def test(env, dac_config, path_to_model):
    run = wandb.init(
        project="assignment_3",
        config=my_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # id=my_config["run_id"]
    )

    # match sb3_algo:
    #     case 'SAC':
    #         model = SAC.load(path_to_model, env=env)
    #     case 'TD3':
    #         model = TD3.load(path_to_model, env=env)
    #     case 'A2C':
    #         model = A2C.load(path_to_model, env=env)
    #     case 'PPO':
    #         model = PPO.load(path_to_model, env=env)
    #     case _:
    #         print('Algorithm not found')
    #         return

    obs = env.reset()[0]
    
    done = False
    extra_steps = 60
    agent = ASquaredCPPOAgent(dac_config)
    agent.load(path_to_model)
    while True:
        # agent.step()
        # env.reset()
        # print(obs)
        action = agent.record_step(obs)
        obs, _, done, _, _ = env.step(action)
        if done or extra_steps < 60:
            extra_steps -= 1

            if extra_steps < 0:
                obs = env.reset()[0]
                extra_steps = 60
        #         # break


if __name__ == '__main__':
    
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    # parser.add_argument('gymenv', help='Gymnasium environment i.e. Humanoid-v4')
    # parser.add_argument('sb3_algo', help='StableBaseline3 RL algorithm i.e. SAC, TD3')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-s', '--test', metavar='path_to_model')
    args = parser.parse_args()

    
    # FIX_MODEL = PPO.load('models/PPO_1000000')
    # print(FIX_MODEL.policy)
    # print(FIX_MODEL.predict([0]*59,deterministic=True))
    # print("DONE")
    dac_config = dacConfigSetup(game='Fencer')
    if args.train:
        gymenv = gym.make("Fencer", render_mode=None)
        print(gymenv.action_space, gymenv.observation_space)
        print(gymenv)
        train(gymenv, dac_config)
        # wandb.finish()


    if (args.test):
        # if os.path.isfile(args.test):
        gymenv = gym.make("Fencer", render_mode='human',first_state_step=1e6,alter_state_step=5e5)
        test(gymenv, dac_config, path_to_model=args.test)
        wandb.finish()
        # else:
        #     print(f'{args.test} not found.')
        

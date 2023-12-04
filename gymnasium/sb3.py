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
# Create directories to hold models and logs
model_dir = "models"
log_dir = "logs"
os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
my_config = {
    "run_id": "please",

    "algorithm": PPO,
    "policy_network": "MlpPolicy",
    "save_path": "models/",

   
}


def train(env, sb3_algo):
    run = wandb.init(
        project="assignment_3",
        config=my_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # id=my_config["run_id"]
    )
    match sb3_algo:
        case 'SAC':
            model = SAC('MlpPolicy', env, verbose=1,
                        device='cuda', tensorboard_log=log_dir)
        case 'TD3':
            model = TD3('MlpPolicy', env, verbose=1,
                        device='cuda', tensorboard_log=log_dir)
        case 'A2C':
            model = A2C('MlpPolicy', env, verbose=1,
                        device='cuda', tensorboard_log=log_dir)
        case 'PPO':
            model = PPO('MlpPolicy', env, verbose=1,
                        device='cuda', tensorboard_log=log_dir,
                        )
        case _:
            print('Algorithm not found')
            return

    TIMESTEPS = 1e5
    iters = 0
    while True:
        iters += 1

        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, callback=WandbCallback(
                            gradient_save_freq=100,
                            verbose=2,
                        ))
        model.save(f"{model_dir}/new_{sb3_algo}_{int(TIMESTEPS*iters)}")
        


def test(env, sb3_algo, path_to_model):
    run = wandb.init(
        project="assignment_3",
        config=my_config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        # id=my_config["run_id"]
    )

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

    
    # FIX_MODEL = PPO.load('models/PPO_1000000')
    # print(FIX_MODEL.policy)
    # print(FIX_MODEL.predict([0]*59,deterministic=True))
    # print("DONE")
    if args.train:
        gymenv = gym.make("Fencer", render_mode=None)
        print(gymenv.action_space, gymenv.observation_space)
        print(gymenv)
        train(gymenv, args.sb3_algo)
        wandb.finish()


    if (args.test):
        if os.path.isfile(args.test):
            gymenv = gym.make("Fencer", render_mode='human',first_state_step=10,alter_state_step=5)
            test(gymenv, args.sb3_algo, path_to_model=args.test)
        else:
            print(f'{args.test} not found.')

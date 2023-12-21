import gymnasium as gym
# from Gymnasium.gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv
from stable_baselines3 import SAC, TD3, A2C, PPO
import os
import argparse
from gymnasium.wrappers import TimeLimit, RecordVideo, RecordEpisodeStatistics


gymenv = gym.make("Fencer", render_mode="human", first_state_step=1e7)
gymenv.reset()
count = 0
action = [0]*10
while True:
    gymenv.render()
    count += 1
    # if count % 10 == 0:
    # action[0] = 1
    # print(action)
    gymenv.step(action)
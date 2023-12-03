import gymnasium as gym
# from Gymnasium.gymnasium.envs.mujoco.humanoid_v4 import HumanoidEnv
from stable_baselines3 import SAC, TD3, A2C, PPO
import os
import argparse
from gymnasium.wrappers import TimeLimit, RecordVideo, RecordEpisodeStatistics


gymenv = gym.make("Fencer", render_mode="human")
gymenv.reset()
while True:
    gymenv.render()
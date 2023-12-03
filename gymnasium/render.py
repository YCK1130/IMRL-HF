import gymnasium as gym
import numpy as np

if __name__ == '__main__':

    gymenv = gym.make("Fencer", render_mode='human')
    gymenv.reset()

    while True:
        gymenv.render()
        # gymenv.step(gymenv.action_space.sample())
        # gymenv.step(np.array([
        #     1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        #     1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0
        # ]))

import gymnasium as gym
# env = gym.make('Ant-v4')
env = gym.make('Ant-v4', render_mode='human')
# env = gym.make('Ant-v4', render_mode="rgb_array")

env.reset()

for _ in range(10000):
    env.step(env.action_space.sample())
    image = env.render()

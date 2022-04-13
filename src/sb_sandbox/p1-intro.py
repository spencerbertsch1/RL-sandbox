import gym
from stable_baselines3 import PPO, AC2
import os

# define the RL model that we will use 
MODEL: str = "PPO"

# define the local directory where we will store models 
models_directory = ''

env = gym.make("LunarLander-v2")
env.reset()

# define the stable baselines model 
model = PPO("MlpPolicy", env, verbose=1)

# now we can train the new model 
model.learn(total_timesteps=10_000)

episodes: int = 10

for ep in range(episodes):
    obs = env.reset()
    done = False

    while not done: 
        env.render()
        obs, reward, done, info = env.step(env.action_space.sample())

    # print('\n \n \n')
    # print(f'REWARD: {reward}')
    # print(f'DONE: {done}')
    # print(f'Observation: {obs}')

env.close()

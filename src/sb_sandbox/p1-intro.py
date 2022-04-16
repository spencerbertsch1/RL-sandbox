import gym
from stable_baselines3 import PPO, A2C
import os

# define the RL model that we will use 
MODEL: str = "PPO"

env = gym.make("LunarLander-v2")
env.reset()

# define the stable baselines model 
if MODEL == 'A2C':
    model = A2C("MlpPolicy", env, verbose=1)
elif MODEL == 'PPO':
    model = PPO("MlpPolicy", env, verbose=1)
else:
    raise Exception(f'MODEL should be \'A2C\' or \'PPO\', not {MODEL}. Please update and re-run.')

TIMESTEPS = 1000
model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=MODEL)

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

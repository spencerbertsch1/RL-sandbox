from stable_baselines3.common.env_checker import check_env
from wildfire_env import WildFireEnv
from test_env import TestEnv
from portfolio_opt import PortfolioOptEnv

env = WildFireEnv()
episodes = 3

for episode in range(episodes):
    done = False
    obs = env.reset()

    while done is False:#not done:
        random_action = env.action_space.sample()
        print("action",random_action)
        obs, reward, done, info = env.step(random_action)
        print('reward',reward, 'done', done)
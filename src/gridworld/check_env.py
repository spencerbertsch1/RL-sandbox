from stable_baselines3.common.env_checker import check_env
from wildfire_env import WildFireEnv
from test_env import TestEnv
from portfolio_opt import PortfolioOptEnv


env = WildFireEnv()
# It will check your custom environment and output additional warnings if needed
check_env(env)
print('Passing')

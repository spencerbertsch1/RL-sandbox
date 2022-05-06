from stable_baselines3 import PPO
import os
import time
from wildfire_env import WildFireEnv
from test_env import TestEnv
from portfolio_opt import PortfolioOptEnv

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = WildFireEnv(TRAIN_MODE=True, SHOW_IMAGE_BACKGROUND=False, SHOW_BURNED_NODES=False)
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 10  # <-- should be at least 1000
for i in range(3):
	# train the model TIMESTEPS number of times before saving a copy of the new model to disk
	model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*i}")

from stable_baselines3 import PPO
import os
import time
from wildfire_env import WildFireEnv
from test_env import TestEnv
from portfolio_opt import PortfolioOptEnv
import time

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = WildFireEnv(TRAIN_MODE=False, SHOW_IMAGE_BACKGROUND=False, SHOW_BURNED_NODES=False)
env.reset()

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 1000  # <-- should be at least 1000
for i in range(250):
	tic = time.time()
	# train the model TIMESTEPS number of times before saving a copy of the new model to disk
	model.learn(total_timesteps=10, reset_num_timesteps=False, tb_log_name=f"PPO")
	model.save(f"{models_dir}/{TIMESTEPS*i}")
	toc = time.time()
	print(f'Model Training Iteration finished in {round(toc - tic, 3)} seconds')

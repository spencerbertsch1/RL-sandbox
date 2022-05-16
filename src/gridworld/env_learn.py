from stable_baselines3 import PPO
import os
import time
from test_env import TestEnv
from portfolio_opt import PortfolioOptEnv
import time
import click

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

# tensorboard comman with optional args needed to run
# tensorboard --logdir logs --load_fast=false --reload_multifile=true --reload_multifile_inactive_secs=-1


@click.command()
@click.option('--env_version', default=9, help='Which one of the environments do you want to use for training?')
@click.option('--board_size', default=4, help='What is the side length of the grid you would like to use for training?')
def main(env_version, board_size):
    """
    Script that trains a PPO algorithm using the environment specifies in the command line argument
    """

    print(f'USING ENVIRONMENT VERSION: {env_version}')
    print(f'USING BOARD SIZE: {board_size}x{board_size}')

	# load the correct env based on the arg from click 
    if env_version == 1:
        from wildfire_env import WildFireEnv
    elif env_version == 2:
        from wildfire_env_v2 import WildFireEnv
    elif env_version == 3:
        from wildfire_env_v3 import WildFireEnv
    elif env_version == 4:
        from wildfire_env_v4 import WildFireEnv
    elif env_version == 5:
        from wildfire_env_v5 import WildFireEnv
    elif env_version == 6:
        from wildfire_env_v6 import WildFireEnv
    elif env_version == 7:
        from wildfire_env_v7 import WildFireEnv
    elif env_version == 8:
        from wildfire_env_v8 import WildFireEnv
    elif env_version == 9:
        from wildfire_env_v9 import WildFireEnv
    elif env_version == 10:
        from wildfire_env_v10 import WildFireEnv
    else:
        raise Exception('Please pass an environment number represented above! (1, 2, 3, ...)')

    env = WildFireEnv(TRAIN_MODE=False, SHOW_IMAGE_BACKGROUND=False, SHOW_BURNED_NODES=False, BOARD_SIZE=board_size)
    env.reset()

    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=logdir)

    TIMESTEPS = 10000  # <-- should be at least 1000
    for i in range(250):
        tic = time.time()
        # train the model TIMESTEPS number of times before saving a copy of the new model to disk
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
        model.save(f"{models_dir}/{TIMESTEPS*i}")
        toc = time.time()
        print(f'Model Training Iteration finished in {round(toc - tic, 3)} seconds')


# call main to kick off the training job
if __name__ == "__main__":
	main()

"""
Update observation to dictionary and include remaining phos chek in plane

"""
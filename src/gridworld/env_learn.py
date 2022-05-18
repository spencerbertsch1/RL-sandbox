from stable_baselines3 import PPO, DQN, A2C
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
@click.option('--env_version', default=12, help='Which one of the environments do you want to use for training?')
@click.option('--board_size', default=50, help='What is the side length of the grid you would like to use for training?')
@click.option('--policy', default='mlp', help='What is the policy you want to use for training?')
@click.option('--algorithm', default='DQN', help='What is the RL algorithm you want to use?')
def main(env_version, board_size, policy, algorithm):
    """
    Script that trains a PPO algorithm using the environment specifies in the command line argument
    """

    print(f'USING ENVIRONMENT VERSION: {env_version}')
    print(f'USING BOARD SIZE: {board_size}x{board_size}')
    print(f'USING POLICY: {policy}')

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
    elif env_version == 11:
        from wildfire_env_v11 import WildFireEnv
    elif env_version == 12:
        from wildfire_env_v12 import WildFireEnv
    else:
        raise Exception('Please pass an environment number represented above! (1, 2, 3, ...)')

    env = WildFireEnv(TRAIN_MODE=True, SHOW_IMAGE_BACKGROUND=False, SHOW_BURNED_NODES=False, BOARD_SIZE=board_size)
    env.reset()

    # define the policy that will be used for training
    if policy.lower() == 'mlp':
        rl_policy = 'MlpPolicy'
    elif policy.lower() == 'cnn':
        rl_policy = 'CnnPolicy'
    else:
        raise Exception(f'policy should be either \'mlp\' or \'cnn\', not {policy}')

    # define the RL algoeithm that will be used for training
    if algorithm == 'PPO':
        model = PPO(rl_policy, env, verbose=1, tensorboard_log=logdir)
    elif algorithm == 'DQN':
        model = DQN(rl_policy, env, verbose=1, tensorboard_log=logdir)
    elif algorithm == 'A2C':
        model = A2C(rl_policy, env, verbose=1, tensorboard_log=logdir)
    else:
        raise Exception(f'RL algorithm should be \'PPO\', \'A2C\', or \'DQN\' not {algorithm}')

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
TODO - Things to work on down the road...

1. Update observation to dictionary and include remaining phos chek in plane
"""

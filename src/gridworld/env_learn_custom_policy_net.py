from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import os
import time
from test_env import TestEnv
from portfolio_opt import PortfolioOptEnv
import time
import click

import gym
import torch as th
import torch.nn as nn



class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 256):
        super(CustomCNN, self).__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=1, padding="same"),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=128),
)

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
@click.option('--board_size', default=10, help='What is the side length of the grid you would like to use for training?')
@click.option('--policy', default='cnn', help='What is the policy you want to use for training?')
@click.option('--algorithm', default='PPO', help='What is the RL algorithm you want to use?')
def main(env_version, board_size, policy, algorithm):
    """
    Script that trains a PPO algorithm using the environment specifies in the command line argument
    """

    print(f'USING ENVIRONMENT VERSION: {env_version}')
    print(f'USING BOARD SIZE: {board_size}x{board_size}')
    print(f'USING POLICY: {policy}')
    print(f'USING RL ALGORITHM: {algorithm}')
    print(f'LOGGING TO THE FOLLOWING LOCATION: {logdir}')
    print(f'SAVING MODEL CHECKPOINTS TO THE FOLLOWING LOCATION: {models_dir}')

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
        if rl_policy == 'CnnPolicy':
            model = PPO("CnnPolicy", env, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=logdir)
        else:
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

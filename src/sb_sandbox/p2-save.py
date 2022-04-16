import gym
from stable_baselines3 import PPO, A2C
import os
import time 


def main(MODEL: str):
    """
    Train our specified RL model and save the model checkpoints locally 
    :return: NA
    """

    # define the local directory where we will store models 
    # models_dir = f'src/sb_sandbox/models/{MODEL}-{int(time.time())}'
    # log_dir = f'src/sb_sandbox/logs-{int(time.time())}'
    # ^^^ use this to train several models at once! open up multiple terminals and run this script in parallel

    models_dir = f'src/sb_sandbox/models/{MODEL}'
    log_dir = f'src/sb_sandbox/logs'

    # if that directory doesn't exist we should make it 
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    env = gym.make("LunarLander-v2")
    env.reset()

    # define the stable baselines model 
    if MODEL == 'A2C':
        model = A2C("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    elif MODEL == 'PPO':
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    else:
        raise Exception(f'MODEL should be \'A2C\' or \'PPO\', not {MODEL}. Please update and re-run.')

    # now we can train the new model 
    TIMESTEPS: int = 2000

    for i in range(1, 10):
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=MODEL)
        model.save(f"{models_dir}/{TIMESTEPS*i}")

    env.close()


if __name__ == "__main__":
    # define the RL model that we will use 
    MODEL: str = "A2C"
    # train our RL model and save the model checkpoints locally 
    main(MODEL=MODEL)

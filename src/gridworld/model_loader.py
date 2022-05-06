import gym
from stable_baselines3 import PPO, A2C
from wildfire_env import WildFireEnv
import os
import time

def main(MODEL: str):
    """
    Loads a trained model and displays five episodes
    :return: NA
    """

    env = WildFireEnv(TRAIN_MODE=False, SHOW_IMAGE_BACKGROUND=False, SHOW_BURNED_NODES=False)
    env.reset()

    models_dir = f'src/sb_sandbox/models/{MODEL}'
    models_path = f'{models_dir}/18000.zip'

    # FIXME - remove this hard coded path later
    models_path = '/Users/spencerbertsch/Desktop/dev/RL-sandbox/models/1651867463/40.zip'

    # load the model 
    model = PPO.load(models_path, env=env)

    episodes: int = 5

    for ep in range(episodes):
        obs = env.reset()
        done = False

        while not done: 
            # env.render()  # <-- we don't use a render() method, so we don't need this! (we use TRAIN_MODE instance variable instead)
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)


    env.close()


if __name__ == "__main__":
    
    # define the RL model that we will use 
    MODEL: str = "PPO"
    # Load a trained model and displays ten episodes
    main(MODEL=MODEL)
import gym
from stable_baselines3 import PPO, A2C
import os
import time

def main(MODEL: str):
    """
    Loads a trained model and displays ten episodes
    :return: NA
    """

    env = gym.make("LunarLander-v2")
    env.reset()

    models_dir = f'src/sb_sandbox/models/{MODEL}'
    models_path = f'{models_dir}/18000.zip'

    # load the model 
    model = PPO.load(models_path, env=env)

    episodes: int = 10

    for ep in range(episodes):
        obs = env.reset()
        done = False

        while not done: 
            env.render()
            action, _ = model.predict(obs)
            obs, reward, done, info = env.step(action)


    env.close()


if __name__ == "__main__":
    
    # define the RL model that we will use 
    MODEL: str = "PPO"
    # Load a trained model and displays ten episodes
    main(MODEL=MODEL)
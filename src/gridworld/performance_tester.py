import gym
from stable_baselines3 import PPO, A2C
from wildfire_env import WildFireEnv
from wildfire_env_v13 import WildFireEnv
# from wildfire_env_v12 import WildFireEnv
# from env_test_only_move import WildFireEnv
import os
import time
import pickle

from routines import boxplot_dict
from heuristics import true_optimal_ex1, heuristic_ex1

def main(MODEL: str, SAVE_REWARDS: bool, RUN_NAME: str, SHOW_REWARDS: True, HEURISTIC_NOISE: False):
    """
    Loads a trained model and displays five episodes
    :return: NA
    """

    reward_dir = f"reward_tracking/EX1/"
    figure_dir = f"figures/"

    if not os.path.exists(reward_dir):
        os.makedirs(reward_dir)

    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    if SAVE_REWARDS: 

        env = WildFireEnv(TRAIN_MODE=False, SHOW_IMAGE_BACKGROUND=False, SHOW_BURNED_NODES=False, BOARD_SIZE=10)
        env.reset()

        # Add path to the model that we want to load
        models_path = '/Users/spencerbertsch/Desktop/code/RL-sandbox/src/gridworld/models/1653147804/1000000.zip'
        # models_path = '/Users/spencerbertsch/Desktop/code/RL-sandbox/src/gridworld/models/1652982511/1760000.zip'

        # load the model 
        model = PPO.load(models_path, env=env)

        episodes: int = 15

        all_rewards = []
        for ep in range(episodes):
            obs = env.reset()
            done = False
            episode_reward = 0
            print(f'Performance Testing {RUN_NAME}_{MODEL} - Episode {ep+1}/{episodes}.')

            ep_length = 0
            while not done: 
                # random run 
                if (RUN_NAME == 'random') | (MODEL == 'random'):
                    action = env.action_space.sample()
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward

                # true optimal run - teleportation allowed (constraint relaxation)
                if (RUN_NAME == 'relaxation') | (MODEL == 'relaxation'):
                    action = env.action_space.sample()
                    obs, reward, done, info = env.step(action)
                    reward = true_optimal_ex1(obs=obs)
                    episode_reward += reward

                # heuristic run - with or without noise
                elif (RUN_NAME == 'heuristic') | (MODEL == 'heuristic'):
                    action = heuristic_ex1(obs=obs, noise=HEURISTIC_NOISE)
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward
                
                # traditional run with a trained model 
                else:
                    action, _ = model.predict(obs)
                    obs, reward, done, info = env.step(action)
                    episode_reward += reward
                    ep_length += 1

            all_rewards.append(episode_reward)
            print(f'Steps taken during episode: {ep_length}, Total Episode Reward: {episode_reward} \n')
            ep_length = 0

        env.close()

        # save the pickle file 
        run_full_name: str = f'{RUN_NAME}_{MODEL}'
        reward_dict = {run_full_name: all_rewards}
        # save the reward dict to disk via pickle 
        file_to_write = open(f"{reward_dir}/{run_full_name}.pickle", "wb")
        pickle.dump(reward_dict, file_to_write)

        print(f'Performance test for {RUN_NAME}_{MODEL} complete.')

    if SHOW_REWARDS: 
        rewards_dict = {}
        for filename in os.listdir(reward_dir):
            f = os.path.join(reward_dir, filename)
            # checking if it is a file
            if os.path.isfile(f):
                # reading the data from the file
                with open(f, 'rb') as handle:
                    data = handle.read()
                # reconstructing the data as dictionary
                d = pickle.loads(data)

                # add all the k-v pairs in d to rewards_dict
                rewards_dict.update(d)

        # now we just need to make a nice boxplot of the results 
        boxplot_dict(fname=f'performance_test', input_dict=rewards_dict, 
                     boxplot_title='Reward Per Episode for Each Policy', 
                     x_label='Policy Being Evaluated', 
                     y_label='Reward Per Episode', image_type='png',
                     save_results=True, show_results=True, sort_keys=False, 
                     path_to_figs=figure_dir)


if __name__ == "__main__":
    
    # define the RL model that we will use 
    MODEL: str = "PPO"
    # Load a trained model and displays ten episodes
    SAVE_REWARDS = True
    # Display a chart of the rewards for different models 
    PLOT_REWARDS = False
    # name for the current run
    RUN_NAME = 'RL'
    # Define heuristic noise
    HEURISTIC_NOISE = True
    main(MODEL=MODEL, SAVE_REWARDS=SAVE_REWARDS, RUN_NAME=RUN_NAME, SHOW_REWARDS=PLOT_REWARDS, 
         HEURISTIC_NOISE=HEURISTIC_NOISE)

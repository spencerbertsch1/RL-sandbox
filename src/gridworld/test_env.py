import gym
from gym import spaces
import numpy as np
import random

class TestEnv(gym.Env):
    """Custom Environment that follows gym interface"""

    BOARD_SIZE = 3  # 3x3 board 

    def __init__(self):
        super(TestEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(3)
        # Example for using image as input (channel-first; channel-last also works):
        # observation is the x, y coordinate of the grid
        low = np.zeros(self.BOARD_SIZE, dtype=int)
        high =  np.array(self.BOARD_SIZE, dtype=int) - np.ones(self.BOARD_SIZE, dtype=int)
        # self.observation_space = spaces.Box(low, high, dtype=np.int64)
        self.observation_space = spaces.Box(low=np.array([0]), high=np.array([255]), dtype=np.int64)

    def step(self, action):
        
        if action == 0:
            self.environment[0] += 1 
        elif action == 1: 
            self.environment[1] += 1
        elif action == 2: 
            self.environment[2] += 1

        self.reward = float(sum(self.environment))

        if sum(self.environment) > 765: 
            self.done = True
        else: 
            self.done = False

        self.observation = np.array([38 + random.randint(-3,3)]).astype(float)

        info = {}
        return self.observation, self.reward, self.done, info
    
    def reset(self):
        # we observe where we are on the board
        # self.observation = np.zeros(2)

        self.environment = np.array([0, 0, 0])

        self.observation = np.array([38 + random.randint(-3,3)]).astype(float)

        return self.observation  # reward, done, info can't be included

# type of GAN where a RL model builds an image pixel by pixel and a network discriminates whether or not it's good
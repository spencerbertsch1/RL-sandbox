import gym
import numpy as np


# we create a new MountainCar environment from scratch here 
env = gym.make("MountainCar-v0")
# here we reset the environment so we always start from the same place 
env.reset()

NUM_WINDOWS = 20  # <-- this is a hyperparameter and generally requires hyperparameter optimization 

print(env.observation_space.high)
print(env.observation_space.low)
print(env.action_space.n)  # <-- number of actions available

# create a list filled with the int NUM_WINDOWS with length of the number of dimensions of our environemnt (2 in this case)
discrete_os_size: list = [NUM_WINDOWS] * len(env.observation_space.high)

# find the window sizes we will use across the x and y axes 
discrete_os_window_size = (env.observation_space.high - env.observation_space.low) / discrete_os_size

# base these numbers on the rewards for this environment!
initial_q_low: int = -2
initial_q_high: int = 0
# initialize the Q table (dimensions: (20 x 20 x 3))
q_table = np.random.uniform(low=initial_q_low, high=initial_q_high, size=(discrete_os_size + [env.action_space.n]))  





done: bool = False

# STATE: [position, velocity]
# ACTIONS: [go-left, go-right, do-nothing]
while not done:
    action = 2  # <-- here we choose to always go right (for this toy example)
    new_state, reward, done, _ = env.step(action)
    print(f'REWARD: {reward}, STATE_POSITION: {round(new_state[0], 3)}, STATE_VELOCITY: {round(new_state[1], 3)}')
    env.render()

env.close()


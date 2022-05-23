
import numpy as np
import math
from math import dist

def centeroidnp(arr):
    length = arr.shape[0]
    sum_x = np.sum(arr[:, 0])
    sum_y = np.sum(arr[:, 1])
    return sum_x/length, sum_y/length
    

def true_optimal_ex1(obs):
    """
    Given an observation, this function returns the true optimal value for experiment 1. 
    """
    num_burning_nodes = 0

    # Step 1: make the array for the centroid calculation: 
    # create an empty np array with 2 columns
    arr = np.empty((0, 2), int)
    # array shape for 10 x 10 grid: (1, 10, 10)
    for i in range(np.shape(obs)[1]):
        for j in range(np.shape(obs)[2]):
            if obs[0][i][j] == 100: 
                arr = np.append(arr, np.array([[i, j]]), axis=0)
                num_burning_nodes += 1

    # Step 2: find the centroid 
    centroid = centeroidnp(arr=arr)

    # we define the reward_offset so that even if the plane is all the way in the opposite corner, the reward is still positive
    reward_offset: float = math.sqrt((10**2) + (10**2))

    fire_dist = 0
    # array shape for 10 x 10 grid: (1, 10, 10)
    for i in range(np.shape(obs)[1]):
        for j in range(np.shape(obs)[2]):
            if obs[0][i][j] == 100: 
                curr_fire_dist: float = round(dist(centroid, (i, j)), 3)
                fire_dist += curr_fire_dist

    # # here we handle the edge case the comes up at the very end of each run where the fire has burned out
    if num_burning_nodes == 0: 
        fire_dist = 1
    else:
        # normalize the fire distance over the number of burning nodes
        fire_dist: float = float(round(fire_dist / num_burning_nodes, 3))
    
    # DEFINE REWARD FOR EACH STEP HERE: 
    reward: float = round((reward_offset - fire_dist), 4)

    return reward


def heuristic_ex1(obs, noise: bool):

    num_burning_nodes = 0

    # Step 1: make the array for the centroid calculation: 
    # create an empty np array with 2 columns
    arr = np.empty((0, 2), int)
    plane_state = [0, 0]
    # array shape for 10 x 10 grid: (1, 10, 10)
    for i in range(np.shape(obs)[1]):
        for j in range(np.shape(obs)[2]):
            if obs[0][i][j] == 100: 
                arr = np.append(arr, np.array([[i, j]]), axis=0)
                num_burning_nodes += 1
            elif obs[0][i][j] == 200:
                plane_state = [i, j] 
                
    # Step 2: find the centroid 
    centroid = centeroidnp(arr=arr)

    x_diff = centroid[0] - plane_state[0]
    y_diff = centroid[1] - plane_state[1]

    # We can now select the action that goes in the direction of the centroid
    # 0: east, 
    # 1: south, 
    # 2: west, 
    # 3: north

    if (centroid[0] == plane_state[0]) & (centroid[1] == plane_state[1]):
        # the plane is sitting on top of the centroid
        action =  np.random.choice([0, 1, 2, 3])
    
    else:
        if abs(x_diff) > abs(y_diff):
            # we need to move east or west
            if x_diff > 0: 
                action = 0
            else:
                action = 2

        elif abs(y_diff) > abs(x_diff):
            # we need to move east or west
            if y_diff > 0: 
                action = 1
            else:
                action = 3
        else: 
            # the centroid is in an off diagonal location so the plane can move either in the x or y direction
            if (x_diff < 0) & (y_diff < 0):
                action = np.random.choice([2, 3])
            elif (x_diff < 0) & (y_diff > 0):
                action = np.random.choice([1, 2])
            elif (x_diff > 0) & (y_diff < 0):
                action = np.random.choice([3, 0])
            else:
                action = np.random.choice([1, 0])

    if noise: 
        U = np.random.uniform(1,0)
        if U > 0.5: 
            actions = [0, 1, 2, 3]
            actions.remove(action)
            action = np.random.choice(actions)

    return action

import gym
from gym import spaces
import cv2
import imageio
import random
import numpy as np
import math
from math import dist
import time
import matplotlib.pyplot as plt
from pyrsistent import freeze
import seaborn as sns
random.seed(10)


class Airport:
    """
    Airport class - allows planes to fill more Phos Chek and get more fuel 
    """
    def __init__(self, state):
        self.state = state    # [x, y] position of the airport in the grid 


class Plane:
    """
    Plane (agent) for the wild fire RL environment 
    """
    def __init__(self, state, previous_state, phos_chek_level: int, direction: int, BOARD_SIZE: int):
        self.state = state                          # [x, y] position of the plane in the grid  
        self.previous_state = previous_state        # [x, y] position of the plane's previous position in the grid  
        self.phos_chek_level = phos_chek_level      # int representing the amount of Phos Chek left in the plane
        self.direction = direction                  # the direction that the plane is flying (N, E, S, W)  
        self.BOARD_SIZE = BOARD_SIZE                # the size of the board (env) that the plane is flying over

    def move(self):
        """
        Method used to move the agent around the environment 
        """
        # update the previous state to the current state before we move the plane
        self.previous_state = self.state.copy()

        # Checks what its current direction is and moves accordingly
        if self.direction == 0:
            # move east
            self.state[0] += 1

            # if we are falling off the East wall 
            if self.state[0] == self.BOARD_SIZE:
                # undo the move
                self.state[0] -= 1
                # are we in the north east corner?
                if self.state[1] == 0:
                    # move south
                    self.direction = 1
                    self.state[1] += 1
                else:
                    # move north instead
                    self.direction = 3
                    self.state[1] -= 1

        elif self.direction == 1:
            # move south
            self.state[1] += 1

            # if we are falling off the south wall 
            if self.state[1] == self.BOARD_SIZE:
                # undo the move
                self.state[1] -= 1
                # are we in the south east corner?
                if self.state[0] == self.BOARD_SIZE-1:
                    # if yes then we have to move west 
                    self.direction = 2
                    self.state[0] -= 1
                else:
                    # move east instead
                    self.direction = 0
                    self.state[0] += 1

        elif self.direction == 2:
            # move west 
            self.state[0] -= 1

            # if we are falling off the west wall 
            if self.state[0] < 0:  
                # undo the move
                self.state[0] += 1
                # are we in the south west corner?
                if self.state[1] == self.BOARD_SIZE-1:
                    # if yes then we have to move north 
                    self.direction = 3
                    self.state[1] -= 1
                else:
                    # move south instead
                    self.direction = 1
                    self.state[1] += 1
        
        elif self.direction == 3:
            # move north
            self.state[1] -= 1

            # if we are falling off the north wall 
            if self.state[1] < 0:  
                # undo the move
                self.state[1] += 1
                # are we in the north west corner?
                if self.state[0] == 0:
                    # if yes then we have to move east 
                    self.direction = 0
                    self.state[0] += 1
                else:
                    # move west instead
                    self.direction = 2
                    self.state[0] -= 1


class Node:
    """
    Forest Node - node for the wild fire RL envoironment 
    """

    def __init__(self, state, heuristic, fuel=1, burn_speed=None, phos_chek=False, neighbors=None, burning=None):
        self.state = state              # [x, y] position of the node in the grid 
        self.heuristic = heuristic      # euclidean distance to closest human inhabited area 
        self.burn_speed = burn_speed    # {1: slow (sandy area), 2: moderate (wet area), 3: fast (dry forest)}
        self.fuel = fuel                # amount of remaining fuel in this node 
        self.phos_chek = phos_chek      # phos_chek concentration in this cell
        self.neighbors = neighbors      # 2-4 nodes that represent the adjacent nodes 
        self.burning = burning          # bool representing whether or not the cell is burning

def lists_overlap(a, b):
    """
    small utility function that returns True if the input lists share any elements
    """
    # convert the list contents into tuples
    a_tuples = set(tuple(i) for i in a)
    b_tuples = set(tuple(i) for i in b) 
    overlap: set = a_tuples & b_tuples
    if len(overlap) != 0: 
        # there is overlap
        return True
    else:
        # there is no overlap 
        return False

class WildFireEnv(gym.Env):
    """
    Custom Environment that follows gym interface

    This environment is designed to represent a wildfire with a single plane that can drop fire retardant
    in an attempt to put out the fire.
    """

    def __init__(self, TRAIN_MODE: bool, SHOW_IMAGE_BACKGROUND: bool, SHOW_BURNED_NODES: bool, BOARD_SIZE: int = 4):
        super(WildFireEnv, self).__init__()
        
        # environment parameters
        self.BOARD_SIZE = BOARD_SIZE
        self.CELL_SIZE = int(1500/BOARD_SIZE)
        
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(5)
        # Example for using image as input (channel-first; channel-last also works):
        # self.observation_shape = (100, 100, 1)  # <-- MAKE SURE THIS MATCHES THE BOARD SIZE! (For observations of the board)
        low = np.zeros(shape=(self.BOARD_SIZE, self.BOARD_SIZE, 1), dtype=np.uint8)
        high =  np.ones(shape=(self.BOARD_SIZE, self.BOARD_SIZE, 1), dtype=np.uint8)*255
        # self.observation_space = spaces.Box(low, high, dtype=np.int64)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.uint8)

        # parameters used to render the map using CV2
        self.TRAIN_MODE = TRAIN_MODE
        self.SHOW_IMAGE_BACKGROUND = SHOW_IMAGE_BACKGROUND
        self.SHOW_BURNED_NODES = SHOW_BURNED_NODES


    def initiate_drop(self):
        """
        Method that ensures the plane will drop phos chek for its maximum drop length unless it flies over 
        any old phos chek drop, any burned nodes, or any burning nodes. If that happens, then the self.phos_check_dump
        instance variable is set to False. 
        """

        if self.phos_check_dump: 
            if self.plane.phos_chek_level > 0:
                # we're already dumping phos chek so this does nothing
                pass
            else:
                self.phos_check_dump = False

        else:
            # we need to start dropping phos chek for n time steps
            self.phos_check_dump = True
            self.plane.phos_chek_level = self.MAX_PHOS_CHEK


    def generate_heatmap(self):

        map: np.array = np.empty((self.BOARD_SIZE, self.BOARD_SIZE))

        reward_offset: float = math.sqrt((self.BOARD_SIZE**2) + (self.BOARD_SIZE**2))

        burning_states: list = [n.state for n in self.burning_nodes]

        # iterate through each cell of the map
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):

                # get the fire distance of each cell
                fire_dist = 0
                for state in burning_states:
                    curr_fire_dist: float = round(dist([i, j], state), 3)
                    fire_dist += curr_fire_dist

                # calculate the reward 
                curr_reward = round((reward_offset - fire_dist)/3, 2)
                map[i][j] = curr_reward

        # transpose array 
        map = np.transpose(map)

        # make the heatmap and save it
        sns.heatmap(map, annot=True, linewidths=.5)

    def step(self, action):
        """
        TODO
        """
        t = time.time()

        if len(self.burning_nodes) == 0:
            # the game is over, so we set done to True
            self.done = True
            if self.VERBOSE:
                self.print_results()
        
        # Makes and displays the board_states
        self.display_dict: dict = self.display(fire_time=self.fire_time, curr_score=self.curr_score, 
                                     layer2_cache=self.layer2_cache, old_burned_nodes=self.old_burned_nodes)
        self.key = self.display_dict['key']
        self.old_burned_nodes = self.display_dict['old_burned_nodes']
        self.layer2_cache = self.display_dict['layer2_cache']

        # Gets key presses and moves accordingly
        # 8 and 27 are delete and escape keys
        # Arrow keys are tricky in OpenCV. So we use
        # keys 'w', 'a','s','d' for movement. 
        # w = top, a = left, s = down, d = right

        if action == 0:
            self.plane.direction = 0
        elif action == 1:
            self.plane.direction = 1
        elif action == 2:
            self.plane.direction = 2
        elif action == 3:
            self.plane.direction = 3
        elif action == 4:
            # activate airal "attack"
            self.initiate_drop()
        # elif key == ord("p"): 
        #     self.quit = True

        # Moving the plane
        self.plane.move()    

        """
        NEW CONSTRAINTS - no dumping on burned, burning, or phos chek nodes! 
        """
        if self.phos_check_dump: 
            # we don't need to worry that the plane is dumping with no phos chek because the plane has 1 million nodes worth
            # TODO these loops, especially the burned nodes and phos chek nodes will take a LONG time for large grids - we need to vectorize here
            if (self.plane.previous_state in [node.state for node in self.burning_nodes]) | \
               (self.plane.previous_state in [node.state for node in self.burned_nodes]) | \
               (self.plane.previous_state in [node.state for node in self.phos_chek_nodes]):  # TODO <-- this line could take a long time!! We need to vectorize here
               self.phos_check_dump = False

        if self.phos_check_dump:
            if self.plane.phos_chek_level > 0:
                # we are in a time of dumping the phos_chek so we add this node to the phos_chek nodes
                self.phos_chek_nodes = self.get_phos_chek_nodes(plane_x=self.display_dict['plane_x'], plane_y=self.display_dict['plane_y'])
                # reduce Phos Chek level in the plane
                self.plane.phos_chek_level = self.plane.phos_chek_level - 1

        if self.plane.state == self.airport.state: 
            # plane is stopping at the airport
            self.phos_check_dump = False
            self.plane.phos_chek_level = self.MAX_PHOS_CHEK

        # cause the fire to spread either deterministically or via a stochastic function 
        if self.c == self.FIRE_SPEED: 
            if len(self.burning_nodes) == 0:
                # the game is over, so we set done to True
                self.done = True
                if self.VERBOSE:
                    self.print_results()
            else:
                self.burning_nodes = self.get_next_burning(self.burning_nodes, first_ignition=self.first_ignition)
                self.first_ignition=False
                self.c = 0
                # increment the clock 
                self.fire_time = self.fire_time + self.FIRE_TIMESTEP
                self.curr_score = (self.BOARD_SIZE*self.BOARD_SIZE) - len(self.burned_nodes)

                # create the heatmap if we need to 
                if self.CREATE_HEATMAP:
                    self.generate_heatmap()
        else:
            self.c += 1

        if self.VERBOSE: 
            step_time: float = round(time.time() - t, 5)
            print(f'TIME TAKEN FOR THIS STEP: {round(step_time, 4) * 1000} MS')
            self.step_times.append(step_time)


        # calculate the reward
        self.reward = self.calculate_reward()

        # define new observation after the step 
        self.observation = self.make_observation()

        # self.observation: dict = {
        #     """
        #     This encapsulates all the information that the agent can 'see' based on what 
        #     the air tactical group supervisor knows at any given time.

        #     TODO we may need to pass the states within these objects, but this should be fine
        #     """
        #     'burned_nodes':  self.burned_nodes,
        #     'burning_nodes': self.burning_nodes, 
        #     'phos_chek_nodes': self.phos_chek_nodes,
        #     'plane': self.plane, 
        #     'airport': self.airport 
        # }

        info: dict = {}
        return self.observation, self.reward, self.done, info


    def reset(self):
        """
        TODO Add docstring
        
        """
        self.done=False
        # it's very important that these simulations are repeatable - we rely on random.seed() for this

        # define globals 

        # THESE ARE NOW CLASS VARIABLES !!! CHANGE THEM AT THE CLASS LEVEL!
        # Size of each cell in the board game
        # self.CELL_SIZE = 15
        # Number of cells along the width in the game
        # self.BOARD_SIZE = 100

        # Change SPEED to make the game go faster
        self.SPEED = 25 # <-- lower is faster
        # Maximum speed at which the fire advances
        self.FIRE_SPEED = 15  # <-- lower is faster
        # the amount of time in actual minutes that go by before the fire advances one step (this needs to be calibrated realistically)
        self.FIRE_TIMESTEP = int(self.FIRE_SPEED*3)  # minutes
        # define max amount of Phos Chek that a plane can carry (will depend on type of aircraft)
        self.MAX_PHOS_CHEK = 10
        # Wind
        self.WIND_DIRECTION = 4
        self.wind_direction_lookup = {'north': 1, 'northeast': 2, 'east': 3, 'southeast': 4, 'south': 5, 'southwest': 6, 'west': 7, 'northwest': 8}
        self.WIND_SPEED = 5
        # set to True if the reward function should calculate distance to fire based on upwind node
        self.USE_UPWIND_NODE_FOR_REWARD = False

        # adds helpful print statements 
        self.VERBOSE = False
        # True if we want to create a GIF of the output
        self.CREATE_GIF = False
        # True if you want to create a heatmap at each fire step update
        self.CREATE_HEATMAP = True
        # if we do want to create the gif, let's create a list to store the image frames 
        if self.CREATE_GIF: 
            self.frames: list = []
        
        # controls the tradeoff between the short term rewards in the game and the final reward of the number of nodes saved
        self.REWARD_BALANCER = 0.5

        if self.SHOW_IMAGE_BACKGROUND is True: 
            background_image = cv2.imread('/Users/spencerbertsch/Desktop/dev/RL-sandbox/src/images/occidental_vet_hospital.png')
            self.layer1 = np.zeros([background_image.shape[0], background_image.shape[1], 4])

            for i in range(self.layer1.shape[0]):
                for j in range(self.layer1.shape[1]): 
                    # TODO we will eventually set up the RGB of the board depending on the fuel in each node
                    self.layer1[i][j] = np.uint8(np.append(background_image[i][j], 255))

        else:
            # we could make this configurable later on - for now this will only work with 1500px by 1500px images 
            self.layer1 = np.zeros([1500, 1500, 4])

        # list of lists representing the board of all nodes 
        self.node_map: list = self.generate_initial_nodes()

        self.plane_start_state = [int((self.BOARD_SIZE - 1)/2), int((self.BOARD_SIZE - 1)/2)]
        
        # plane starts at the center of the board. 
        self.plane = Plane(state=self.plane_start_state, previous_state=self.plane_start_state, 
                           phos_chek_level=self.MAX_PHOS_CHEK, direction=1, BOARD_SIZE=self.BOARD_SIZE)

        # initislize fire start location - use the below code to add more starting fires
        # self.fire_start_state = [(0,0)]  # [(10, 10), (15, 55)]  # <-- good values for larger boards
        # self.burning_nodes: list = [self.node_map[self.fire_start_state[0][0]][self.fire_start_state[0][1]], 
        #                             self.node_map[self.fire_start_state[1][0]][self.fire_start_state[1][1]]]
        # self.burning_nodes: list = [self.node_map[self.fire_start_state[0][0]][self.fire_start_state[0][1]]]

        self.fire_start_state = [(0, 0)]  
        self.burning_nodes: list = [self.node_map[self.fire_start_state[0][0]][self.fire_start_state[0][1]]]

        self.airport = Airport(state=[int((self.BOARD_SIZE - 1)), int((self.BOARD_SIZE - 1))])

        # define blackened nodes (already burned)
        self.burned_nodes = []
        # define retardant nodes (PHOS-CHEK already dropped here)
        self.phos_chek_nodes = []
        self.board_start_state = np.zeros([self.BOARD_SIZE * self.CELL_SIZE, self.BOARD_SIZE * self.CELL_SIZE, 4])
        # use this cache to speed up the rendering of the layered image
        self.layer2_cache = self.board_start_state.copy()
        self.old_burned_nodes = []

        # Ugly trick to bring the window in focus
        if not self.TRAIN_MODE:
            self.win_focus()

        # define other parameters for this run 
        self.c = 0
        self.fire_time = 0
        self.curr_score: int = self.BOARD_SIZE*self.BOARD_SIZE
        self.first_ignition = True
        self.phos_check_dump = False
        self.step_times = []

        # todo we could add more dimansions to this later to add multiple values to the same cell...
        self.observation = self.make_observation()

        self.reward = 0

        # to visualize the observation
        # plt.imshow(self.observation, cmap='hot', interpolation='nearest')
        # plt.show()
    
        return self.observation


    def get_neighbors(self, node: Node, node_map: list):

        x = node.state[0]
        y = node.state[1]

        neighbor_states: list = [(x2, y2) for x2 in range(x-1, x+2)
                                    for y2 in range(y-1, y+2)
                                    if (-1 < x <= self.BOARD_SIZE and
                                        -1 < y <= self.BOARD_SIZE and
                                        (x != x2 or y != y2) and
                                        (0 <= x2 <= self.BOARD_SIZE) and
                                        (0 <= y2 <= self.BOARD_SIZE))]

        neighbor_nodes = []
        
        # perform the lookup in the node_map to get the neighbor nodes
        for state in neighbor_states:
            if (state[0] < self.BOARD_SIZE) and (state[1] < self.BOARD_SIZE):
                n_node = node_map[state[0]][state[1]]
                neighbor_nodes.append(n_node)

        return neighbor_nodes

    def get_down_wind_state(self, state: list) -> list:
        """
        :param: state - list of two ints representing [x, y] of the node in question - [10, 10] for example
        Note that the origin is in the UPPER LEFT of the grid 
        """
        if self.WIND_DIRECTION == 1: 
            # fire burns north
            down_wind_state = [state[0]-1, state[1]]
        elif self.WIND_DIRECTION == 4:
            down_wind_state = [state[0]+1, state[1]+1]

        return down_wind_state


    def get_up_wind_state(self, state: list) -> list:
        """
        :param: state - list of two ints representing [x, y] of the node in question - [10, 10] for example
        Note that the origin is in the UPPER LEFT of the grid 
        """
        if self.WIND_DIRECTION == 1: 
            # fire burns north
            up_wind_state = [state[0]+1, state[1]]
        elif self.WIND_DIRECTION == 4:
            up_wind_state = [state[0]-1, state[1]-1]

        return up_wind_state


    def get_phos_chek_nodes(self, plane_x: int, plane_y: int) -> list:  
        # all we have to do here is set the current node's phos_chek value to True and add it to the phos_check nodes
        x: int = int(plane_x/self.CELL_SIZE)
        y: int = int((plane_y/self.CELL_SIZE))
        curr_plane_node: Node = self.node_map[x][y]

        # we set the phos check to true here
        curr_plane_node.phos_chek = True

        self.phos_chek_nodes.append(curr_plane_node)

        return self.phos_chek_nodes

    def get_next_burning(self, currently_burning_nodes: list, first_ignition: bool) -> list:
        
        next_burning_nodes: list = []
        
        for node in currently_burning_nodes:

            node.fuel = 0
            node.burning = False
            self.burned_nodes.append(node)
            downwind_state = self.get_down_wind_state(state=node.state)

            # generate random number between 0 and 1
            r = random.uniform(0, 1)

            for neighbor_node in node.neighbors:
                
                if (neighbor_node.fuel != 0) and (neighbor_node.burning == False):
                    
                    if first_ignition:
                        if neighbor_node.phos_chek is False: 
                            next_burning_nodes.append(neighbor_node)
                            neighbor_node.burning = True
                    else:
                        # lets first handle nodes that are not in the wind's direction
                        if neighbor_node.state != downwind_state:
                            if neighbor_node.phos_chek is False: 
                                if r < 0.2:
                                    next_burning_nodes.append(neighbor_node)
                                    neighbor_node.burning = True

                        # now we can handle the down wind nodes 
                        if neighbor_node.state == downwind_state:
                            if neighbor_node.phos_chek is False: 
                                if r > 0.2:
                                    next_burning_nodes.append(neighbor_node)
                                    neighbor_node.burning = True

        return next_burning_nodes

    def make_score_box(self):
        window_name = 'Score_Box'
    
        # Start coordinate, here (5, 5)
        # represents the top left corner of rectangle
        start_point = (5, 5)
        # Ending coordinate, here (220, 220)
        # represents the bottom right corner of rectangle
        end_point = (220, 220)
        # Blue color in BGR
        color = (255, 0, 0)
        # Line thickness of 2 px
        thickness = 2
        # Using cv2.rectangle() method
        # Draw a rectangle with blue line borders of thickness of 2 px
        image = cv2.rectangle(image, start_point, end_point, color, thickness)

        return {'window_name': window_name, 'image': image}

    def print_results(self):
        s = '-'*50
        print(f'\n\n\n{s} \nTOTAL TIME TAKEN TO EXTINGUISH FIRE: {self.fire_time} MINUTES')
        print(f'FINAL SCORE: {self.curr_score} \n {s}')

    def win_focus(self):
        # Ugly trick to get the window in focus.
        # Opens an image in fullscreen and then back to normal window
        cv2.namedWindow("Wildfire Test", cv2.WINDOW_AUTOSIZE);
        board_states = np.zeros([self.BOARD_SIZE * self.CELL_SIZE, self.BOARD_SIZE * self.CELL_SIZE, 3])
        cv2.imshow("Wildfire Test", board_states);
        cv2.setWindowProperty("Wildfire Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
        cv2.waitKey(2000)
        cv2.setWindowProperty("Wildfire Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_AUTOSIZE)


    def generate_initial_nodes(self):

        # TODO we will get these maps using the CV2 library on geospatial images 
        fuel_remaining = np.ones([self.BOARD_SIZE, self.BOARD_SIZE])
        burn_speeds = np.random.rand(self.BOARD_SIZE, self.BOARD_SIZE)
        city_state = [int(self.BOARD_SIZE - 5), int(self.BOARD_SIZE - 5)]

        # initialize np.array that we will fill with node objects
        # all_nodes = np.zeros([BOARD_SIZE, BOARD_SIZE])

        all_nodes = []
        for i in range(self.BOARD_SIZE):
            row: list = [random.uniform(0, 1)] * self.BOARD_SIZE
            all_nodes.append(row)

        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                all_nodes[i][j] =  Node(state=[i, j], heuristic=math.dist([i, j], city_state), \
                                        burn_speed=burn_speeds[i][j], fuel=fuel_remaining[i][j], phos_chek=False, burning=False)

        # now that we have the node map without neighbor nodes defined, lets define those here: 
        for i in range(self.BOARD_SIZE):
            for j in range(self.BOARD_SIZE):
                all_nodes[i][j].neighbors = self.get_neighbors(node=all_nodes[i][j], node_map=all_nodes)

        return all_nodes

    def display(self, fire_time: int, curr_score: int, layer2_cache: np.ndarray, old_burned_nodes: list):

        # Create a blank image
        layer2 = layer2_cache.copy()

        # draw the fire, burned area, and plane on the second layer 
        # We can use this to display all of the currently burning states 

        # only iterate through the NEW burned nodes, not all the burned nodes (increases speed) 
        # new_burned_nodes: list = list(set(burned_nodes) - set(old_burned_nodes))
        # old_burned_nodes.extend(new_burned_nodes)

        if self.SHOW_BURNED_NODES: 
            for burned_node in self.burned_nodes:
                x = burned_node.state[0] * self.CELL_SIZE
                y = burned_node.state[1] * self.CELL_SIZE
                layer2[y:y + self.CELL_SIZE, x:x + self.CELL_SIZE] = [173, 220, 255, 200]
        # TODO ^ Speed this up in the future 

        # create the layer 2 cache so we don't need to iterate through thousands of burned nodes for the render
        layer2_cache = layer2.copy()

        # display the fire retardant nodes
        for phos_chek_node in self.phos_chek_nodes:
            x = phos_chek_node.state[0] * self.CELL_SIZE
            y = phos_chek_node.state[1] * self.CELL_SIZE
            layer2[y:y + self.CELL_SIZE, x:x + self.CELL_SIZE] = [255, 10, 10, 1]

        for burning_node in self.burning_nodes:
            x = burning_node.state[0] * self.CELL_SIZE
            y = burning_node.state[1] * self.CELL_SIZE
            layer2[y:y + self.CELL_SIZE, x:x + self.CELL_SIZE] = [0, 0, 255, 1]

        # display the airport 
        airport_x = self.airport.state[0] * self.CELL_SIZE
        airport_y = self.airport.state[1] * self.CELL_SIZE
        layer2[airport_y:airport_y + self.CELL_SIZE, airport_x:airport_x + self.CELL_SIZE] = [255,255,0, 255]

        # Display the plane  
        x = self.plane.state[0] * self.CELL_SIZE
        y = self.plane.state[1] * self.CELL_SIZE
        layer2[y:y + self.CELL_SIZE, x:x + self.CELL_SIZE] = [255, 255, 255, 255]
        
        if self.TRAIN_MODE: 
            res = layer2
        else:
            # copy the first layer into the resulting image
            res = np.uint8(self.layer1.copy()) 

            # copy the first layer into the resulting image
            cnd = layer2[:, :, 3] > 0

            # copy the first layer into the resulting image
            res[cnd] = layer2[cnd]

        # add the score to the image 
        pix: int = int(self.CELL_SIZE * self.BOARD_SIZE)
        cv2.rectangle(res, ((pix-300), 0), (pix, (125)), (211, 211, 211), -1)

        cv2.putText(res, text=f'Time: {fire_time} minutes', org=((pix-300), 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0),thickness=2)
        cv2.putText(res, text=f'Score: {curr_score}', org=((pix-300), 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0),thickness=2)

        # show the output image
        if not self.TRAIN_MODE:
            if self.CREATE_GIF:                
                cv2.imshow("Wildfire Environment", res)
                self.frames.append(res)
            else:
                cv2.imshow("Wildfire Environment", res)
            key = cv2.waitKey(int(self.SPEED))
        else:
            # cv2.imshow("Wildfire Environment", res)
            key = cv2.waitKey(1)

        # cv2.imshow("Wildfire Environment", np.uint8(board_states))
        # key = cv2.waitKey(int(1000/SPEED))

        # Return the key pressed. It is -1 if no key is pressed. 
        return {'key': key, 'plane_x': x, 'plane_y': y, 'layer2_cache': layer2_cache, 'old_burned_nodes': old_burned_nodes}

    def make_observation(self):
        """
        Generate the observation
        
        """
        self.observation = np.zeros(shape=(self.BOARD_SIZE, self.BOARD_SIZE, 1), dtype=np.uint8)

        # we set the burned states in the matrix to 1
        burned_states = [x.state for x in self.burned_nodes] 
        for burned_state in burned_states:
            self.observation[burned_state[0], burned_state[1]] = 1

        # we set the burning states in the matrix to 1
        burning_states = [x.state for x in self.burning_nodes] 
        for burning_state in burning_states:
            self.observation[burning_state[0], burning_state[1]] = 2

        # we set the phos chek states in the matrix to 1
        phos_chek_states = [x.state for x in self.phos_chek_nodes] 
        for phos_chek_state in phos_chek_states:
            self.observation[phos_chek_state[0], phos_chek_state[1]] = 3
        
        self.observation[self.plane.state[0], self.plane.state[1]] = 4  

        self.observation[self.airport.state[0], self.airport.state[1]] = 5

        return self.observation

    def calculate_reward(self):
        """
        Generate the reward after each step 
        """
        if self.done:
            if self.CREATE_GIF:
                # at this point we want to save our list of images as a gif
                print("Saving GIF file")
                with imageio.get_writer("trained_model_run.gif", mode="I") as writer:
                    for idx, frame in enumerate(self.frames):
                        # print("Adding frame to GIF file: ", idx + 1)
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        writer.append_data(rgb_frame)
            else:
                # get percentage of board burned
                percent_unburned = (self.curr_score / (self.BOARD_SIZE*self.BOARD_SIZE)) * 100
                # (reward_balancer * current_reward) + ((1 - reward_balancer) * percent_unburned_trees)
                self.reward = float(self.REWARD_BALANCER*self.reward + (1-self.REWARD_BALANCER) * percent_unburned)
        else:

            # if the plane has NO Phos Chek        
            if (self.plane.phos_chek_level == 0):
                # if the plane has NO phos chek and it's flying towards the airport 
                prev_dist_to_airport = dist(self.plane.previous_state, self.airport.state)
                curr_dist_to_airport = dist(self.plane.state, self.airport.state)
                
                # we want to reward the agent if it flies towards the airport here
                if curr_dist_to_airport < prev_dist_to_airport:
                    self.reward = self.reward + 1
                else:
                    self.reward = self.reward - 1

            else:
                # we now want to ensure that the plan is either dropping phos chek in the right place, or at least 
                # flying towards the fire if it has a non-zero level of phos chek
                # DROPPING PHOS CHEK
                if self.phos_check_dump: 
                    # we don't need to worry that the plane is dumping with no phos chek because that's the previous test-case
                    if (self.plane.previous_state not in [node.state for node in self.burning_nodes]) & \
                       (self.plane.previous_state not in [node.state for node in self.burned_nodes]):
                       
                        # get the neighboring nodes to the plane's current location
                        x: int = int(self.plane.previous_state[0])
                        y: int = int((self.plane.previous_state[1]))
                        curr_plane_node: Node = self.node_map[x][y]
                        
                        # if we want to check whether or not the up wind nodes are burning
                        if self.USE_UPWIND_NODE_FOR_REWARD: 

                            # find the up wind state first 
                            upwind_state = self.get_up_wind_state(curr_plane_node.state)
                            x: int = int(upwind_state[0])
                            y: int = int((upwind_state[1]))
                            # use the up wind node 
                            curr_plane_node: Node = self.node_map[x][y]

                        plane_neighbor_nodes: list = curr_plane_node.neighbors

                        # if any of the plane's current neighbord are burning, then we give a large reward 
                        l1 = [node.state for node in plane_neighbor_nodes]
                        l2 = [node.state for node in self.burning_nodes]
                        if lists_overlap(l1, l2):
                            # plane is dropping phos chek on a forest node that borders a burning node - Good! 
                            self.reward = self.reward + 5
                        else:
                            # plane is dropping phos chek on a forest node that DOES NOT border a burning node - Bad! 
                            self.reward = self.reward - 3
                    
                    else:
                        # if we dump the phos chek on burning or burned nodes, that earns a big penalty! 
                        self.reward = self.reward - 10

                # NOT DROPPING PHOS CHEK
                else:
                    # here we want to reward the plane if it's flying towards ANY part of the fire. 
                    # TODO vectorize this to make it faster in the future!!! 
                    flying_towards_any_burning_node: bool = False
                    burning_states = [x.state for x in self.burning_nodes] 
                    for burning_state in burning_states:
                        prev_dist_to_buring_state = dist(self.plane.previous_state, burning_state)
                        curr_dist_to_buring_state = dist(self.plane.state, burning_state)

                        # we want to reward the agent if it flies towards the fire here
                        if curr_dist_to_buring_state < prev_dist_to_buring_state:
                            self.reward = self.reward + 1
                            flying_towards_any_burning_node = True
                            break
                    
                    # here we penalize the agent if it's flying in a direction that's away from ALL the burning nodes
                    if not flying_towards_any_burning_node:
                        self.reward = self.reward - 1

        return self.reward


# some test code
if __name__ == "__main__":

    env = WildFireEnv(TRAIN_MODE=True, SHOW_IMAGE_BACKGROUND=False, SHOW_BURNED_NODES=False)
    episodes = 2

    for episode in range(episodes):
        done = False
        obs = env.reset()

        while done is False:#not done:
            random_action = env.action_space.sample()
            print("action",random_action)
            obs, reward, done, info = env.step(random_action)
            print('reward',reward, 'done', done)


# https://learnopencv.com/snake-game-with-opencv-python/

import cv2
import numpy as np
from random import randint
from random import choice
import math
import random
import time
import pickle

# it's very important that these simulations are repeatable - we rely on random.seed() for this
random.seed(10)

# define globals 
# Size of each cell in the board game
CELL_SIZE = 15
# Number of cells along the width in the game
BOARD_SIZE = 100
# Change SPEED to make the game go faster
SPEED = 15
# Maximum speed at which the fire advances
FIRE_SPEED = 15  # <-- inverse 
# the amount of time in actual minutes that go by before the fire advances one step (this needs to be calibrated realistically)
FIRE_TIMESTEP = int(FIRE_SPEED*3)  # minutes
# define max amount of Phos Chek that a plane can carry (will depend on type of aircraft)
MAX_PHOS_CHEK = 15
# Wind
WIND_DIRECTION = 4
wind_direction_lookup = {'north': 1, 'northeast': 2, 'east': 3, 'southeast': 4, 'south': 5, 'southwest': 6, 'west': 7, 'northwest': 8}
WIND_SPEED = 5
# train mode - true if we want the simulations to run fast and we don't care about aesthetics
TRAIN_MODE = True
# Show the burned nodes 
SHOW_BURNED_NODES = False


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
    def __init__(self, state, previous_state, phos_chek_level: int, direction: int):
        self.state = state                          # [x, y] position of the plane in the grid  
        self.previous_state = previous_state        # [x, y] position of the plane's previous position in the grid  
        self.phos_chek_level = phos_chek_level      # int representing the amount of Phos Chek left in the plane
        self.direction = direction

    def move(self):
        # update the previous state to the current state before we move the plane
        self.previous_state = self.state.copy()

        # Checks what its current direction is and moves accordingly
        if self.direction == 0:
            self.state[0] += 1
        elif self.direction == 1:
            self.state[1] += 1
        elif self.direction == 2:
            self.state[0] -= 1
        elif self.direction == 3:
            self.state[1] -= 1


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


def get_neighbors(node: Node, node_map: list):

    x = node.state[0]
    y = node.state[1]

    neighbor_states: list = [(x2, y2) for x2 in range(x-1, x+2)
                                for y2 in range(y-1, y+2)
                                if (-1 < x <= BOARD_SIZE and
                                    -1 < y <= BOARD_SIZE and
                                    (x != x2 or y != y2) and
                                    (0 <= x2 <= BOARD_SIZE) and
                                    (0 <= y2 <= BOARD_SIZE))]

    neighbor_nodes = []
    
    # perform the lookup in the node_map to get the neighbor nodes
    for state in neighbor_states:
        if (state[0] < BOARD_SIZE) and (state[1] < BOARD_SIZE):
            n_node = node_map[state[0]][state[1]]
            neighbor_nodes.append(n_node)

    return neighbor_nodes

def get_down_wind_state(state: list) -> list:
    """
    :param: state - list of two ints representing [x, y] of the node in question - [10, 10] for example
    Note that the origin is in the UPPER LEFT of the grid 
    """
    if WIND_DIRECTION == 1: 
        # fire burns north
        down_wind_state = [state[0]-1, state[1]]
    elif WIND_DIRECTION == 4:
        down_wind_state = [state[0]+1, state[1]+1]

    return down_wind_state


def get_phos_chek_nodes(plane_x: int, plane_y: int) -> list:
    # all we have to do here is set the current node's phos_chek value to True and add it to the phos_check nodes
    x: int = int(plane_x/CELL_SIZE)
    y: int = int((plane_y/CELL_SIZE))
    curr_plane_node: Node = node_map[x][y]

    # we set the phos check to true here
    curr_plane_node.phos_chek = True

    phos_chek_nodes.append(curr_plane_node)

    return phos_chek_nodes

def get_next_burning(currently_burning_nodes: list, first_ignition: bool) -> list:
    
    next_burning_nodes: list = []
    
    for node in currently_burning_nodes:

        node.fuel = 0
        node.burning = False
        burned_nodes.append(node)
        downwind_state = get_down_wind_state(state=node.state)

        # generate random number between 0 and 1
        r = random.uniform(0, 1)

        for neighbor_node in node.neighbors:
            
            if (neighbor_node.fuel != 0) and (neighbor_node.burning == False):
                
                if first_ignition:
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

def make_score_box():
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

def print_results(fire_time: int, curr_score: int):
    s = '-'*50
    print(f'\n\n\n{s} \nTOTAL TIME TAKEN TO EXTINGUISH FIRE: {fire_time} MINUTES')
    print(f'FINAL SCORE: {curr_score} \n {s}')

def win_focus():
    # Ugly trick to get the window in focus.
    # Opens an image in fullscreen and then back to normal window
    cv2.namedWindow("Wildfire Test", cv2.WINDOW_AUTOSIZE);
    board_states = np.zeros([BOARD_SIZE * CELL_SIZE, BOARD_SIZE * CELL_SIZE, 3])
    cv2.imshow("Wildfire Test", board_states);
    cv2.setWindowProperty("Wildfire Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN);
    cv2.waitKey(2000)
    cv2.setWindowProperty("Wildfire Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_AUTOSIZE)


def generate_initial_nodes():

    # TODO we will get these maps using the CV2 library on geospatial images 
    fuel_remaining = np.ones([BOARD_SIZE, BOARD_SIZE])
    burn_speeds = np.random.rand(BOARD_SIZE, BOARD_SIZE)
    city_state = [int(BOARD_SIZE - 5), int(BOARD_SIZE - 5)]

    # initialize np.array that we will fill with node objects
    # all_nodes = np.zeros([BOARD_SIZE, BOARD_SIZE])

    all_nodes = []
    for i in range(BOARD_SIZE):
        row: list = [random.uniform(0, 1)] * BOARD_SIZE
        all_nodes.append(row)

    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            all_nodes[i][j] =  Node(state=[i, j], heuristic=math.dist([i, j], city_state), \
                                    burn_speed=burn_speeds[i][j], fuel=fuel_remaining[i][j], phos_chek=False, burning=False)

    # now that we have the node map without neighbor nodes defined, lets define those here: 
    for i in range(BOARD_SIZE):
        for j in range(BOARD_SIZE):
            all_nodes[i][j].neighbors = get_neighbors(node=all_nodes[i][j], node_map=all_nodes)

    return all_nodes


def display(fire_time: int, curr_score: int, layer2_cache: np.ndarray, old_burned_nodes: list):

    # Create a blank image
    layer2 = layer2_cache.copy()

    # draw the fire, burned area, and plane on the second layer 
    # We can use this to display all of the currently burning states 

    # only iterate through the NEW burned nodes, not all the burned nodes (increases speed) 
    # new_burned_nodes: list = list(set(burned_nodes) - set(old_burned_nodes))
    # old_burned_nodes.extend(new_burned_nodes)

    if SHOW_BURNED_NODES: 
        for burned_node in burned_nodes:
            x = burned_node.state[0] * CELL_SIZE
            y = burned_node.state[1] * CELL_SIZE
            layer2[y:y + CELL_SIZE, x:x + CELL_SIZE] = [173, 220, 255, 255]
    # TODO ^ Speed this up in the future 

    # create the layer 2 cache so we don't need to iterate through thousands of burned nodes for the render
    layer2_cache = layer2.copy()

    for burning_node in burning_nodes:
        x = burning_node.state[0] * CELL_SIZE
        y = burning_node.state[1] * CELL_SIZE
        layer2[y:y + CELL_SIZE, x:x + CELL_SIZE] = [0, 0, 255, 255]

    # display the fire retardant nodes
    for phos_chek_node in phos_chek_nodes:
        x = phos_chek_node.state[0] * CELL_SIZE
        y = phos_chek_node.state[1] * CELL_SIZE
        layer2[y:y + CELL_SIZE, x:x + CELL_SIZE] = [255, 10, 10, 255]

    # display the airport 
    airport_x = airport.state[0] * CELL_SIZE
    airport_y = airport.state[1] * CELL_SIZE
    layer2[airport_y:airport_y + CELL_SIZE, airport_x:airport_x + CELL_SIZE] = [255,255,0, 255]

    # Display the plane  
    x = plane.state[0] * CELL_SIZE
    y = plane.state[1] * CELL_SIZE
    layer2[y:y + CELL_SIZE, x:x + CELL_SIZE] = [255, 255, 255, 255]
    
    if TRAIN_MODE: 
        res = layer2
    else:
        # copy the first layer into the resulting image
        res = np.uint8(layer1.copy()) 

        # copy the first layer into the resulting image
        cnd = layer2[:, :, 3] > 0

        # copy the first layer into the resulting image
        res[cnd] = layer2[cnd]

    # add the score to the image 
    pix: int = int(CELL_SIZE * BOARD_SIZE)
    cv2.rectangle(res, ((pix-300), 0), (pix, (125)), (211, 211, 211), -1)

    cv2.putText(res, text=f'Time: {fire_time} minutes', org=((pix-300), 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0),thickness=2)
    cv2.putText(res, text=f'Score: {curr_score}', org=((pix-300), 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0),thickness=2)

    # show the output image
    cv2.imshow("Wildfire Environment", res)
    key = cv2.waitKey(int(1000/SPEED))

    # cv2.imshow("Wildfire Environment", np.uint8(board_states))
    # key = cv2.waitKey(int(1000/SPEED))

    # Return the key pressed. It is -1 if no key is pressed. 
    return {'key': key, 'plane_x': x, 'plane_y': y, 'layer2_cache': layer2_cache, 'old_burned_nodes': old_burned_nodes}


if __name__ == '__main__' : 

    # list of lists representing the board of all nodes 
    node_map: list = generate_initial_nodes()

    plane_start_state = [int((BOARD_SIZE - 1)/2), int((BOARD_SIZE - 1)/2)]
    # plane starts at the center of the board. 
    plane = Plane(state=plane_start_state, previous_state=plane_start_state, phos_chek_level=MAX_PHOS_CHEK, direction=1)

    # initislize fire start location
    fire_start_state = [(2, 7), (8, 3)]  # [(10, 10), (15, 55)]  # <-- good values for larger boards
    # node_map[fire_start_state[0]][fire_start_state[1]].burning = True
    burning_nodes: list = [node_map[fire_start_state[0][0]][fire_start_state[0][1]], 
                           node_map[fire_start_state[1][0]][fire_start_state[1][1]]]
    # burning_nodes: list = [node_map[fire_start_state[0][0]][fire_start_state[0][1]]]

    airport = Airport(state=[int((BOARD_SIZE - 3)), int((BOARD_SIZE - 3))])

    # for i in range(board_start_state.shape[0]):
    #     for j in range(board_start_state.shape[1]): 
    #         # TODO we will eventually set up the RGB of the board depending on the fuel in each node
    #         board_start_state[i][j] = [34,139,34]  # <-- texturize this

    background_image = cv2.imread('/Users/spencerbertsch/Desktop/dev/RL-sandbox/src/images/occidental_vet_hospital.png')
    layer1 = np.zeros([background_image.shape[0], background_image.shape[1], 4])

    if not TRAIN_MODE:
        for i in range(layer1.shape[0]):
            for j in range(layer1.shape[1]): 
                # TODO we will eventually set up the RGB of the board depending on the fuel in each node
                layer1[i][j] = np.uint8(np.append(background_image[i][j], 255))
        # TODO ^^^ Speed up this step - for training we can skip this

    # Start the game by printing instructions
    print('w = top, a = left, s = down, d = right, p = exit the game')

    # Ugly trick to bring the window in focus
    win_focus()

    # define blackened nodes (already burned)
    burned_nodes = []
    # define retardant nodes (PHOS-CHEK already dropped here)
    phos_chek_nodes = []
    board_start_state = np.zeros([BOARD_SIZE * CELL_SIZE, BOARD_SIZE * CELL_SIZE, 4])
    # use this cache to speed up the rendering of the layered image
    layer2_cache = board_start_state.copy()
    old_burned_nodes = []

    # define other parameters for this run 
    c = 0
    fire_time = 0
    curr_score: int = BOARD_SIZE*BOARD_SIZE
    first_ignition = True
    phos_check_dump = False
    step_times = []
    while True:
        t = time.time()
        
        # Makes and displays the board_states
        display_dict: dict = display(fire_time=fire_time, curr_score=curr_score, 
                                     layer2_cache=layer2_cache, old_burned_nodes=old_burned_nodes)
        key = display_dict['key']
        old_burned_nodes = display_dict['old_burned_nodes']
        layer2_cache = display_dict['layer2_cache']

        # Gets key presses and moves accordingly
        # 8 and 27 are delete and escape keys
        # Arrow keys are tricky in OpenCV. So we use
        # keys 'w', 'a','s','d' for movement. 
        # w = top, a = left, s = down, d = right

        if key == 8 or key == 27:
            break
        elif key == ord("d"):
            plane.direction = 0
        elif key == ord("s"):
            plane.direction = 1
        elif key == ord("a"):
            plane.direction = 2
        elif key == ord("w"):
            plane.direction = 3
        elif key == ord("b"):
            # activate airal "attack"
            phos_check_dump = not phos_check_dump
        elif key == ord("p"): 
            quit = True
            break

        # Moving the plane
        plane.move()    

        if phos_check_dump:
            if plane.phos_chek_level > 0:
                # we are in a time of dumping the phos_chek so we add this node to the phos_chek nodes
                phos_chek_nodes = get_phos_chek_nodes(plane_x=display_dict['plane_x'], plane_y=display_dict['plane_y'])
                # reduce Phos Chek level in the plane
                plane.phos_chek_level = plane.phos_chek_level - 1

        if plane.state == airport.state: 
            # plane is stopping at the airport
            phos_check_dump = False
            plane.phos_chek_level = MAX_PHOS_CHEK

        # cause the fire to spread either deterministically or via a stochastic function 
        if c == FIRE_SPEED: 
            if len(burning_nodes) == 0:
                quit = True
                print_results(fire_time=fire_time, curr_score=curr_score)
                break
            else:
                burning_nodes = get_next_burning(burning_nodes, first_ignition=first_ignition)
                first_ignition=False
                c = 0
                # increment the clock 
                fire_time = fire_time + FIRE_TIMESTEP
                curr_score = (BOARD_SIZE*BOARD_SIZE) - len(burned_nodes)
        else:
            c += 1

        step_time: float = round(time.time() - t, 5)
        print(step_time)
        step_times.append(step_time)

    # USE THIS FOR AD-HOC PERFORMANCE TESTING
    # with open('/Users/spencerbertsch/Desktop/dev/RL-sandbox/src/gridworld/metadata/burned_nodes.pkl', 'wb') as f:
    #     pickle.dump(step_times, f)

"""
----------------- TODOs -----------------

1. Keep the plane inside the environment 
2. add new section of reward so that if you drop phos-chek down wind of fire it's a big reward
 ++ if you drop phos chek away from the fire its a big penalty 

"""

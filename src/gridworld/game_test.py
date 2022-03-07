
import numpy as np
import cv2
import random
import time


class Plane:
    """
    Plane (agent) for the wild fire RL environment 
    """
    def __init__(self, state, jell_foam_dropped: bool):
        self.state = state                              # [x, y] position of the plane in the grid 
        self.jell_foam_dropped = jell_foam_dropped      # bool representing whether or not the watery jell-foam has been dropped


class Node:
    """
    Forest Node - node for the wild fire RL envoironment 
    """

    def __init__(self, state, heuristic, fuel=1, burn_speed=None, jell_foam=None, neighbors=None):
        self.state = state              # [x, y] position of the node in the grid 
        self.heuristic = heuristic      # euclidean distance to closest human inhabited area 
        self.neighbors = neighbors      # neighboring (adjacent) nodes in the grid 
        self.burn_speed = burn_speed    # {1: slow (sandy area), 2: moderate (wet area), 3: fast (dry forest)}
        self.fuel = fuel                # amount of remaining fuel in this node 
        self.jell_foam = jell_foam      # jell_foam concentration in this cell


def get_next_burning(currently_burning_nodes: list):
    
    next_burning_nodes: list = []
    
    for node in currently_burning_nodes:
        neighbor_nodes: list = get_neighbor_nodes(node)

        for neighbor_node in neighbor_nodes:
            #  we can make this much more sophistocated later - for now let's make it binary
            if neighbor_node.fuel != 0:
                next_burning_nodes.append(adjacent_node)

    return next_burning_nodes



def collision_with_apple(apple_position, score):
    apple_position = [random.randrange(1,50)*10,random.randrange(1,50)*10]
    score += 1
    return apple_position, score

def collision_with_boundaries(snake_head):
    if snake_head[0]>=500 or snake_head[0]<0 or snake_head[1]>=500 or snake_head[1]<0 :
        return 1
    else:
        return 0

def collision_with_self(snake_position):
    snake_head = snake_position[0]
    if snake_head in snake_position[1:]:
        return 1
    else:
        return 0


def main(fire_start_state: list):

    img = np.zeros((500,500,3),dtype='uint8')

    # Initial Fire Start Position
    fire_start_state = [250,250]
    plane_start_state = [random.randrange(1,50)*5,random.randrange(1,50)*5]
    score = 0
    prev_button_direction = 1
    button_direction = 1

    while True:
        cv2.imshow('a',img)
        cv2.waitKey(1)
        img = np.zeros((500,500,3),dtype='uint8')
        
        # Display Plane
        cv2.rectangle(img,(plane_start_state[0],plane_start_state[1]),(plane_start_state[0]+10,plane_start_state[1]+10),(0,0,255),3)
        
        # Display Fire
        for position in [fire_start_state]:  # <-- fix this so we iterate through a list of nodes 
            cv2.rectangle(img,(position[0],position[1]),(position[0]+10,position[1]+10),(0,255,0),3)

        # FIXME do this vvv
        # for node in burning_nodes:
            # position: tuple = node.state
            # cv2.rectangle(img,(position[0],position[1]),(position[0]+10,position[1]+10),(0,255,0),3)
        
        # Takes step after fixed time
        # TODO utilize this functionality for the fire! 
        t_end = time.time() + 0.1
        k = -1
        while time.time() < t_end:
            if k == -1:
                k = cv2.waitKey(125)
            else:
                continue
                
        # 0-Left, 1-Right, 3-Up, 2-Down, q-Break
        # a-Left, d-Right, w-Up, s-Down

        if k == ord('a') and prev_button_direction != 1:
            button_direction = 0
        elif k == ord('d') and prev_button_direction != 0:
            button_direction = 1
        elif k == ord('w') and prev_button_direction != 2:
            button_direction = 3
        elif k == ord('s') and prev_button_direction != 3:
            button_direction = 2
        elif k == ord('q'):
            break
        else:
            button_direction = button_direction
        prev_button_direction = button_direction

        # Move the plane around the environment
        if button_direction == 1:
            plane_start_state[0] += 10
        elif button_direction == 0:
            plane_start_state[0] -= 10
        elif button_direction == 2:
            plane_start_state[1] += 10
        elif button_direction == 3:
            plane_start_state[1] -= 10

        # # Increase Snake length on eating apple
        # if snake_head == apple_position:
        #     apple_position, score = collision_with_apple(apple_position, score)
        #     snake_position.insert(0,list(snake_head))

        # else:
        #     snake_position.insert(0,list(snake_head))
        #     snake_position.pop()
        
        # # On collision kill the snake and print the score
        # if collision_with_boundaries(snake_head) == 1 or collision_with_self(snake_position) == 1:
        #     font = cv2.FONT_HERSHEY_SIMPLEX
        #     img = np.zeros((500,500,3),dtype='uint8')
        #     cv2.putText(img,'Your Score is {}'.format(score),(140,250), font, 1,(255,255,255),2,cv2.LINE_AA)
        #     cv2.imshow('a',img)
        #     cv2.waitKey(0)
        #     cv2.imwrite('D:/downloads/ii.jpg',img)
        #     break
            
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(fire_start_state = [350,3ss50])

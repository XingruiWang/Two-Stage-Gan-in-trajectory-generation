import cv2 as cv
import numpy as np
import random


def transform(enter_points, exit_points, w, h):
    '''
    e.g. (0, h - 1) --> (1, 1); (w - 1, 0) --> (-1, -1)
    '''
    enter_points = enter_points[:, ::-1]
    exit_points = exit_points[:, ::-1]
    enter_points = enter_points * \
        np.array([[2/(h-1), 2/(1-w)]]) + np.array([[-1.0, 1.0]])
    exit_points = exit_points * \
        np.array([[2/(h-1), 2/(1-w)]]) + np.array([[-1.0, 1.0]])
    return enter_points, exit_points

def get_position(side):
    '''
    return the indexes of road points
    '''
    WHITE = (255, 255, 255)
    YELLOW = (255, 222, 109)
    LIGHT_YELLOW = (255, 236, 186)
    points = []
    l, c = side.shape
    b, g, r = side[:, 0], side[:, 1], side[:, 2]
    points = [i for i in range(l) if (r[i], g[i], b[i]) == WHITE]
    points += [i for i in range(l) if (r[i], g[i], b[i]) == YELLOW]
    points += [i for i in range(l) if (r[i], g[i], b[i]) == LIGHT_YELLOW]
    if len(points) == 0: ### # In case points is empty
        points = [int(l/2)]
    return points

def get_point(map_grid, direction):
    '''
    return all the road points on the given side
    '''
    side = None
    w, h, c = map_grid.shape
    if direction == 'left':
        side = map_grid[:, 0, :]
        x_position = get_position(side)
        points = [[x, 0] for x in x_position]
    elif direction == 'right':
        side = map_grid[:, h-1, :]
        x_position = get_position(side)
        points = [[x, h-1] for x in x_position]
    elif direction == 'up':
        side = map_grid[0, :, :]
        y_position = get_position(side)
        points = [[0, y] for y in y_position]
    elif direction == 'down':
        side = map_grid[w-1, :, :]
        y_position = get_position(side)
        points = [[w-1, y] for y in y_position]
    assert(len(points) != 0) # In case points is empty
    return np.array(points)

def get_all_enter_exit_point(map_grids, enter_direction, exit_direction, downsample=1.0):
    '''
    return all the points on the road in the enter direction & exit direction
    '''
    w, h, c = map_grids.shape
    if downsample < 1:
        w, h = int(w*downsample), int(h*downsample)
        map_grids = cv.resize(map_grids, (h, w))
    enter_points = get_point(map_grids, enter_direction)
    exit_points = get_point(map_grids, exit_direction)
    enter_points, exit_points = transform(enter_points, exit_points, w, h)
    return enter_points, exit_points


def get_random_enter_exit_point(map_grid, enter_direction, exit_direction, downsample=1.0, last_exit_point=None):
    '''
    return the enter point & exit point
    '''
    enter_points, exit_points = get_all_enter_exit_point(
        map_grid, enter_direction, exit_direction, downsample)
    if last_exit_point is None:
        enter_point = enter_points[random.sample(range(enter_points.shape[0]), 1)]
    else:
        if enter_direction == 'left':
            enter_point = np.array([[-1, last_exit_point[0,1]]])
        elif enter_direction == 'right':
            enter_point = np.array([[1, last_exit_point[0,1]]])
        elif enter_direction == 'up':
            enter_point = np.array([[last_exit_point[0,0], 1]])
        elif enter_direction == 'down':
            enter_point = np.array([[last_exit_point[0,0], -1]])
    exit_point = exit_points[random.sample(range(exit_points.shape[0]), 1)] #shape=(1,2)
    return enter_point, exit_point


def grid(map_whole):
    '''
    return a dict of map grids
    '''
    map_dict = {}
    w, h, c = map_whole.shape
    wn = int(w/32)
    hn = int(h/32)
    for i in range(32):
        for j in range(32):
            map_dict['%d_%d' % (i, j)] = map_whole[i *
                                                       wn:i*wn+wn, j*hn:j*hn+hn, :]
    return map_dict


if __name__ == '__main__':
    map_whole = cv.imread('test.png')
    map_grids = grid(map_whole)  # map_grids[point] point = (x,y)
    # print(map_grids['1_2'])
    enter_point, exit_point = get_random_enter_exit_point(
        map_grids['1_2'], 'left', 'down', downsample=0.2)
    print(enter_point, exit_point)

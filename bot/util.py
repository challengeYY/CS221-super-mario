from enum import *
import numpy as np

def out_of_frame(x,y):
    return x < 0 or y < 0 or x >= Window.Width or y >= Window.Height

# get tile at (mario.x + step_x, mario.y + step_y)
# where positive step_x and step_y indicates right and up
# obs is 13 x 16 (y, x) with (0,0) on top left corner
def get_tile_from_mario(obs, step_x, step_y):
    ys, xs = np.where(obs == Tile.MARIO)
    if len(xs) == 0:
        return None
    mariox = xs[0]
    marioy = ys[0]
    focusx = mariox + step_x
    focusy = marioy - step_y
    if out_of_frame(focusx, focusy):
        return None
    return obs[focusx, focusy]

def get_reward(state):
    obs, reward, is_finished, info = state
    return reward

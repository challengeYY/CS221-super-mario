from enum import *
import numpy as np
import heapq


def out_of_frame(x, y):
    return x < 0 or y < 0 or x >= Window.Width or y >= Window.Height


# get tile at (mario.x + step_x, mario.y + step_y)
# where positive step_x and step_y indicates right and up
# obs is 13 x 16 (y, x) with (0,0) on top left corner
def get_tile_from_mario(obs, step_x, step_y):
    focusy, focusx = get_coord_from_mario(obs, step_x, step_y)
    return obs[focusy, focusx]


def get_mario_coord(obs):
    ys, xs = np.where(obs == Tile.MARIO)
    if len(xs) == 0:
        return None
    mariox = xs[0]
    marioy = ys[0]
    return marioy, mariox


def get_coord_from_mario(obs, step_x, step_y):
    coord = get_mario_coord(obs)
    if coord is None:
        return None
    marioy, mariox = coord
    focusx = mariox + step_x
    focusy = marioy - step_y
    if out_of_frame(focusx, focusy):
        return None
    return focusy, focusx


class GameState(object):
    def __init__(self, frames, prev_actions):
        self.frames = frames
        self.prev_actions = prev_actions

    def get_frames(self):
        return self.frames

    def get_last_frame(self):
        return self.frames[-1]

    def get_prev_actions(self):
        return self.prev_actions

    def get_last_n_obs(self, n=1):
        obs = [[np.zeros([Window.Width, Window.Height])]] * (n - len(self.frames))
        for i in range(min([n, len(self.frames)])):
            obs.append(self.frames[-1 - i].get_obs())

        return obs


class GameFrame(object):
    def __init__(self, obs, reward, is_finished, info):
        self.obs = obs
        self.reward = reward
        self.is_finished = is_finished
        self.info = info

    def get_reward(self):
        return self.reward

    def set_reward(self, newReward):
        self.reward = newReward
        return self

    def get_obs(self):
        return self.obs

    def get_is_finished(self):
        return self.is_finished

    def get_info(self):
        return self.info


# get mario's velocity from state1 to state2
def get_velocity(state1, state2):
    # 13 x 16 numpy array
    obs1 = state1.get_obs()
    obs2 = state2.get_obs()
    min_diff = float('inf')
    for v in range(3):
        shifted_obs1 = np.roll(obs1, v, axis=1)
        shifted_obs1[:, :v] = obs2[:, :v]
        diff = np.sum(np.abs(shifted_obs1 - obs2))
        if diff < min_diff:
            min_diff = diff
            min_v = v
    return v


def get_death_penalty_value(time, distance):
    return (time - Time.TOTAL_GAME_TIME) - distance / 3


class ExplorationRate:
    def __init__(self, maxExplorationRate, minExplorationRate, percentile=0.9, growthFactor=200.0):
        self.maxExplorationRate = maxExplorationRate
        self.minExplorationRate = minExplorationRate
        self.percentile = percentile
        self.growthFactor = growthFactor
        self.midPoint = 0.0

    def updateFunction(self, gameStates):
        def extractDistance(gameState):
            distance = int(gameState[-1].get_last_frame().get_info()['distance'])
            return distance

        totalGames = len(gameStates) - 1
        topn = int(np.ceil((totalGames * (1 - self.percentile))))
        topn_list = heapq.nlargest(topn, gameStates[:-1], extractDistance)
        self.midPoint = float(topn_list[-1][-1].get_last_frame().get_info()['distance'])
        return

    def getExplorationRate(self, distance):
        mag = (self.maxExplorationRate - self.minExplorationRate) / 2.0
        center = (self.maxExplorationRate + self.minExplorationRate) / 2.0
        return mag * (np.arctan((distance - self.midPoint) / self.growthFactor) / (np.pi / 2)) + center

from pynput import keyboard
from abc import ABCMeta, abstractmethod
import traceback
from enum import *
from util import *


class FeatureExtractor:
    __metaclass__ = ABCMeta

    # Agent's initial action
    @abstractmethod
    def featureSize(self): pass

    @abstractmethod
    def extract(self, feature, state): pass


class TileFeatureExtractor(FeatureExtractor):
    def featureSize(self):
        raise Exception("Cannot get featureSize for TileFeatureExtractor")

    def extract(self, feature, state):
        tiles = []
        obs = state.get_last_frame().get_obs()
        tiles.append(obs)
        return np.array(np.transpose(tiles, (2, 1, 0)))


class InfoFeatureExtractor(FeatureExtractor):
    def featureSize(self): return 4

    def extract(self, feature, state):
        info = state.get_last_frame().get_info()
        feature['distance'] = info['distance']
        feature['coins'] = info['coins']
        feature['player_status'] = info['player_status']
        feature['time'] = info['time']


class VelocityFeatureExtractor(FeatureExtractor):
    def featureSize(self): return 1

    def extract(self, feature, state):
        # extract velocity
        dist_delta = state.get_last_frame().get_info()['distance'] - state.get_frames()[0].get_info()['distance']
        v = dist_delta * 1.0 / len(state.get_frames())
        feature['velocity'] = v


class MarioFeatureExtractor(FeatureExtractor):
    def featureSize(self):
        return 3

    def extract(self, feature, state):
        # extract x, y coordinate of Mario
        obs = state.get_last_frame().get_obs()
        coord = get_mario_coord(obs)
        if coord is None:
            marioy, mariox = (0, 0)
            feature['has_mario'] = 0
        else:
            marioy, mariox = coord
            feature['has_mario'] = 1
        feature['marioy'] = marioy
        feature['mariox'] = mariox


class FrontFeatureExtractor(FeatureExtractor):
    def featureSize(self):
        return 4

    def extract(self, feature, state):
        # extract information about object in front of MARIO
        obs = state.get_last_frame().get_obs()
        for i in range(1, 5):
            horizontal_tile = get_coord_from_mario(obs, i, 0)
            if horizontal_tile is None:
                feature['ahead_{}_height'.format(i)] = 0
            else:
                hor_y, hor_x = horizontal_tile
                feature['ahead_{}_height'.format(i)] = Window.Height
                for j in range(Window.Height):
                    if obs[hor_y - j, hor_x] == Tile.EMPTY_SPACE:
                        feature['ahead_{}_height'.format(i)] = j
                        break


class BehindFeatureExtractor(FeatureExtractor):
    def featureSize(self):
        return 4

    def extract(self, feature, state):
        # extract information about object behind MARIO
        obs = state.get_last_frame().get_obs()
        for i in range(1, 5):
            horizontal_tile = get_coord_from_mario(obs, -i, 0)
            if horizontal_tile is None:
                feature['behind_{}_height'.format(i)] = 0
            else:
                hor_y, hor_x = horizontal_tile
                feature['behind_{}_height'.format(i)] = Window.Height
                for j in range(Window.Height):
                    if obs[hor_y - j, hor_x] == Tile.EMPTY_SPACE:
                        feature['behind_{}_height'.format(i)] = j
                        break


class PitFeatureExtractor(FeatureExtractor):
    def featureSize(self):
        return 1

    def extract(self, feature, state):
        # extract whether there is a pit within distance of 3
        feature['pit_ahead'] = 0
        obs = state.get_last_frame().get_obs()
        coord = get_mario_coord(obs)
        if coord is not None:
            marioy, mariox = coord
            for i in range(1, 4):
                if not out_of_frame(mariox + i, marioy + 1) and obs[marioy + 1, mariox + i] == Tile.EMPTY_SPACE:
                    feature['pit_ahead'] = 1
                    break

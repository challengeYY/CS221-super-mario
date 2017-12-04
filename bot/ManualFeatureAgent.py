from collections import OrderedDict
from nn.q_model import *
from QLearnAgent import *
import numpy as np
from FeatureExtractor import *


class ManualFeatureAgent(QLearnAgent):
    def __init__(self, options, env):
        super(ManualFeatureAgent, self).__init__(options, env)
        self.featureExtractors = []
        self.featureExtractors.append(InfoFeatureExtractor())
        self.featureExtractors.append(VelocityFeatureExtractor())
        self.featureExtractors.append(MarioFeatureExtractor())
        self.featureExtractors.append(FrontFeatureExtractor())
        self.featureExtractors.append(PitFeatureExtractor())
        self.featureExtractors.append(BehindFeatureExtractor())

        featureSize = sum([fe.featureSize() for fe in self.featureExtractors])
        print('featureSize', featureSize)
        self.model = QModel(
            featureSize,
            num_actions=len(self.actions),
            tile_row=1,
            tile_col=1,
            window_size=self.windowsize,
            optimizer='adam',
            lr=0.01,
            decay_step=1000,
            decay_rate=0,
            regularization=0.01,
            conv=False,
            model_dir=options.model_dir
        )
        self.model.initialize_model(options.model_dir)
        self.algo.set_model(self.model)

    def featureExtractor(self, window):
        feature = OrderedDict()
        for fe in self.featureExtractors:
            fe.extract(feature, window)
        return feature.values()
        # obs: 13 x 16 numpy array (y, x). (0, 0) is the top left corner

        # info dict
        # A value of -1 indicates that the value is unknown
        # distance = info['distance'] # Total distance from the start (x-axis)
        # level = info['level']
        # coins = info['coins'] # The current number of coins
        # player_status = info['player_status'] # Indicates if Mario is small (value of 0), big (value of 1), or can shoot fireballs (2+)
        # score = info['score'] # The current score
        # time = info['time'] # # The current time left
        # ignore = info['ignore'] # Will be added with a value of True if the game is stuck and is terminated early
        # def featureExtractor(self, window):
        # last_state = window[-1]
        # # extract velocity
        # dist_delta = get_info(last_state)['distance'] - get_info(window[0])['distance']
        # v = dist_delta * 1.0 / len(window)
        # # extract x, y coordinate of Mario
        # obs = get_obs(last_state)
        # y, x = np.where(obs == Tile.MARIO)
        # if len(y) == 0:
        # return None
        # y = y[0]
        # x = x[0]
        # # extract information about object in front of MARIO
        # obj = OrderedDict()
        # for i in range(1, 5):
        # horizontal_tile = get_coord_from_mario(obs, i, 0)
        # if horizontal_tile is None:
        # obj[i] = 0
        # else:
        # hor_y, hor_x = horizontal_tile
        # for j in range(5):
        # if obs[hor_y-j, hor_x] == Tile.EMPTY_SPACE:
        # obj[i] = j
        # break
        # feature = obj.values()
        # feature.append(v)
        # feature.append(y)
        # # extract whether there is a pit within distance of 3
        # pit = 0
        # for i in range(1, 4):
        # if not out_of_frame(x+i,y+1) and obs[y+1, x+i] == Tile.EMPTY_SPACE:
        # pit = 1
        # break
        # feature.append(pit)
        # print "obs:", obs
        # print "feature:", feature
        # return np.array(feature)

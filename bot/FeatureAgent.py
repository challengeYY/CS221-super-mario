from collections import OrderedDict
import numpy as np
from nn.q_model import *
from QLearnAgent import *
from FeatureExtractor import *


class FeatureAgent(QLearnAgent):
    def __init__(self, options, env):
        super(FeatureAgent, self).__init__(options, env)
        self.featureExtractors = []
        self.featureExtractors.append(InfoFeatureExtractor())
        self.featureExtractors.append(VelocityFeatureExtractor())
        self.featureExtractors.append(MarioFeatureExtractor())
        self.featureExtractors.append(FrontFeatureExtractor())
        self.featureExtractors.append(PitFeatureExtractor())
        self.featureExtractors.append(BehindFeatureExtractor())
        self.featureExtractors.append(PrevActionsFeatureExtractor(self.prevActionsSize))

        self.tileFeatureExtractor = TileFeatureExtractor(options)

        featureSize = sum([fe.featureSize() for fe in self.featureExtractors])
        tile_row, tile_col, window_size = self.tileFeatureExtractor.featureSize()
        self.model = QModel(
            options=options,
            info_size=featureSize,
            tile_row=tile_row,
            tile_col=tile_col,
            window_size=window_size,
<<<<<<< HEAD
            num_actions=len(self.actions)
=======
            num_actions=len(self.actions),
            optimizer='adam',
            lr=1e-4,
            decay_step=1000,
            decay_rate=1,
            regularization=0.005,
            model_dir=options.model_dir
>>>>>>> ca13db72515ae9c8475c6a66091957db7c897c1e
        )
        self.algo.set_model(self.model)

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
    def featureExtractor(self, window):
        feature = OrderedDict()
        for fe in self.featureExtractors:
            fe.extract(feature, window)
        tiles = self.tileFeatureExtractor.extract(OrderedDict(), window)
        return tiles, feature.values()

        # feature = []
        # tiles = []
        # last_state = window[-1]
        # info = get_info(last_state)
        # feature.append(info['distance'])
        # feature.append(info['coins'])
        # feature.append(info['player_status'])
        # feature.append(info['time'])
        # for state in window:
        # obs = get_obs(state)
        # tiles.append(obs)
        # return np.array(np.transpose(tiles, (2, 1, 0))), np.array(feature)

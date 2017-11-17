from nn.q_model import *
from QLearnAgent import *

class ManualFeatureAgent(QLearnAgent):
    def __init__(self, options, env):
        super(FeatureAgent, self).__init__(options, env)
        featureSize = # TODO
        print('featureSize', featureSize)
        self.model = QModel(
            state_size=featureSize,
            num_actions=len(self.actions),
            optimizer='adam',
            lr=0.01,
            decay_step=1000,
            decay_rate=0,
            regularization=0.01
        )
        self.model.initialize_model(options.model_dir)
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
        # extract velocity
        dist_delta = get_info(window[-1])['distance'] - get_info(window[0])['distance']
        v = dist_delta * 1.0 / len(window)

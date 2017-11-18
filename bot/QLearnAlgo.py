from numpy import random
import operator
from util import *


class QLearningAlgorithm():
    def __init__(self, options, actions, discount, featureExtractor, windowsize, explorationRate):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        # self.batchsize = 10 # number of frames to retrain the model
        self.batchsize = 1
        self.windowsize = windowsize  # number of frames to look back in a state
        self.statecache = []
        self.actioncache = []
        self.batchcounter = 0
        self.options = options
        self.model = None
        self.explorationRate = explorationRate

    def set_model(self, model):
        self.model = model

    # Return the Q function associated with the weights and features
    def getQ(self, window):
        tile, info = self.featureExtractor(window)
        scores = self.model.inference_Q([tile], [info])[0]
        return scores

    def getProb(self, window):
        tile, info = self.featureExtractor(window)
        scores = self.model.inference_Prob([tile], [info])[0]
        return scores

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        rand = random.random()
        actionIdx = 0
        if random.random() < self.explorationRate:
            actionIdx = random.choice(range(len(self.actions)))
            print "randomly select action id: {}".format(actionIdx)
        elif len(self.statecache) < self.windowsize:
            actionIdx = self.actions.index('Right')
        else:
            window = []
            if self.windowsize > 1:
                window = self.statecache[-self.windowsize + 1:] + [state]
            else:
                window = [state]
            allQs = self.getQ(window)
            actionIdx, q = max(enumerate(allQs), key=operator.itemgetter(1))
            print "Q: {} best action id: {}".format(allQs, actionIdx)
        return Action.act(self.actions[actionIdx]), actionIdx

    # Call this function to get the step size to update the weights.

    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, action_idx, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        self.batchcounter += 1

        if self.batchcounter >= self.batchsize:
            self.batchcounter = 0
            gamma = self.discount

            tiles = []
            infos = []
            actions = []
            Y = []
            # for i in range(1, self.batchsize+1):
            #     window = self.statecache[-self.windowsize-i:-i]
            #     if len(window) < self.windowsize: continue
            #     X.append(self.featureExtractor(window, action))
            #     reward = get_reward(self.statecache[-i])
            #     Vopt = max([self.getQ(window, a) for a in self.actions(newState)])
            #     target = (reward + gamma * Vopt)
            #     Y.append(target)

            # try batchsize = 1
            window = self.statecache[-self.windowsize:]
            tile, info = self.featureExtractor(window)
            tiles.append(tile)
            infos.append(info)
            actions.append(action_idx)
            reward = get_reward(newState)
            Vopt = max(self.getQ([newState]))
            target = (reward + gamma * Vopt)
            if get_info(newState)['life'] == 0:
                info = get_info(self.statecache[-1])
                distance = info['distance']
                time = info['time']
                target = time-400-distance/2.0
            Y.append(target)
            print 'target: {}'.format(target)
            self.model.update_weights(tiles, infos, actions, Y)

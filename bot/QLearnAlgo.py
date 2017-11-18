from numpy import random
import operator
from util import *


class QLearningAlgorithm():
    def __init__(self, options, actions, discount, featureExtractor, windowsize):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.updateInterval = 10 # number of frames to retrain the model
        self.updateCounter = 0
        self.batchSize = 5
        self.windowsize = windowsize  # number of frames to look back in a state
        self.statecache = [[]] # list of states for each game
        self.actioncache = [[]] # list of actions for each game
        self.options = options
        self.model = None

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
        if self.options.isTrain:
            if self.windowsize > 1:
                probs = self.getProb(self.statecache[-1][-self.windowsize + 1:] + [state])
                actionIdx = random.choice(range(len(self.actions)),
                                          p=probs)
                print "randomly select action id: {}".format(actionIdx)
                print "Probs: {}".format(probs)
            else:
                probs = self.getProb([state])
                actionIdx = random.choice(range(len(self.actions)), p=probs)
                print "randomly select action id: {}".format(actionIdx)
                print "Probs: {}".format(probs)
        elif len(self.statecache[-1]) < self.windowsize:
            actionIdx = self.actions.index('Right')
        else:
            if self.windowsize > 1:
                actionIdx, q = max(enumerate(self.getQ(self.statecache[-1][-self.windowsize + 1:] + [state])),
                                   key=operator.itemgetter(1))
                print "Q: {} best action id: {}".format(q, actionIdx)
            else:
                actionIdx, q = max(enumerate(self.getQ([state])), key=operator.itemgetter(1))
                print "Q: {} best action id: {}".format(q, actionIdx)
        return Action.act(self.actions[actionIdx]), actionIdx

    # Call this function to get the step size to update the weights.

    def sample(self):
        gameIdx = random.randint(0, len(self.statecache)) 
        gameFrames = self.statecache[gameIdx]
        if len(gameFrames) < self.windowsize:
            return self.sample()
        frameIdx = random.randint(0, len(gameFrames) - self.windowsize)
        window = gameFrames[frameIdx:frameIdx + self.windowsize]
        tile, info = self.featureExtractor(window)
        action = self.actions[gameIdx][frameIdx]
    # We will call this function with (s, a, r, s'), which you should use to update |weights|.
    # Note that if s is a terminal state, then s' will be None.  Remember to check for this.
    # You should update the weights using self.getStepSize(); use
    # self.getQ() to compute the current estimate of the parameters.
    def incorporateFeedback(self, action_idx, newState):
        # BEGIN_YOUR_CODE (our solution is 12 lines of code, but don't worry if you deviate from this)
        self.updateCounter += 1

        if self.updateCounter >= self.updateInterval:
            self.updateCounter = 0
            gamma = self.discount

            tiles = []
            infos = []
            actions = []
            Y = []
            for _ in range(self.batchSize):
                gameIdx = random.randint(0, len(self.statecache))
                states = self.statecache[gameIdx]
                frameIdx = random.randint(1, len(states))
                window = self.statecache[-1][-self.windowsize-frameIdx:-frameIdx]
                if len(window) < self.windowsize: continue
                tile, info = self.featureExtractor(window)
                tiles.append(tile)
                infos.append(info)
                actions.append(action_idx)
                reward = get_reward(self.statecache[gameIdx][-frameIdx])
                if get_info(newState)['life'] == 0:
                    reward = -100000
                Vopt = max(self.getQ([newState]))
                target = (reward + gamma * Vopt)
                Y.append(target)

            print('incorporateFeedback batchSize={}'.format(len(Y)))

            print('tiles', len(tiles))
            print('info', len(info))
            print('actions', len(actions))
            self.model.update_weights(tiles,infos, actions, Y)

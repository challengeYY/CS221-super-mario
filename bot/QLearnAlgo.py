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
        self.batchSize = 20
        self.windowsize = windowsize  # number of frames to look back in a state
        self.statecache = [[]] # list of states for each game
        self.actioncache = [[]] # list of actions for each game
        self.options = options
        self.model = None

    def set_model(self, model):
        self.model = model

    # Return the Q function associated with the weights and features
    def getQ(self, window):
        if self.model.conv:
            tile, info = self.featureExtractor(window)
            scores = self.model.inference_Q([info], tile=[tile])[0]
        else:
            info = self.featureExtractor(window)
            if info is None:
                return [0]
            scores = self.model.inference_Q([info])[0]
        return scores

    def getProb(self, window):
        if self.model.conv:
            tile, info = self.featureExtractor(window)
            scores = self.model.inference_Prob([info], tile=[tile])[0]
        else:
            info = self.featureExtractor(window)
            if info is None:
                return [1.0/len(self.actions)] * len(self.actions)
            scores = self.model.inference_Prob([info])[0]
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
        # randomly choose a game
        gameIdx = random.randint(0, len(self.statecache)) 
        # Should have cache of s0, a0, ....., sn, an, sn+1, where reward of an is stored in sn+1
        gameFrames = self.statecache[gameIdx]
        gameActions = self.actioncache[gameIdx]
        assert(len(gameFrames) == len(gameActions) + 1)

        if len(gameActions) < self.windowsize:
            return self.sample() # resample a different game

        # randomly choose index for last frame in the window 
        if len(gameActions) == self.windowsize:
            frameIdx = len(gameActions)
        else:
            frameIdx = random.randint(self.windowsize, len(gameActions))

        window = gameFrames[frameIdx - self.windowsize:frameIdx]
        if self.model.conv:
            tile, info = self.featureExtractor(window)
        else:
            tile = None
            info = self.featureExtractor(window)
        action = gameActions[frameIdx]

        frame_np1 = gameFrames[frameIdx+1]
        if get_info(frame_np1)['life'] == 0:
            reward = -100000
        else:
            reward = get_reward(frame_np1)
        window_np1 = gameFrames[frameIdx - self.windowsize + 1 : frameIdx + 1]
        Vopt = max(self.getQ(window_np1))
        gamma = self.discount
        target = (reward + gamma * Vopt)

        return tile, info, action, target

    # once a while train the model
    def incorporateFeedback(self):
        if not self.options.isTrain: return

        self.updateCounter += 1
        if self.updateCounter < self.updateInterval:
            return
        self.updateCounter = 0

        print('incorporateFeedback ...')
        tiles = [] # a list of None if self.mode.conv is False
        infos = []
        actions = []
        target_Qs = []
        for i in range(self.batchSize):
            tile, info, action, target = self.sample()
            tiles.append(tile)
            infos.append(info)
            actions.append(action)
            target_Qs.append(target)

        self.model.update_weights(tiles=tiles, infos=infos, actions=actions, target_Qs=target_Qs)

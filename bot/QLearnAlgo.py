from numpy import random
import operator
from util import *


class QLearningAlgorithm():
    def __init__(self, options, actions, discount, featureExtractor):
        self.actions = actions
        self.discount = discount
        self.featureExtractor = featureExtractor
        self.updateInterval = 10 # number of frames to retrain the model
        self.updateCounter = 0
        self.batchSize = 20
        self.statecache = [[]] # list of states for each game. A state is a window of frames
        self.actioncache = [[]] # list of actions for each game
        self.options = options
        self.model = None
        self.explorationProb = 0.2

    def set_model(self, model):
        self.model = model

    # Return the Q function associated with the weights and features
    def getQ(self, state):
        if self.model.conv:
            tile, info = self.featureExtractor(state)
            scores = self.model.inference_Q([info], tile=[tile])[0]
        else:
            info = self.featureExtractor(state)
            if info is None:
                return [0]
            scores = self.model.inference_Q([info])[0]
        return scores

    def getProb(self, state):
        if self.model.conv:
            tile, info = self.featureExtractor(state)
            scores = self.model.inference_Prob([info], tile=[tile])[0]
        else:
            info = self.featureExtractor(state)
            if info is None:
                return [1.0/len(self.actions)] * len(self.actions)
            scores = self.model.inference_Prob([info])[0]
        return scores

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        actionIdx = 0
        if self.options.isTrain:
            rand = random.random()
            if rand < self.explorationProb:
                probs = [1.0 / len(self.actions)] * len(self.actions) # uniform probability 
            else:
                probs = self.getProb(state) # soft max prob
            actionIdx = random.choice(range(len(self.actions)),
                                      p=probs)
            print "randomly select action id: {}".format(actionIdx)
            print "Probs: {}".format(probs)
        else:
            actionIdx, q = max(enumerate(self.getQ(state)), key=operator.itemgetter(1))
            print "Q: {} best action id: {}".format(q, actionIdx)
        return Action.act(self.actions[actionIdx]), actionIdx

    # Call this function to get the step size to update the weights.

    def sample(self):
        # randomly choose a game
        gameIdx = random.randint(0, len(self.statecache)) 
        # Should have cache of s0, a0, ....., sn, an, sn+1, where reward of an is stored in sn+1
        gameStates = self.statecache[gameIdx]
        gameActions = self.actioncache[gameIdx]

        # randomly choose a state except last one in the game
        stateIdx = random.randint(0, len(gameStates)-1) if len(gameStates) > 1 else 0

        state_n = gameStates[stateIdx]
        if self.model.conv:
            tile, info = self.featureExtractor(state_n)
        else:
            tile = None
            info = self.featureExtractor(state_n)
        action = gameActions[stateIdx]

        state_np1 = gameStates[stateIdx+1]
        reward = get_reward(state_np1[-1])
        Vopt = max(self.getQ(state_np1))
        gamma = self.discount
        target = (reward + gamma * Vopt)
        if get_info(state_np1)['life'] == 0:
            info = get_info(state_np1)
            distance = info['distance']
            time = info['time']
            target = time-400-distance/2.0

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

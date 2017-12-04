from numpy import random
import operator
from util import *


class QLearningAlgorithm():
    def __init__(self, options, actions, discount, featureExtractor):
        # hyper parameter
        self.updateInterval = options.updateInterval 
        self.updateTargetInterval = options.updateTargetInterval
        self.batchSize = options.batchSize 
        self.batchPerFeedback = options.batchPerFeedback
        self.maxCache = options.maxCache
        self.explorationProb = options.explorationProb

        # inits
        self.actions = actions
        self.options = options
        self.featureExtractor = featureExtractor
        self.discount = discount
        self.updateCounter = 0
        self.updateTargetCounter = 0
        self.statecache = [[]]  # list of states for each game. A state is a window of frames
        self.actioncache = [[]]  # list of actions for each game
        self.model = None
        self.softmaxExplore = options.softmaxExploration

    # reset after each game iteration
    def reset(self):
        self.actioncache.append([])
        self.statecache.append([])
        if self.options.isTrain and self.explorationProb >= 0.05:
            self.explorationProb = self.explorationProb * 0.8
        if len(self.statecache) > self.maxCache:
            self.actioncache.pop(0)
            self.statecache.pop(0)

    def set_model(self, model):
        self.model = model

    # Return the Q function associated with the weights and features
    def getQ(self, vs, state):
        if self.model.conv:
            tile, info = self.featureExtractor(state)
            info = info.values()
            scores = self.model.inference_Q(vs, [info], tile=[tile])[0]
        else:
            info = self.featureExtractor(state)
            if info is None:
                return [0]
            info = info.values()
            scores = self.model.inference_Q(vs, [info])[0]
        return scores

    def getProb(self, state):
        if self.model.conv:
            tile, info = self.featureExtractor(state)
            info = info.values()
            scores = self.model.inference_Prob([info], tile=[tile])[0]
        else:
            info = self.featureExtractor(state)
            if info is None:
                return [1.0 / len(self.actions)] * len(self.actions)
            info = info.values()
            scores = self.model.inference_Prob([info])[0]
        return scores

    def formatQ(self, Q):
        info = 'Q: '
        for i, q in enumerate(Q):
            info += '{}={:.2f}, '.format('_'.join(self.actions[i][0]) + '_' +
                    str(self.actions[i][1]), q) 
        return info

    # This algorithm will produce an action given a state.
    # Here we use the epsilon-greedy algorithm: with probability
    # |explorationProb|, take a random action.
    def getAction(self, state):
        actionIdx = 0
        if self.options.isTrain:
            rand = random.random()
            if rand < self.explorationProb:
                actionIdx = random.choice(range(len(self.actions)))
                print "randomly select action: {}".format(self.actions[actionIdx])
            else:
                if self.softmaxExplore:
                    prob = self.getProb(state)
                    actionIdx = random.choice(range(len(self.actions)), p=prob)
                    print "Prob: {} selected action: {}".format(prob, self.actions[actionIdx])
                else:
                    q = self.getQ(self.model.prediction_vs, state)
                    actionIdx, _ = max(enumerate(q), key=operator.itemgetter(1))
                    print self.formatQ(q)
                    print "Max action: {}".format(self.actions[actionIdx])
        # probs = self.getProb(state)  # soft max prob
        #    actionIdx = random.choice(range(len(self.actions)),
        #                              p=probs)
        #    print "randomly select action id: {}".format(actionIdx)
        #    print "Probs: {}".format(probs)
        else:
            q = self.getQ(self.model.prediction_vs, state)
            actionIdx, _ = max(enumerate(q), key=operator.itemgetter(1))
            print self.formatQ(q)
            print "Max action: {}".format(self.actions[actionIdx])

        # info = self.featureExtractor(state)
        # show = ['pit_ahead', 'ahead_1_height', , 'ahead_2_height']
        # for k in show:
            # print(k, info[k])
        return self.actions[actionIdx], actionIdx

    # Call this function to get the step size to update the weights.

    def sample(self, sampleSize):
        # randomly choose a game and get its states
        samples = []
        gameLenSum = sum([len(a) for a in self.actioncache])
        gameProb = [len(a) * 1.0 / gameLenSum for a in self.actioncache]
        for i in range(sampleSize):
            gameIdx = random.choice(range(0, len(self.statecache)), p=gameProb)
            # Should have cache of s0, a0, ....., sn, an, sn+1, where reward of an is stored in sn+1
            gameStates = self.statecache[gameIdx]
            gameActions = self.actioncache[gameIdx]

            if len(gameActions) == 0:
                return self.sample()  # resample a different game

            # randomly choose a state except last one in the game
            stateIdx = random.randint(0, len(gameActions)) if len(gameActions) > 1 else 0

            state_n = gameStates[stateIdx]
            if self.model.conv:
                tile, info = self.featureExtractor(state_n)
                info = info.values()
            else:
                tile = None
                info = self.featureExtractor(state_n)
                info = info.values()
            action = gameActions[stateIdx]

            state_np1 = gameStates[stateIdx + 1]
            reward = state_np1.get_last_frame().get_reward()
            Vopt = max(self.getQ(self.model.target_vs, state_np1))
            gamma = self.discount
            target = (reward + gamma * Vopt)
            if state_np1.get_last_frame().get_is_finished():
                target = reward

            samples.append((tile, info, action, target))

        return samples

    # once a while train the model
    def incorporateFeedback(self):
        if not self.options.isTrain: return

        self.updateCounter += 1
        if self.updateCounter < self.updateInterval:
            return
        self.updateCounter = 0

        print('incorporateFeedback ...')
        for i in range(self.batchPerFeedback):
            tiles = []  # a list of None if self.mode.conv is False
            infos = []
            actions = []
            target_Qs = []
            samples = self.sample(self.batchSize)
            for tile, info, action, target in samples:
                tiles.append(tile)
                infos.append(info)
                actions.append(action)
                target_Qs.append(target)

            self.model.update_weights(tiles=tiles, infos=infos, actions=actions, target_Qs=target_Qs)

        self.updateTargetCounter += 1
        if self.updateTargetCounter < self.updateTargetInterval:
            return
        self.updateTargetCounter = 0
        print('Updating Target Network ...')
        self.model.update_target_network()

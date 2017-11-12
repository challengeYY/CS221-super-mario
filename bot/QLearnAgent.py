import traceback
from agent import *
from time import sleep
import numpy as np
from enum import *
from util import *

class QLearnAgent(Agent):
    def __init__(self, options, env):
        self.action = [0,0,0,0,0,0]
        self.state = None
        self.maxGameIter = options.maxGameIter
        self.gameIter = 0
        self.isTrain = options.isTrain
        self.model = #TODO
        self.env = env
        self.algo = QLearningAlgorithm(
                options = options,
                actions = self.get_possible_actions, 
                discount = 1,
                featureExtractor = self.featureExtractor,
                model = self.model,
                explorationProb = 0.2
                )

    def get_possible_actions(self, state):
        return Action.NAME + [Action.NO_ACTION]

    def featureExtractor(self, window, action):
        pass

    def initAction(self):
        return self.action

    def act(self, obs, reward, is_finished, info):
        self.state = (obs, reward, is_finished, info)

        if len(self.algo.statecache) >= 1:
            prevState = self.algo.statecache[-1]
            prevAction = self.algo.action[-1]
            prevReward = self.algo.rewardcache[-1]
            self.algo.incorporateFeedback(prevState, prevAction, prevReward, self.state)

        action = self.getAction(self.state)

        self.logAction()
        return self.action

    def exit(self):
        if self.state is None:
            return False
        self.gameIter += 1
        self.env.reset()

        (obs, reward, is_finished, info) = self.state
        total_score = info["distance"]
        # TODO: if stay at the same place for too long
        stop = total_score > 32000

        reachMaxIter = self.gameIter == self.maxGameIter

        exit = (reachMaxIter and self.isTrain) or not self.isTrain
        exit = exit and (is_finished or stop)
        return exit 

    def handle(self, e):
        print('encountering error, exiting ...')
        traceback.print_exc()
        exit(-1)


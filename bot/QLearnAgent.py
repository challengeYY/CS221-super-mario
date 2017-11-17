import traceback
from agent import *
from time import sleep
import numpy as np
from enum import *
from util import *
from QLearnAlgo import *

class QLearnAgent(Agent):
    def __init__(self, options, env):
        self.action = [0, 0, 0, 0, 0, 0]
        self.state = None
        self.maxGameIter = options.maxGameIter
        self.windowsize = options.windowsize
        self.gameIter = 0
        self.isTrain = options.isTrain
        self.env = env
        self.actions = ['Left', 'Right', 'A', ['Right', 'A'], ['Right', 'B'], ['Right', 'A', 'B']]
        self.algo = QLearningAlgorithm(
            options=options,
            actions=self.actions,
            discount=1,
            featureExtractor=self.featureExtractor,
            windowsize=self.windowsize,
        )

    def featureExtractor(self, window, action):
        raise Exception('Abstract method! should be overridden')

    def initAction(self):
        return self.action

    def act(self, obs, reward, is_finished, info):
        self.state = (obs, reward, is_finished, info)

        if len(self.algo.statecache) >= 1 and self.isTrain:
            # prevState = self.algo.statecache[-1]
            prevAction = self.algo.actioncache[-1]
            self.algo.incorporateFeedback(prevAction, self.state)

        if is_finished:
            self.action = Action.empty()
        else:
            self.action, action_idx = self.algo.getAction(self.state)
            # caching states
            self.algo.statecache.append(self.state)
            # names can be a list
            self.algo.actioncache.append(action_idx)

        self.log(self.action, reward)
        return self.action

    def exit(self):
        if self.state is None:
            return False
        if is_finished(self.state):
            self.gameIter += 1
            self.env.reset()
            self.algo.actioncache = []
            self.algo.statecache = []
            self.algo.batchcounter = 0

        info = get_info(self.state)
        stuck = False
        if 'ignore' in info:
            stuck = info['ignore']

        reachMaxIter = self.gameIter >= self.maxGameIter

        exit = reachMaxIter
        exit = exit and (is_finished or stuck)
        return exit

    def handle(self, e):
        print('encountering error, exiting ...')
        traceback.print_exc()
        exit(-1)

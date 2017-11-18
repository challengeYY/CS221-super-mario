import traceback
from agent import *
from time import sleep
import numpy as np
from enum import *
from util import *
from QLearnAlgo import *

class QLearnAgent(Agent):
    def __init__(self, options, env):
        self.action = Action.empty() 
        self.state = None
        self.maxGameIter = options.maxGameIter
        self.windowsize = options.windowsize
        self.gameIter = 0
        self.isTrain = options.isTrain
        self.env = env
        self.actions = ['Left', 'Right', 'A', ['Right', 'A'], ['Right', 'B'], ['Right', 'A', 'B'], ['Left', 'A'],
                        ['Left', 'B'], ['Left', 'A', 'B']]
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
        # actionIdx = self.actions.index('Right')
        # self.algo.actioncache[-1].append(action_idx)
        # self.action = Action.act(self.actions[actionIdx])
        return self.action

    def act(self, obs, reward, is_finished, info):
        self.state = (obs, reward, is_finished, info)

        # caching new state
        self.algo.statecache[-1].append(self.state)

        self.algo.incorporateFeedback()

        # get new action
        if is_finished:
            self.action = Action.empty()
        else:
            # get and cache new action 
            self.action, action_idx = self.algo.getAction(self.state)
            self.algo.actioncache[-1].append(action_idx)

        self.log(self.action, reward)
        return self.action

    def exit(self):
        if self.state is None:
            return False
        if is_finished(self.state):
            self.gameIter += 1
            self.env.reset()
            print('statecache {}'.format(len(self.algo.statecache[-1])))
            print('action {}'.format(len(self.algo.actions[-1])))
            self.algo.actioncache.append([])
            self.algo.statecache.append([])

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

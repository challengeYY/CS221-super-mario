import traceback
from agent import *
from time import sleep
import numpy as np
from enum import *
from util import *
from QLearnAlgo import *

class QLearnAgent(Agent):
    def __init__(self, options, env):
        self.action = Action.act('Right')
        self.frame = None
        self.maxGameIter = options.maxGameIter
        self.windowsize = options.windowsize
        self.framecache = [] # list of frames for each game, cleared at the end of the game
        self.gameIter = 0
        self.isTrain = options.isTrain
        self.env = env
        self.actions = ['Left', 'Right', 'A', ['Right', 'A'], ['Right', 'B'], ['Right', 'A', 'B'], ['Left', 'A'],
                        ['Left', 'B'], ['Left', 'A', 'B']]
        self.algo = QLearningAlgorithm(
            options=options,
            actions=self.actions,
            discount=1,
            featureExtractor=self.featureExtractor
        )
        self.stepCounter = 0
        self.stepCounterMax = 5

    def featureExtractor(self, window, action):
        raise Exception('Abstract method! should be overridden')

    def initAction(self):
        return self.action

    def act(self, obs, reward, is_finished, info):
        self.frame = (obs, reward, is_finished, info)

        # caching new frame
        self.framecache.append(self.frame)
        if len(self.framecache) > (self.windowsize + 3):
            self.framecache.pop(0) # remove frame outside window to save memory

        # only update state and action once a while
        self.stepCounter += 1 
        if self.stepCounter < self.stepCounterMax:
            if is_finished:
                state = self.framecache[-self.windowsize:]
                self.algo.statecache[-1].append(state)
            return self.action

        # if not enough frame for a window, keep playing 
        if len(self.framecache) < self.windowsize:
            return self.action

        self.stepCounter = 0

        # caching new state
        state = self.framecache[-self.windowsize:]
        self.algo.statecache[-1].append(state)

        self.algo.incorporateFeedback()

        # get new action
        if is_finished:
            self.action = Action.empty()
        else:
            # get and cache new action 
            self.action, action_idx = self.algo.getAction(state)
            self.algo.actioncache[-1].append(action_idx)

        self.log(self.action, reward)
        return self.action

    def exit(self):
        if self.frame is None:
            return False
        if is_finished(self.frame):
            self.gameIter += 1
            self.env.reset()
            self.framecache = []
            self.algo.actioncache.append([])
            self.algo.statecache.append([])

        info = get_info(self.frame)
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

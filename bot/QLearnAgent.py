import traceback
from agent import *
from time import sleep
import numpy as np
from enum import *
from util import *
import pickle
from QLearnAlgo import *
import os


class QLearnAgent(Agent):
    def __init__(self, options, env):

        # hyper parameter
        self.maxGameIter = options.maxGameIter
        self.windowsize = options.windowsize
        self.prevActionsSize = options.prevActionsSize
        self.stepCounterMax = options.stepCounterMax

        # inits
        self.options = options
        self.frame = None
        self.framecache = []  # list of frames for each game, cleared at the end of the game
        self.gameIter = 0
        self.bestScore = 0
        self.isTrain = options.isTrain
        self.env = env
        self.init_actions()
        self.actionQueue = []
        self.algo = QLearningAlgorithm(
            options=options,
            actions=self.actions,
            discount=0.9,
            featureExtractor=self.featureExtractor
        )
        self.stepCounter = 0
        self.totalReward = 0
        self.score_log_file = options.model_dir + "/score_log"

    def init_actions(self):
        options = self.options
        action_path = options.model_dir + '/action.pickle'
        stepMax = self.options.stepCounterMax
        if self.options.isTrain:
            self.actions = [
                GameAction([[Action.NO_ACTION]]), 
                GameAction([['Right', 'A']] * 3), 
                GameAction([['Right', 'A']] * 1), 
                GameAction([['Right', 'B']] * 3), 
                GameAction([['Right', 'B']] * 1), 
                GameAction([['Right', 'A', 'B']] * 6),
                GameAction([['Left', 'A']] * 3), 
                GameAction([['Left', 'B']] * 2),
            ]
            pickle.dump(self.actions, open(action_path, 'wb'))
        else:
            if not os.path.isfile(action_path):
                print('No action stored in {}'.format(action_path))
                exit(-1)
            else:
                self.actions = pickle.load(open(action_path, 'rb'))

    def featureExtractor(self, window, action):
        raise Exception('Abstract method! should be overridden')

    def nextAction(self):
        if len(self.actionQueue) == 0:
            return [Action.NO_ACTION]
        else:
            return self.actionQueue.pop(0)

    def initAction(self):
        return Action.act(self.nextAction())

    def cacheState(self):
        frames = self.framecache[-self.windowsize:]
        prevActions = self.algo.actioncache[-1][-self.prevActionsSize:] 
        last_frame = frames[-1]
        last_frame = last_frame.set_reward(self.totalReward)
        state = GameState(frames[:-1] + [last_frame], prevActions[:])
        self.totalReward = 0
        self.algo.statecache[-1].append(state)
        print('reward:', state.get_last_frame().get_reward())
        return state

    def cacheNewAction(self, is_finished, state):
        if is_finished:
            self.actionQueue = [[Action.NO_ACTION]]
        else:
            # get and cache new action 
            action_idx = self.algo.getAction(state)
            self.actionQueue = self.actions[action_idx].get_actions()
            self.algo.actioncache[-1].append(action_idx)

    def cacheFrame(self):
        # caching new frame
        self.framecache.append(self.frame)
        if len(self.framecache) > self.windowsize + 10: # slightly larger for look at stuck_frames
            self.framecache.pop(0)  # remove frame outside window to save memory

    def is_stuck(self):
        stuck_frames = 5
        if len(self.framecache)>=stuck_frames:
            frames = self.framecache[-stuck_frames:]
            return get_stuck(frames)
        return False
    
    def calcReward(self, reward, is_finished, info):
        # Customized reward
        # if stuck at the same location, small negative reward. Increase exploration probability
        if self.is_stuck():
            self.algo.explorationProb *= 1.2
            return -0.5
        # if dead reward = -10
        if is_finished and info['distance'] < 3250:
            return self.options.death_penalty # dead reward
        self.totalReward += reward
        return reward

    def act(self, obs, reward, is_finished, info):

        reward = self.calcReward(reward, is_finished, info)

        self.frame = GameFrame(np.copy(obs), reward, is_finished, info.copy())

        self.cacheFrame()

        # only update state and action once a while
        self.stepCounter += 1
        if self.stepCounter < self.stepCounterMax:
            if is_finished:
                self.cacheState()
            return Action.act(self.nextAction())

        # if not enough frame for a window, keep playing 
        if len(self.framecache) < self.windowsize:
            return Action.act(self.nextAction()) 

        self.stepCounter = 0

        # caching new state
        state = self.cacheState()

        self.algo.incorporateFeedback()

        # get new action
        self.cacheNewAction(is_finished, state)

        return Action.act(self.nextAction()) 

    def exit(self):
        if self.frame is None:
            return False
        if self.frame.get_is_finished():
            frame_info = self.frame.get_info()
            if 'distance' in frame_info:
                distance = frame_info['distance']
                if self.bestScore < distance:
                    self.bestScore = distance
                    print "Best Score: {}".format(self.bestScore)
                with open(self.score_log_file,'a+') as score_log :
                    score_log.write("{}\n".format(distance))
                print "Score: {}".format(distance)
            self.gameIter += 1
            self.env.reset()
            self.framecache = []
            self.totalReward = 0
            self.algo.reset()

        info = self.frame.get_info()
        stuck = False
        if 'ignore' in info:
            stuck = info['ignore']

        reachMaxIter = self.gameIter >= self.maxGameIter

        if reachMaxIter and self.options.isTrain:
            self.algo.model.save_model()

        exit = reachMaxIter
        exit = exit and (self.frame.get_is_finished() or stuck)
        return exit

    def handle(self, e):
        print('encountering error, exiting ...')
        traceback.print_exc()
        exit(-1)

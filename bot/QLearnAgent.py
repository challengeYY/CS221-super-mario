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
        self.action = Action.act('Right')
        self.framecache = []  # list of frames for each game, cleared at the end of the game
        self.gameIter = 0
        self.bestScore = 0
        self.isTrain = options.isTrain
        self.env = env
        self.init_actions()
        self.algo = QLearningAlgorithm(
            options=options,
            actions=self.actions,
            discount=0.9,
            featureExtractor=self.featureExtractor
        )
        self.stepCounter = 0
        self.totalReward = 0
        self.actionCounter = 0
        self.score_log_file = options.model_dir + "/score_log"

    def init_actions(self):
        options = self.options
        action_path = options.model_dir + '/action.pickle'
        if self.options.isTrain:
            self.actions = [
                (['Right', 'A'], 6),
                (['Right', 'A'], 3),
                (['Right', 'B'], 3), 
                (['Right', 'B'], 1), 
                (['Right', 'A'], 1), 
                (['Right', 'A', 'B'], 2),
                (['Left', 'A'], 3), 
                (['Left', 'B'], 2),
                ([Action.NO_ACTION], 1), 
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

    def initAction(self):
        return self.action

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

    def cacheAction(self):
        self.algo.actioncache[-1].append(action_idx)

    def is_stuck(self):
        stuck_frames = 5
        if len(self.framecache)>=stuck_frames:
            frames = self.framecache[-stuck_frames:]
            get_stuck(frames)
        return False

    def act(self, obs, reward, is_finished, info):

        # Customized reward
        # if stuck at the same location, small negative reward. Increase exploration probability
        if self.is_stuck():
            reward = -0.5
            self.algo.explorationProb *= 1.2
        # if dead reward = -10
        if is_finished and info['distance'] < 3250:
            reward = self.options.death_penalty # dead reward

        self.frame = GameFrame(np.copy(obs), reward, is_finished, info.copy())
        self.totalReward += reward

        # caching new frame
        self.framecache.append(self.frame)

        if len(self.framecache) > self.windowsize + 10: # slightly larger for look at stuck_frames
            self.framecache.pop(0)  # remove frame outside window to save memory

        # only update state and action once a while
        self.stepCounter += 1
        if self.stepCounter < self.stepCounterMax:
            if is_finished:
                self.cacheState()
            if self.actionCounter <= 0:
                return Action.empty()
            else:
                self.actionCounter -= 1
                return self.action

        # if not enough frame for a window, keep playing 
        if len(self.framecache) < self.windowsize:
            return self.action

        self.stepCounter = 0

        # caching new state
        state = self.cacheState()

        self.algo.incorporateFeedback()

        # get new action
        if is_finished:
            self.action = Action.empty()
        else:
            # get and cache new action 
            (actionOption, self.actionCounter), action_idx = self.algo.getAction(state)
            self.action = Action.act(actionOption)
            self.algo.actioncache[-1].append(action_idx)

        return self.action

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

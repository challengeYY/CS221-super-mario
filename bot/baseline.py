from abc import ABCMeta, abstractmethod
import traceback
from agent import *
from time import sleep

class BaselineAgent(Agent):
    def __init__(self, options):
        self.action = [0,0,0,0,0,0]
        self.state = None

    def initAction(self):
        print('init act ...')
        return self.action

    def act(self, obs, reward, is_finished, info):
        self.state = (obs, reward, is_finished, info)
        print('acting {}'.format(self.action))
        sleep(1.0/30)
        return self.action

    def exit(self):
        if self.state is None:
            return False
        (obs, reward, is_finished, info) = self.state
        total_score = info["distance"]
        stop = total_score > 32000
        if is_finished or stop:
            if stop:
                print('exiting the game cuz I no longer wanna play ...')
            return True
        else:
            return False

    def handle(self, e):
        print('encountering error, exiting ...')
        traceback.print_exc()
        exit(-1)

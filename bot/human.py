from pynput import keyboard
from abc import ABCMeta, abstractmethod
import traceback
from agent import *
from time import sleep


class HumanAgent(Agent):
    def __init__(self, options):
        self.action = [0, 0, 0, 0, 0, 0]
        self.state = None
        # with keyboard.Listener(
        #    on_press=self.on_press,
        #    on_release=self.on_release) as listener:
        #    self.listener = listener

    def initAction(self):
        return self.action

    def act(self, obs, reward, is_finished, info):
        self.state = (obs, reward, is_finished, info)
        sleep(0.1)
        # print('acting {}'.format(self.action))
        # sleep(1.0/30)
        return self.action

    def exit(self):
        if self.state is None:
            return False
        (obs, reward, is_finished, info) = self.state
        total_score = info["distance"]
        stop = total_score > 32000
        if is_finished or stop:
            # self.listener.join()
            if stop:
                print('exiting the game cuz I no longer wanna play ...')
            return True
        else:
            return False

    def handle(self, e):
        # self.listener.join()
        print('encountering error, exiting ...')
        traceback.print_exc()
        exit(-1)

        # def on_press(self, key):
        #    if key.char == 'a':
        #        self.action[1] = 1
        #    elif key.char == 'd':
        #        self.action[3] = 1
        #    elif key.char == 'j':
        #        self.action[4] = 1
        #    elif key.char == 'k':
        #        self.action[5] = 1

        # def on_release(self, key):
        #    if key.char == 'a':
        #        self.action[1] = 0
        #    elif key.char == 'd':
        #        self.action[3] = 0
        #    elif key.char == 'j':
        #        self.action[4] = 0
        #    elif key.char == 'k':
        #        self.action[5] = 0
